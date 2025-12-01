import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """
    Shared MLP implemented as a stack of 1x1 Conv1d + BN + ReLU layers.

    Expected input:  x: [B, C, N]
    Output:          x: [B, C_out, N]
    """
    def __init__(self, channels):
        super().__init__()
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(c_in, c_out, 1),
                       nn.BatchNorm1d(c_out),
                       nn.ReLU(True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, C, N]
        return self.net(x)


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder.

    in_dim:
        Number of per-point input features.
        For your Lagrangian ROM case, this should be:
            in_dim = 6 ? [x, y, z, field, CloudID, CloudID_base]

    Output:
        z:    [B, latent_dim]        (global latent / shape code)
        feat: [B, N, point_feat_dim] (per-point high-level features)
    """
    def __init__(self,
                 in_dim: int = 3,
                 feat_dims=(64, 128, 1024),
                 latent_dim: int = 256):
        super().__init__()
        self.point_feat_dim = feat_dims[-1]

        # Shared MLP over points
        self.mlp = SharedMLP([in_dim, *feat_dims])

        # Global feature ? latent vector
        self.fc = nn.Sequential(
            nn.Linear(self.point_feat_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, mask=None):
        """
        x:    [B, N, C]  (C = in_dim)
        mask: [B, N] or None. If provided, mask out invalid points.
        """
        # [B, N, C] ? [B, C, N]
        x = x.transpose(1, 2)

        # Shared MLP ? per-point features [B, F, N]
        feat = self.mlp(x)

        # Global max pooling over points (mask-aware if provided)
        if mask is None:
            global_feat, _ = torch.max(feat, dim=2)  # [B, F]
        else:
            m = mask.unsqueeze(1).expand_as(feat)    # [B, F, N]
            feat_masked = feat.masked_fill(~m, float('-inf'))
            global_feat, _ = torch.max(feat_masked, dim=2)

        z = self.fc(global_feat)                    # [B, latent_dim]

        # Return per-point features as [B, N, F]
        return z, feat.transpose(1, 2)


class PointNetDecoder(nn.Module):
    """PointNet-style decoder.

    Inputs:
        per_point_feat: [B, N, point_feat_dim]   (e.g. 1024)
        x_in:           [B, N, in_stream_dim]    (same as encoder in_dim)
            For your setup:
                x_in[..., :4]   ? [x, y, z, field]
                x_in[..., 4:6]  ? [CloudID, CloudID_base]
        z:              [B, latent_dim]
        params (opt):   [B, param_dim]

    Output:
        out:            [B, N, out_dim]

    For your case:
        in_stream_dim = 6
        out_dim       = 4   ? predicts [x, y, z, field] only
    """
    def __init__(self,
                 point_feat_dim: int = 1024,
                 in_stream_dim: int = 3,
                 latent_dim: int = 256,
                 param_dim: int = 0,
                 out_dim: int = 3):
        super().__init__()
        self.point_feat_dim = point_feat_dim
        self.in_stream_dim = in_stream_dim
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.out_dim = out_dim

        # Total channels after concatenation:
        # [per_point_feat, x_in, cond(z, params)]
        cat_dim = point_feat_dim + in_stream_dim + (latent_dim + param_dim)

        self.conv1 = nn.Conv1d(cat_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, 1)

    def forward(self, per_point_feat, x_in, z, params=None):
        """Decode per-point outputs from latent + per-point features.

        per_point_feat: [B, N, point_feat_dim]
        x_in:           [B, N, in_stream_dim]
        z:              [B, latent_dim]
        params:         [B, param_dim] or None

        Returns:
            out: [B, N, out_dim]
        """
        B, N, _ = x_in.shape

        # -----------------------------
        # Build conditioning vector of fixed size latent_dim + param_dim
        # -----------------------------
        if params is None:
            if self.param_dim > 0:
                # Zero-pad the missing parameter part so that the
                # conditioning channel count always matches the
                # conv1 expectation (latent_dim + param_dim).
                zeros = torch.zeros(B,
                                    self.param_dim,
                                    device=z.device,
                                    dtype=z.dtype)
                cond = torch.cat([z, zeros], dim=-1)  # [B, latent_dim + param_dim]
            else:
                cond = z  # [B, latent_dim]
        else:
            # Concatenate latent + parameters
            cond = torch.cat([z, params], dim=-1)     # [B, latent_dim + param_dim]

        # Expand cond over all points: [B, 1, C] ? [B, N, C]
        cond = cond.unsqueeze(1).expand(B, N, cond.shape[-1])

        # -----------------------------
        # Concatenate all streams
        # -----------------------------
        # [B, N, point_feat_dim + in_stream_dim + latent_dim + param_dim]
        cat = torch.cat([x_in, per_point_feat, cond], dim=-1)

        # Conv1d expects [B, C, N]
        cat = cat.transpose(1, 2)  # [B, C, N]

        # Decode
        y = F.relu(self.bn1(self.conv1(cat)))
        y = F.relu(self.bn2(self.conv2(y)))
        out = self.conv3(y)        # [B, out_dim, N]

        # Back to [B, N, out_dim]
        out = out.transpose(1, 2)
        return out


class PointNetAutoencoder(nn.Module):
    """
    Complete PointNet-style autoencoder wrapper.

    For your Lagrangian ROM, you should instantiate as:

        model = PointNetAutoencoder(
            in_dim=6,     # [x, y, z, field, CloudID, CloudID_base]
            out_dim=4,    # [x, y, z, field]
            latent_dim=256,
            param_dim=0   # or >0 if you use operating conditions as extra inputs
        )

    The training loss should be applied ONLY on the first 4 channels
    of the input (dynamic features), not on CloudID / CloudID_base.
    """
    def __init__(self,
                 in_dim: int = 3,
                 out_dim: int = 3,
                 latent_dim: int = 256,
                 param_dim: int = 0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.param_dim = param_dim

        self.encoder = PointNetEncoder(in_dim=in_dim,
                                       latent_dim=latent_dim)

        self.decoder = PointNetDecoder(point_feat_dim=1024,
                                       in_stream_dim=in_dim,
                                       latent_dim=latent_dim,
                                       param_dim=param_dim,
                                       out_dim=out_dim)

    def forward(self, x, mask=None, params=None):
        """
        x:      [B, N, in_dim]
        mask:   [B, N] or None
        params: [B, param_dim] or None
        """
        z, per_point_feat = self.encoder(x, mask=mask)
        recon = self.decoder(per_point_feat, x, z, params)
        return recon, z


__all__ = ["PointNetAutoencoder"]
