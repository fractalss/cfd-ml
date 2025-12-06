import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers += [
                nn.Conv1d(c_in, c_out, 1),
                nn.BatchNorm1d(c_out),
                nn.ReLU(True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        feat_dims=(64, 128, 1024),
        latent_dim: int = 256,
    ):
        super().__init__()
        self.point_feat_dim = feat_dims[-1]
        self.mlp = SharedMLP([in_dim, *feat_dims])
        self.fc = nn.Sequential(
            nn.Linear(self.point_feat_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x : [B, N, in_dim]
        mask : [B, N] or [B, 1, N] or None
        """
        # [B, N, C] -> [B, C, N]
        x = x.transpose(1, 2)
        feat = self.mlp(x)  # [B, C_feat, N]

        if mask is None:
            # global max-pooling over points
            global_feat, _ = torch.max(feat, dim=2)  # [B, C_feat]
        else:
            m = mask.bool()
            if m.dim() == 2:
                m = m.unsqueeze(1)  # [B, 1, N]
            elif m.dim() == 3 and m.shape[1] != 1 and m.shape[1] != feat.shape[1]:
                # try to coerce to [B,1,N]
                if m.shape[1] == feat.shape[2]:
                    m = m[:, 0:1, :]
                else:
                    m = m[:, :1, :]
            m = m.expand_as(feat)  # [B, C_feat, N]
            feat_masked = feat.masked_fill(~m, float("-inf"))
            global_feat, _ = torch.max(feat_masked, dim=2)

        z = self.fc(global_feat)  # [B, latent_dim]

        # per-point features back to [B, N, C_feat] for decoder
        return z, feat.transpose(1, 2)


class PointNetDecoder(nn.Module):
    def __init__(
        self,
        point_feat_dim: int = 1024,
        in_stream_dim: int = 4,  # [x, y, z, field]
        latent_dim: int = 256,
        param_dim: int = 0,
        out_dim: int = 4,
    ):
        super().__init__()
        self.point_feat_dim = point_feat_dim
        self.in_stream_dim = in_stream_dim
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.out_dim = out_dim

        # concat( per_point_feat, x_dynamic, cond[z,params] )
        cat_dim = point_feat_dim + in_stream_dim + (latent_dim + param_dim)

        self.conv1 = nn.Conv1d(cat_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, 1)

    def forward(self, per_point_feat, x_in, z, params=None):
        """
        Parameters
        ----------
        per_point_feat : [B, N, C_feat]
            Per-point features from encoder.
        x_in : [B, N, in_stream_dim]
            Dynamic input stream (typically [x,y,z,field]) used as template.
        z : [B, latent_dim]
            Latent code.
        params : [B, P] or None
            Global conditioning parameters (e.g., [velocity, t_norm]).
        """
        B, N, _ = x_in.shape

        # build conditioning vector [z, params]
        if params is None:
            if self.param_dim > 0:
                zeros = torch.zeros(
                    B, self.param_dim, device=z.device, dtype=z.dtype
                )
                cond = torch.cat([z, zeros], dim=-1)  # [B, latent+P]
            else:
                cond = z  # [B, latent]
        else:
            cond = torch.cat([z, params], dim=-1)  # [B, latent+P]

        # tile cond to per-point shape
        cond = cond.unsqueeze(1).expand(B, N, cond.shape[-1])  # [B, N, latent+P]

        # only use dynamic channels from x_in
        x_dynamic = x_in[:, :, : self.in_stream_dim]  # [B, N, 4]

        # concat per-point features + x_dynamic + cond
        cat = torch.cat([x_dynamic, per_point_feat, cond], dim=-1)  # [B, N, *]
        cat = cat.transpose(1, 2)  # [B, C, N]

        y = F.relu(self.bn1(self.conv1(cat)))
        y = F.relu(self.bn2(self.conv2(y)))
        out = self.conv3(y)       # [B, out_dim, N]
        out = out.transpose(1, 2) # [B, N, out_dim]
        return out


class PointNetAutoencoder(nn.Module):
    """PointNet-style autoencoder for Lagrangian ROM.

    This variant is strictly defined on the 4 *dynamic* features:
        [x, y, z, field]

    Static identifiers like CloudId / CloudId_base are kept
    outside the network and reattached downstream.
    """

    def __init__(
        self,
        in_dim: int = 4,   # [x, y, z, field]
        out_dim: int = 4,  # reconstruct [x, y, z, field]
        latent_dim: int = 256,
        param_dim: int = 0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.param_dim = param_dim

        # Encoder sees only the dynamic features
        self.encoder = PointNetEncoder(in_dim=in_dim, latent_dim=latent_dim)

        # Decoder also operates only on the 4 dynamic channels
        self.decoder = PointNetDecoder(
            point_feat_dim=1024,
            in_stream_dim=in_dim,
            latent_dim=latent_dim,
            param_dim=param_dim,
            out_dim=out_dim,
        )

    def forward(self, x, mask=None, params=None):
        """Forward pass (used during AE training).

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, N, 4]; dynamic features [x, y, z, field].
        mask : torch.Tensor or None
            Optional validity mask [B, N] / [B, 1, N].
        params : torch.Tensor or None
            Optional conditioning parameters [B, P].
        """
        z, per_point_feat = self.encoder(x, mask=mask)       # z: [B,Z], feat: [B,N,C]
        recon = self.decoder(per_point_feat, x, z, params)   # [B,N,4]
        return recon, z

        # ---------- NEW: inference-time decode from latent + template ----------
        # ---------- NEW: inference-time decode from latent + template ----------

    def decode(self, z, template_x, params=None, mask=None):
        """
        Parameters
        ----------
        z : torch.Tensor
            [B, latent_dim] latent codes.
        template_x : torch.Tensor
            [B, N, 4] scaled dynamic template points [x,y,z,field].
        params : torch.Tensor or None
            [B, P] augmented params [physical..., t_norm] if used.
        mask : torch.Tensor or None
            Optional validity mask [B, N] / [B, 1, N].

        Returns
        -------
        recon : torch.Tensor
            [B, N, 4] reconstructed dynamics.
        """
        # Re-encode template to get per-point features (PointNet-style)
        _, per_point_feat = self.encoder(template_x, mask=mask)
        recon = self.decoder(per_point_feat, template_x, z, params)
        return recon


__all__ = [
    "SharedMLP",
    "PointNetEncoder",
    "PointNetDecoder",
    "PointNetAutoencoder",
]
