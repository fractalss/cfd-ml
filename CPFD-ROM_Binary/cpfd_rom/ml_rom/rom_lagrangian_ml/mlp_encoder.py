import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(c_in, c_out, 1), nn.BatchNorm1d(c_out), nn.ReLU(True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, C, N]
        return self.net(x)

class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, feat_dims=(64, 128, 1024), latent_dim=256):
        super().__init__()
        self.mlp = SharedMLP([in_dim, *feat_dims])
        self.fc = nn.Sequential(
            nn.Linear(feat_dims[-1], 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, mask=None):
        # x: [B, N, C]
        x = x.transpose(1, 2)  # -> [B, C, N]
        feat = self.mlp(x)     # [B, F, N]

        if mask is None:
            global_feat, _ = torch.max(feat, dim=2)  # [B, F]
        else:
            m = mask.unsqueeze(1).expand_as(feat)
            feat_masked = feat.masked_fill(~m, float('-inf'))
            global_feat, _ = torch.max(feat_masked, dim=2)

        z = self.fc(global_feat)  # [B, latent_dim]
        return z, feat.transpose(1, 2)

class PointNetDecoder(nn.Module):
    def __init__(self, point_feat_dim=1024, in_stream_dim=3, latent_dim=256, param_dim=0, out_dim=3):
        super().__init__()
        cat_dim = point_feat_dim + in_stream_dim + (latent_dim + param_dim)
        self.conv1 = nn.Conv1d(cat_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, 1)

    def forward(self, per_point_feat, x_in, z, params=None):
        B, N, _ = x_in.shape

        if params is None:
            cond = z
        else:
            cond = torch.cat([z, params], dim=-1)

        cond = cond.unsqueeze(1).expand(B, N, cond.shape[-1])

        cat = torch.cat([x_in, per_point_feat, cond], dim=-1)
        cat = cat.transpose(1, 2)

        y = F.relu(self.bn1(self.conv1(cat)))
        y = F.relu(self.bn2(self.conv2(y)))
        out = self.conv3(y).transpose(1, 2)

        return out

class PointNetAutoencoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, latent_dim=256, param_dim=0):
        super().__init__()
        self.encoder = PointNetEncoder(in_dim=in_dim, latent_dim=latent_dim)
        self.decoder = PointNetDecoder(point_feat_dim=1024, in_stream_dim=in_dim,
                                       latent_dim=latent_dim, param_dim=param_dim,
                                       out_dim=out_dim)

    def forward(self, x, mask=None, params=None):
        z, per_point_feat = self.encoder(x, mask=mask)
        recon = self.decoder(per_point_feat, x, z, params)
        return recon, z

__all__ = ["PointNetAutoencoder"]
