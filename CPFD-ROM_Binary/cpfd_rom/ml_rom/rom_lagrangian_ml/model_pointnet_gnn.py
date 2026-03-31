# cpfd_rom/ml_rom/rom_lagrangian_ml/model_pointnet_gnn.py

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_max_pool


def make_mlp(channels, dropout: float = 0.0, use_bn: bool = True):
    layers = []
    for i in range(len(channels) - 1):
        in_ch = channels[i]
        out_ch = channels[i + 1]
        layers.append(nn.Linear(in_ch, out_ch))
        if i < len(channels) - 2:
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.edge_conv = EdgeConv(
            nn=make_mlp(
                [2 * in_channels, out_channels, out_channels],
                dropout=dropout,
                use_bn=True,
            ),
            aggr="max",
        )

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


class PointNetGNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        num_gnn_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.local_mlp = make_mlp(
            [in_channels, 64, 128, hidden_channels],
            dropout=dropout,
            use_bn=True,
        )

        self.gnn_layers = nn.ModuleList([
            EdgeConvBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                dropout=dropout,
            )
            for _ in range(num_gnn_layers)
        ])

        self.post_gnn = make_mlp(
            [hidden_channels, hidden_channels, hidden_channels],
            dropout=dropout,
            use_bn=True,
        )

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, latent_dim),
        )

    def forward(self, x, edge_index, batch):
        h = self.local_mlp(x)
        for gnn in self.gnn_layers:
            h = h + gnn(h, edge_index)
        h = self.post_gnn(h)
        hg = global_max_pool(h, batch)
        z = self.to_latent(hg)
        return z


class ConditionalDecoder(nn.Module):
    """
    Decoder predicts:
      - delta_xyz : additive correction to template xyz
      - field_raw : scalar logit/raw value to be constrained later

    Output channels are still 4:
      [dx, dy, dz, field_raw]
    """
    def __init__(
        self,
        template_in_channels: int,
        latent_dim: int,
        cond_dim: int,
        hidden_channels: int = 128,
        out_channels: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        if out_channels < 4:
            raise ValueError(
                f"ConditionalDecoder expects out_channels >= 4 for [dx,dy,dz,field], got {out_channels}"
            )

        self.decoder = make_mlp(
            [
                template_in_channels + latent_dim + cond_dim,
                hidden_channels,
                hidden_channels,
                hidden_channels,
                out_channels,
            ],
            dropout=dropout,
            use_bn=True,
        )

    def forward(self, z, template_x, batch, cond):
        z_expanded = z[batch]
        cond_expanded = cond[batch]
        dec_in = torch.cat([template_x, z_expanded, cond_expanded], dim=1)
        out = self.decoder(dec_in)
        return out


class PointNetGNNAutoencoder(nn.Module):
    """
    Contract
    --------
    - forward(x, edge_index, batch, params) -> (recon_x, graph_latent)
    - decode(latent_z, template_x, edge_index, batch, params) -> recon_x
    - graph_latent is PRE-fuse and is the target for latent regression

    Important behavior
    ------------------
    This version predicts:
      [dx, dy, dz, field]
    internally, then reconstructs:
      xyz_pred = xyz_template + dxyz
      field_pred = sigmoid(field_raw)

    So the final returned recon_x is still:
      [x, y, z, field]
    """
    def __init__(
        self,
        in_dim: int = 4,
        param_dim: int = 0,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 4,
        num_gnn_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        if in_dim < 4:
            raise ValueError(f"Expected in_dim >= 4 for [x,y,z,field], got {in_dim}")
        if out_dim < 4:
            raise ValueError(f"Expected out_dim >= 4 for [x,y,z,field], got {out_dim}")

        self.in_dim = in_dim
        self.param_dim = param_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.encoder = PointNetGNNEncoder(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            latent_dim=latent_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
        )

        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + param_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = ConditionalDecoder(
            template_in_channels=in_dim,
            latent_dim=latent_dim,
            cond_dim=param_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            dropout=dropout,
        )

    def _compose_output_from_template(self, raw_out: torch.Tensor, template_x: torch.Tensor) -> torch.Tensor:
        """
        Convert decoder raw output into final reconstructed state.

        raw_out channels:
          [:3]   -> delta_xyz
          [3:4]  -> field_raw (to be constrained with sigmoid)
          [4:]   -> passed through if present

        template_x:
          [:3]   -> template xyz
          [3:4]  -> template scalar (not directly reused here)

        final output:
          [x, y, z, field, ...]
        """
        if raw_out.shape[-1] < 4:
            return raw_out

        template_xyz = template_x[..., :3]
        delta_xyz = raw_out[..., :3]
        pred_xyz = template_xyz + delta_xyz

        field_raw = raw_out[..., 3:4]
        pred_field = torch.sigmoid(field_raw)

        if raw_out.shape[-1] == 4:
            return torch.cat([pred_xyz, pred_field], dim=-1)

        rest = raw_out[..., 4:]
        return torch.cat([pred_xyz, pred_field, rest], dim=-1)

    def forward(self, x, edge_index, batch, params):
        """
        x is both:
          - encoder input
          - decoder template input

        Returned recon_x is absolute [x,y,z,field], not [dx,dy,dz,field_raw].
        """
        graph_latent = self.encoder(x, edge_index, batch)  # PRE-fuse latent

        fused_latent = self.fuse(torch.cat([graph_latent, params], dim=1))
        raw_out = self.decoder(fused_latent, x[:, :self.in_dim], batch, params)
        recon_x = self._compose_output_from_template(raw_out, x[:, :self.in_dim])

        return recon_x, graph_latent

    def decode(self, latent_z, template_x, edge_index, batch, params):
        """
        Decode from regressed latent using template particle features.

        Returned recon_x is absolute [x,y,z,field].
        """
        fused_latent = self.fuse(torch.cat([latent_z, params], dim=1))
        raw_out = self.decoder(fused_latent, template_x[:, :self.in_dim], batch, params)
        recon_x = self._compose_output_from_template(raw_out, template_x[:, :self.in_dim])
        return recon_x


__all__ = ["PointNetGNNAutoencoder"]