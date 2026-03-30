import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, global_max_pool


def _make_mlp(dims, use_bn=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class EdgeConvBlock(nn.Module):
    """
    EdgeConv block:
        x_i <- max_j h([x_i, x_j - x_i])
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = EdgeConv(
            nn=_make_mlp([2 * in_dim, out_dim, out_dim], use_bn=True),
            aggr="max",
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class PointNetGNNAutoencoder(nn.Module):
    """
    PointNet + GNN autoencoder for Lagrangian CFD particle-cloud data.

    Contract:
      - forward(x, edge_index, batch, params) returns (recon_x, graph_latent)
      - latent regressor learns graph_latent (PRE-fuse)
      - decode(latent_z, template_x, edge_index, batch, params) fuses latent_z with params internally
      - field channel (index 3) is soft-bounded to [0,1] via sigmoid

    Input assumptions:
      x[:, :4] = [x, y, z, field]
    """

    def __init__(
        self,
        in_dim=4,
        param_dim=0,
        latent_dim=256,
        hidden_dim=128,
        out_dim=4,
        num_gnn_layers=3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Local PointNet-style per-particle encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # GNN interaction layers
        self.gnn_layers = nn.ModuleList([
            EdgeConvBlock(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])

        # Graph-level latent head
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Latent + parameters -> conditioned latent
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Template feature encoder used at decode time
        self.template_encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )

        # Optional GNN refinement on template branch
        self.template_gnn_layers = nn.ModuleList([
            EdgeConvBlock(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])

        # Decoder MLP
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def _apply_output_constraints(self, out: torch.Tensor) -> torch.Tensor:
        """
        Soft-bound ONLY the field channel to [0,1] using sigmoid.
        Assumes output layout [x, y, z, field].
        """
        if out.shape[-1] < 4:
            return out

        xyz = out[..., :3]
        field = torch.sigmoid(out[..., 3:4])

        if out.shape[-1] == 4:
            return torch.cat([xyz, field], dim=-1)

        rest = out[..., 4:]
        return torch.cat([xyz, field, rest], dim=-1)

    def encode(self, x, edge_index, batch):
        """
        Args:
            x         : [N_total, in_dim]
            edge_index: [2, E]
            batch     : [N_total]

        Returns:
            graph_latent : [B, latent_dim]   (PRE-fuse latent for regression)
        """
        h = self.encoder_mlp(x)  # [N_total, hidden_dim]

        for gnn in self.gnn_layers:
            h = h + gnn(h, edge_index)

        pooled = global_max_pool(h, batch)  # [B, hidden_dim]
        graph_latent = self.to_latent(pooled)  # [B, latent_dim]
        return graph_latent

    def forward(self, x, edge_index, batch, params):
        """
        Training forward pass.

        Args:
            x         : [N_total, in_dim]
            edge_index: [2, E]
            batch     : [N_total]
            params    : [B, param_dim]

        Returns:
            recon_x      : [N_total, out_dim]
            graph_latent : [B, latent_dim]   (PRE-fuse latent for regression)
        """
        graph_latent = self.encode(x, edge_index, batch)

        fused_latent = self.fuse(torch.cat([graph_latent, params], dim=1))  # [B, latent_dim]
        fused_broadcast = fused_latent[batch]  # [N_total, latent_dim]

        x_feat = self.template_encoder(x)  # [N_total, hidden_dim]
        for gnn in self.template_gnn_layers:
            x_feat = x_feat + gnn(x_feat, edge_index)

        decoder_input = torch.cat([x_feat, fused_broadcast], dim=1)  # [N_total, hidden_dim + latent_dim]
        recon_x = self.decoder_mlp(decoder_input)  # [N_total, out_dim]
        recon_x = self._apply_output_constraints(recon_x)

        return recon_x, graph_latent

    def decode(self, latent_z, template_x, edge_index, batch, params):
        """
        Inference-time decode from graph-level latent + template features.

        Args:
            latent_z   : [B, latent_dim] predicted PRE-fuse latent
            template_x : [N_total, in_dim] scaffold/template features
            edge_index : [2, E]
            batch      : [N_total]
            params     : [B, param_dim]

        Returns:
            recon_x : [N_total, out_dim]
        """
        fused_latent = self.fuse(torch.cat([latent_z, params], dim=1))  # [B, latent_dim]
        fused_broadcast = fused_latent[batch]  # [N_total, latent_dim]

        x_feat = self.template_encoder(template_x)  # [N_total, hidden_dim]
        for gnn in self.template_gnn_layers:
            x_feat = x_feat + gnn(x_feat, edge_index)

        decoder_input = torch.cat([x_feat, fused_broadcast], dim=1)
        recon_x = self.decoder_mlp(decoder_input)
        recon_x = self._apply_output_constraints(recon_x)

        return recon_x


__all__ = ["PointNetGNNAutoencoder"]