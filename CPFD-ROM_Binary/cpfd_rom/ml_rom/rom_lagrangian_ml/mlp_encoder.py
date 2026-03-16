import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool


class PointNetAutoencoder(nn.Module):
    """
    PointNet-based autoencoder for Lagrangian CFD point cloud data.
    Compatible with PyTorch Geometric, using batched [N, C] input with batch index.

    Contract:
      - forward(x, batch, params) returns (recon_x, graph_latent)
      - latent regressor should learn graph_latent (PRE-fuse)
      - decode(latent_z, template_x, batch, params) fuses latent_z with params internally
      - field channel (index 3) is soft-bounded to [0,1] via sigmoid
    """

    def __init__(self, in_dim=4, param_dim=0, latent_dim=256, out_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.out_dim = out_dim

        # Encoder MLP (per-point)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Latent + parameters -> conditioned latent
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder MLP (per-point)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, 128),
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

        # future-proof if more channels are added later
        rest = out[..., 4:]
        return torch.cat([xyz, field, rest], dim=-1)

    def forward(self, x, batch, params):
        """
        Training forward pass.

        Args:
            x      : [N_total, in_dim]
            batch  : [N_total]
            params : [B, param_dim]

        Returns:
            recon_x      : [N_total, out_dim]
            graph_latent : [B, latent_dim]   (PRE-fuse latent for regression)
        """
        x_feat = self.encoder_mlp(x)                 # [N_total, latent_dim]
        graph_latent = global_max_pool(x_feat, batch)  # [B, latent_dim]

        fused_latent = self.fuse(torch.cat([graph_latent, params], dim=1))  # [B, latent_dim]
        fused_broadcast = fused_latent[batch]  # [N_total, latent_dim]

        decoder_input = torch.cat([x_feat, fused_broadcast], dim=1)  # [N_total, 2*latent_dim]
        recon_x = self.decoder_mlp(decoder_input)  # [N_total, out_dim]
        recon_x = self._apply_output_constraints(recon_x)

        return recon_x, graph_latent

    def decode(self, latent_z, template_x, batch, params):
        """
        Inference-time decode from graph-level latent + template features.

        Args:
            latent_z   : [B, latent_dim] predicted PRE-fuse latent
            template_x : [N_total, in_dim] scaffold/template features
            batch      : [N_total]
            params     : [B, param_dim]

        Returns:
            recon_x : [N_total, out_dim]
        """
        fused_latent = self.fuse(torch.cat([latent_z, params], dim=1))  # [B, latent_dim]
        fused_broadcast = fused_latent[batch]  # [N_total, latent_dim]

        x_feat = self.encoder_mlp(template_x)  # [N_total, latent_dim]
        decoder_input = torch.cat([x_feat, fused_broadcast], dim=1)  # [N_total, 2*latent_dim]
        recon_x = self.decoder_mlp(decoder_input)  # [N_total, out_dim]
        recon_x = self._apply_output_constraints(recon_x)

        return recon_x


__all__ = ["PointNetAutoencoder"]