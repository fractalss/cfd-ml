import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool


class PointNetAutoencoder(nn.Module):
    """
    PointNet-based autoencoder for Lagrangian CFD point cloud data.
    Compatible with PyTorch Geometric, using batched [N, C] input with batch index.
    """
    def __init__(self, in_dim=4, param_dim=0, latent_dim=256, out_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim

        # Encoder MLP (per-point)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Latent + parameters -> fused latent
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

    def forward(self, x, batch, params):
        """
        Forward pass for training.

        Args:
            x (Tensor): [N, in_dim] point features
            batch (LongTensor): [N] batch indices
            params (Tensor): [B, param_dim] conditioning vector

        Returns:
            recon_x (Tensor): [N, out_dim] reconstructed features
        """
        x_feat = self.encoder_mlp(x)  # [N, latent_dim]
        graph_latent = global_max_pool(x_feat, batch)  # [B, latent_dim]

        fused_latent = self.fuse(torch.cat([graph_latent, params], dim=1))  # [B, latent_dim]
        fused_broadcast = fused_latent[batch]  # [N, latent_dim]

        decoder_input = torch.cat([x_feat, fused_broadcast], dim=1)  # [N, 2*latent_dim]
        recon_x = self.decoder_mlp(decoder_input)  # [N, out_dim]

        return recon_x

    def decode(self, latent_z, template_x, batch, params):
        """
        Inference-time decode: from graph-level latent + template.

        Args:
            latent_z (Tensor): [B, latent_dim] latent vectors
            template_x (Tensor): [N, in_dim] point features
            batch (LongTensor): [N] batch index
            params (Tensor): [B, param_dim] conditioning

        Returns:
            recon_x (Tensor): [N, out_dim] output
        """
        fused_latent = self.fuse(torch.cat([latent_z, params], dim=1))  # [B, latent_dim]
        fused_broadcast = fused_latent[batch]  # [N, latent_dim]

        x_feat = self.encoder_mlp(template_x)  # [N, latent_dim]
        decoder_input = torch.cat([x_feat, fused_broadcast], dim=1)  # [N, 2 * latent_dim]
        recon_x = self.decoder_mlp(decoder_input)  # [N, out_dim]

        return recon_x


__all__ = ["PointNetAutoencoder"]
