import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool


class PointNetAutoencoder(nn.Module):
    """
    PointNet-based autoencoder for Lagrangian CFD point cloud data.

    Encodes per-point features, aggregates via max pooling,
    fuses with graph-level parameters, and decodes per-point reconstruction.
    """

    def __init__(self, input_dim=4, param_dim=8, latent_dim=64, output_dim=4):
        """
        Args:
            input_dim (int): Dimension of input features per point (e.g., [x, y, z, field]).
            param_dim (int): Dimension of auxiliary parameters (e.g., user-defined + normalized time).
            latent_dim (int): Size of latent embedding.
            output_dim (int): Output feature dimension (usually same as input_dim).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim

        # Encoder MLP (pointwise feature extractor)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Conditioning fusion: latent + parameter vector -> fused latent
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder MLP: reconstruct pointwise output
        self.decoder_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x, batch, params):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input features [N, input_dim]
            batch (LongTensor): Batch indices for graph pooling [N]
            params (Tensor): Conditioning parameters per graph [B, param_dim]

        Returns:
            recon_x (Tensor): Reconstructed features [N, output_dim]
        """
        x_feat = self.encoder_mlp(x)  # [N, latent_dim]
        graph_latent = global_max_pool(x_feat, batch)  # [B, latent_dim]

        fused_latent = self.fuse(torch.cat([graph_latent, params], dim=1))  # [B, latent_dim]
        fused_broadcast = fused_latent[batch]  # [N, latent_dim]

        decoder_input = torch.cat([x_feat, fused_broadcast], dim=1)  # [N, 2 * latent_dim]
        recon_x = self.decoder_mlp(decoder_input)  # [N, output_dim]

        return recon_x
