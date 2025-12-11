import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

class PointNetAutoencoder(nn.Module):
    """
    PointNet-style autoencoder for variable-size point clouds with parameter conditioning.

    Encoder:
        - Inputs: x [N, input_dim], batch [N], params [B, param_dim]
        - Encodes point features and pools them per graph using global max pool.
        - Concatenates with simulation parameters.

    Decoder:
        - Broadcasts latent vector back to each point in the graph
        - Outputs reconstructed point-level features (e.g., positions + fields)
    """

    def __init__(self, input_dim=4, param_dim=8, latent_dim=64, output_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim

        # Encoder MLP (shared across points)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Fusion MLP (latent + parameters)
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder MLP (shared across points)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, batch, params):
        """
        Args:
            x: [N, input_dim] - point features
            batch: [N] - graph assignment index per point
            params: [B, param_dim] - simulation parameters per graph

        Returns:
            recon_x: [N, output_dim] - reconstructed features per point
        """
        # Encode per-point features
        x_feat = self.encoder_mlp(x)  # [N, latent_dim]

        # Pool to graph-level latent
        graph_latent = global_max_pool(x_feat, batch)  # [B, latent_dim]

        # Ensure params shape is [B, param_dim]
        if params.dim() == 1:
            params = params.unsqueeze(1)
        if params.shape[0] != graph_latent.shape[0]:
            raise ValueError(f"Mismatch in batch size: graph_latent {graph_latent.shape}, params {params.shape}")

        # Concatenate graph latent and parameters
        fused = torch.cat([graph_latent, params], dim=1)  # [B, latent_dim + param_dim]
        latent = self.fuse(fused)  # [B, latent_dim]

        # Broadcast latent vector to each point
        latent_broadcast = latent[batch]  # [N, latent_dim]

        # Decode per-point
        recon_x = self.decoder_mlp(latent_broadcast)  # [N, output_dim]
        return recon_x
