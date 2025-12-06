import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(c_in, c_out, 1),
                       nn.BatchNorm1d(c_out),
                       nn.ReLU(True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PointNetEncoder(nn.Module):
    def __init__(self,
                 in_dim: int = 3,
                 feat_dims=(64, 128, 1024),
                 latent_dim: int = 256):
        super().__init__()
        self.point_feat_dim = feat_dims[-1]
        self.mlp = SharedMLP([in_dim, *feat_dims])
        self.fc = nn.Sequential(
            nn.Linear(self.point_feat_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        feat = self.mlp(x)
        if mask is None:
            global_feat, _ = torch.max(feat, dim=2)
        else:
            m = mask.bool()
            if m.dim() == 2:
                m = m.unsqueeze(1)
            elif m.dim() == 3 and m.shape[1] != 1 and m.shape[1] != feat.shape[1]:
                if m.shape[1] == feat.shape[2]:
                    m = m[:, 0:1, :]
                else:
                    m = m[:, :1, :]
            m = m.expand_as(feat)
            feat_masked = feat.masked_fill(~m, float("-inf"))
            global_feat, _ = torch.max(feat_masked, dim=2)

        z = self.fc(global_feat)
        return z, feat.transpose(1, 2)


class PointNetDecoder(nn.Module):
    def __init__(self,
                 point_feat_dim: int = 1024,
                 in_stream_dim: int = 4,  # only [x, y, z, field] now
                 latent_dim: int = 256,
                 param_dim: int = 0,
                 out_dim: int = 4):
        super().__init__()
        self.point_feat_dim = point_feat_dim
        self.in_stream_dim = in_stream_dim
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.out_dim = out_dim

        cat_dim = point_feat_dim + in_stream_dim + (latent_dim + param_dim)

        self.conv1 = nn.Conv1d(cat_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, 1)

    def forward(self, per_point_feat, x_in, z, params=None):
        B, N, _ = x_in.shape
        if params is None:
            if self.param_dim > 0:
                zeros = torch.zeros(B, self.param_dim, device=z.device, dtype=z.dtype)
                cond = torch.cat([z, zeros], dim=-1)
            else:
                cond = z
        else:
            cond = torch.cat([z, params], dim=-1)

        cond = cond.unsqueeze(1).expand(B, N, cond.shape[-1])
        x_dynamic = x_in[:, :, :4]  # Use only [x, y, z, field] for decoding
        cat = torch.cat([x_dynamic, per_point_feat, cond], dim=-1)
        cat = cat.transpose(1, 2)

        y = F.relu(self.bn1(self.conv1(cat)))
        y = F.relu(self.bn2(self.conv2(y)))
        out = self.conv3(y)
        out = out.transpose(1, 2)
        return out


class PointNetAutoencoder(nn.Module):
    """PointNet-style autoencoder for Lagrangian ROM.

    This variant is now strictly defined on the 4 *dynamic* features:
        [x, y, z, field]

    Static identifiers like CloudID / CloudID_base must be kept
    outside the network and reattached downstream.
    """
    def __init__(self,
                 in_dim: int = 4,   # only [x, y, z, field]
                 out_dim: int = 4,  # reconstruct [x, y, z, field]
                 latent_dim: int = 256,
                 param_dim: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.param_dim = param_dim

        # Encoder sees only the dynamic features
        self.encoder = PointNetEncoder(in_dim=in_dim, latent_dim=latent_dim)

        # Decoder also operates only on the 4 dynamic channels
        self.decoder = PointNetDecoder(point_feat_dim=1024,
                                       in_stream_dim=4,
                                       latent_dim=latent_dim,
                                       param_dim=param_dim,
                                       out_dim=out_dim)

    def forward(self, x, mask=None, params=None):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, N, 4]; dynamic features [x, y, z, field].
        mask : torch.Tensor or None
            Optional validity mask [B, N] / [B, 1, N].
        params : torch.Tensor or None
            Optional conditioning parameters [B, P].
        """
        z, per_point_feat = self.encoder(x, mask=mask)
        recon = self.decoder(per_point_feat, x, z, params)
        return recon, z


# ----------------------------------------------------------------------
# Residual Decoder for Baseline + Residual ROM
# ----------------------------------------------------------------------

class PointNetResidualDecoder(nn.Module):
    """
    Conditional PointNet-like decoder for residual ROM.

    Input per particle (scaled space):
        - baseline dynamic features: [x, y, z, field]_baseline_scaled (4)
        - tiled global params: [P] (velocity, etc.)

    Output per particle (scaled space):
        - residual dynamic features: [?x, ?y, ?z, ?field] (4)

    This is meant to be trained on:
        resid_dyn_scaled = CFD_dyn_scaled - baseline_dyn_scaled
    and used at inference as:
        dyn_scaled_full = baseline_dyn_scaled_infer + model(baseline_dyn_scaled_infer, params_infer)
    """

    def __init__(
        self,
        in_dim: int = 4,   # baseline dynamic features [x, y, z, field]
        param_dim: int = 0,
        feat_dims=(64, 128, 256),
        latent_dim: int = 256,
        out_dim: int = 4,  # residual [?x, ?y, ?z, ?field]
    ):
        super().__init__()
        self.in_dim = in_dim
        self.param_dim = param_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        # Per-point feature extractor on [baseline_dyn_scaled + params]
        point_in_dim = in_dim + param_dim  # 4 + P
        self.point_mlp = SharedMLP([point_in_dim, *feat_dims])
        self.point_feat_dim = feat_dims[-1]

        # Global feature from per-point features
        self.global_fc = nn.Sequential(
            nn.Linear(self.point_feat_dim, latent_dim),
            nn.ReLU(True),
        )

        # Decoder over concatenated [per-point feat, global feat]
        cat_dim = self.point_feat_dim + latent_dim
        self.conv1 = nn.Conv1d(cat_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, out_dim, 1)

    def forward(self, baseline_dyn_scaled, params=None):
        """
        Parameters
        ----------
        baseline_dyn_scaled : torch.Tensor
            [B, N, 4]  -- baseline dynamic features in scaled space.
        params : torch.Tensor or None
            [B, P]  -- global parameters (e.g. velocity). If provided, they
                       will be tiled per particle and concatenated to inputs.

        Returns
        -------
        resid_pred : torch.Tensor
            [B, N, 4]  -- predicted residual (scaled space).
        """
        B, N, _ = baseline_dyn_scaled.shape

        # Tile params to per-particle if given
        if params is not None and self.param_dim > 0:
            # params: [B, P] -> [B, N, P]
            p_tiled = params.unsqueeze(1).expand(B, N, self.param_dim)
            x_in = torch.cat([baseline_dyn_scaled, p_tiled], dim=-1)  # [B, N, 4+P]
        else:
            x_in = baseline_dyn_scaled  # [B, N, 4]

        # Per-point features via SharedMLP
        x_in_t = x_in.transpose(1, 2)              # [B, 4+P, N]
        feat = self.point_mlp(x_in_t)              # [B, C_feat, N]

        # Global pooling
        global_feat, _ = torch.max(feat, dim=2)    # [B, C_feat]
        global_feat = self.global_fc(global_feat)  # [B, latent_dim]

        # Broadcast global feature to all points
        global_expanded = global_feat.unsqueeze(-1).expand(
            -1, -1, feat.shape[-1]
        )                                          # [B, latent_dim, N]

        # Concatenate per-point + global
        cat = torch.cat([feat, global_expanded], dim=1)  # [B, C_feat+latent_dim, N]

        # Decode residual
        y = F.relu(self.bn1(self.conv1(cat)))
        y = F.relu(self.bn2(self.conv2(y)))
        out = self.conv3(y)                       # [B, out_dim, N]
        resid_pred = out.transpose(1, 2)          # [B, N, out_dim]

        return resid_pred


# ----------------------------------------------------------------------
# GAT-based residual decoder on COARSE centroid graph (K nodes)
# ----------------------------------------------------------------------
try:
    from torch_geometric.nn import GATConv
except ImportError as e:
    raise ImportError(
        "PointNetGATResidualDecoder requires torch-geometric. "
        "Install it with the appropriate wheels for your PyTorch/CUDA version."
    ) from e


class PointNetGATResidualDecoder(nn.Module):
    """
    GAT-based residual decoder for Lagrangian ROM on a *coarsened* graph.

    This implementation assumes:
      - You have already coarsened parcels to K centroids (e.g. via K-means).
      - You have a static centroid graph: edge_index_single [2, E] for K nodes.
      - The same centroid graph is reused across batches, rev dirs, and time.

    Inputs
    ------
    baseline_dyn_nodes : torch.Tensor
        Shape [B, K, 4]; baseline dynamic features in scaled space for each
        coarse node (centroid). Typically aggregated from parcels.
    params : torch.Tensor
        Shape [B, P]; global parameters per snapshot (e.g., inlet velocity).

    Output
    ------
    residual_dyn_nodes : torch.Tensor
        Shape [B, K, 4]; predicted residual (scaled) to be added to the
        baseline node features, to then be prolonged back to parcels.
    """

    def __init__(
        self,
        in_dim: int = 4,        # baseline dynamic channels [x,y,z,field] at nodes
        param_dim: int = 1,     # number of global params (P)
        hidden_dim: int = 64,
        gat_heads: int = 4,
        num_gat_layers: int = 3,
        out_dim: int = 4,       # residual dynamic channels
        num_nodes: int = 2000,  # K (e.g., 2000 centroids)
        edge_index: torch.Tensor | None = None,   # [2, E] for single graph
    ):
        super().__init__()
        self.in_dim = in_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.num_gat_layers = num_gat_layers
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        if edge_index is None:
            raise ValueError(
                "PointNetGATResidualDecoder requires a precomputed edge_index "
                "[2, E] for the centroid graph. None was provided."
            )

        # store single-graph edge_index as buffer so it moves with .to(device)
        edge_index = edge_index.long()
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {edge_index.shape}"
            )
        if int(edge_index.max()) >= num_nodes:
            raise ValueError(
                f"edge_index contains node index >= num_nodes={num_nodes} "
                f"(max index = {int(edge_index.max())})"
            )
        self.register_buffer("edge_index_single", edge_index, persistent=True)

        # Node input features = baseline_dyn(4) + params_broadcast(P)
        node_in_dim = in_dim + param_dim

        # First linear lift on node features
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # GAT layers on node features; graph structure given by edge_index
        gat_layers = []
        in_channels = hidden_dim
        for _ in range(num_gat_layers):
            gat = GATConv(
                in_channels=in_channels,
                out_channels=hidden_dim,
                heads=gat_heads,
                concat=False,  # keep feature dim = hidden_dim
                dropout=0.0,
            )
            gat_layers.append(gat)
        self.gat_layers = nn.ModuleList(gat_layers)

        # Final MLP head maps hidden_dim -> out_dim (residual [x,y,z,field])
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def _expand_edges_for_batch(self, batch_size: int) -> torch.Tensor:
        """
        Take the single-graph edge_index_single (for K nodes) and replicate it
        for a batch of size B, offsetting node indices by b * K so we end up
        with a disjoint union of B graphs.

        Returns
        -------
        edge_index_batched : [2, B * E]
        """
        K = self.num_nodes
        edge_index = self.edge_index_single  # [2, E]
        E = edge_index.shape[1]

        # Create offsets [0, K, 2K, ..., (B-1)*K] on device
        device = edge_index.device
        offsets = torch.arange(batch_size, device=device, dtype=torch.long) * K  # [B]

        # edge_index[None, ...] : [1, 2, E] -> broadcast with offsets[:,None,None]
        edge_index_expanded = edge_index.unsqueeze(0) + offsets.view(batch_size, 1, 1)
        # Now shape [B, 2, E]; reshape to [2, B*E]
        edge_index_batched = edge_index_expanded.permute(1, 0, 2).reshape(2, -1)
        return edge_index_batched

    def forward(self, baseline_dyn: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        baseline_dyn : [B, K, 4]
        params       : [B, P]
        returns residual_dyn : [B, K, 4]
        """
        B, K, C = baseline_dyn.shape
        assert C == self.in_dim, f"Expected in_dim={self.in_dim}, got {C}"
        assert K == self.num_nodes, (
            f"baseline_dyn has K={K} nodes, expected num_nodes={self.num_nodes}"
        )
        P = params.shape[1]
        assert P == self.param_dim, f"Expected param_dim={self.param_dim}, got {P}"

        # 1) Node features: concat baseline_dyn + params_broadcast -> [B,K,4+P]
        params_expanded = params.unsqueeze(1).expand(B, K, P)               # [B,K,P]
        node_feats = torch.cat([baseline_dyn, params_expanded], dim=-1)     # [B,K,4+P]

        # 2) Encode node features
        node_feats = self.node_encoder(node_feats)                          # [B,K,H]

        # 3) Flatten nodes for torch_geometric: [B*K, H]
        x = node_feats.reshape(B * K, self.hidden_dim)                      # [B*K,H]

        # 4) Build batched edge_index for B disjoint copies of the centroid graph
        edge_index = self._expand_edges_for_batch(B)                        # [2,B*E]

        # 5) Apply stacked GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x, inplace=True)

        # 6) Map back to residual dyn features
        x = self.out_mlp(x)                                                 # [B*K,4]
        residual_dyn = x.view(B, K, self.out_dim)                           # [B,K,4]

        return residual_dyn


__all__ = [
    "PointNetAutoencoder",
    "PointNetResidualDecoder",
    "PointNetGATResidualDecoder",
]
