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
# NEW: Residual Decoder for Baseline + Residual ROM
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



# Optional: you can place this in a separate file if you prefer.
try:
    from torch_geometric.nn import GATConv
except ImportError as e:
    raise ImportError(
        "PointNetGATResidualDecoder requires torch-geometric. "
        "Install it with the appropriate wheels for your PyTorch/CUDA version."
    ) from e


class PointNetGATResidualDecoder(nn.Module):
    """
    GAT-based residual decoder for Lagrangian ROM.

    Inputs
    ------
    baseline_dyn : torch.Tensor
        Shape [B, N, 4]; dynamic baseline features [x, y, z, field] in *scaled* space.
    params : torch.Tensor
        Shape [B, P]; global parameters per snapshot (e.g., inlet velocity).

    Output
    ------
    residual_dyn : torch.Tensor
        Shape [B, N, 4]; predicted residual (scaled) to be added to baseline_dyn.
    """

    def __init__(
        self,
        in_dim: int = 4,      # baseline dynamic channels [x,y,z,field]
        param_dim: int = 1,   # number of global params (P)
        hidden_dim: int = 64,
        gat_heads: int = 4,
        num_gat_layers: int = 3,
        out_dim: int = 4,     # residual dynamic channels
        k_neighbors: int = 12,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.num_gat_layers = num_gat_layers
        self.out_dim = out_dim
        self.k_neighbors = k_neighbors

        # Node input features = baseline_dyn(4) + params_broadcast(P)
        node_in_dim = in_dim + param_dim

        # First linear lift on node features
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # GAT layers (on node features; graph built from particle coordinates)
        gat_layers = []
        in_channels = hidden_dim
        for li in range(num_gat_layers):
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

    # ------------------------------------------------------------------
    # Helper: build k-NN graph per batch using coordinates from baseline
    # ------------------------------------------------------------------
    def _build_knn_graph(self, pos: torch.Tensor):
        """
        pos : [B, N, 3] coordinates (x,y,z).
        Returns:
            edge_index : [2, E_total] concatenated over batch with proper offsets
            batch_vec  : [B*N] batch index for each node (for torch_geometric)
        NOTE: This is O(B * N^2) naive. For production, replace with
              precomputed edges or a better kNN implementation.
        """
        B, N, _ = pos.shape
        device = pos.device

        all_src = []
        all_dst = []
        all_batch = []

        for b in range(B):
            # (N,3)
            p = pos[b]  # [N,3]
            # Compute pairwise distances (N,N) - naive; OK as a sketch
            with torch.no_grad():
                dists = torch.cdist(p, p, p=2)  # [N,N]
                # For each node, get k+1 nearest (including self)
                knn = torch.topk(dists, k=self.k_neighbors + 1, largest=False).indices  # [N, k+1]

            # Exclude self (index 0 in knn row)
            nbrs = knn[:, 1:]  # [N, k]

            src = torch.arange(N, device=device).unsqueeze(1).expand_as(nbrs)  # [N,k]
            dst = nbrs

            src = src.reshape(-1)
            dst = dst.reshape(-1)

            # Offset node indices by batch
            offset = b * N
            src = src + offset
            dst = dst + offset

            all_src.append(src)
            all_dst.append(dst)
            all_batch.append(torch.full((N,), b, dtype=torch.long, device=device))

        edge_src = torch.cat(all_src, dim=0)
        edge_dst = torch.cat(all_dst, dim=0)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)  # [2, E_total]
        batch_vec = torch.cat(all_batch, dim=0)                # [B*N]

        return edge_index, batch_vec

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, baseline_dyn: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        baseline_dyn : [B, N, 4]
        params       : [B, P]
        returns residual_dyn : [B, N, 4]
        """
        B, N, C = baseline_dyn.shape
        assert C == self.in_dim, f"Expected in_dim={self.in_dim}, got {C}"
        P = params.shape[1]
        assert P == self.param_dim, f"Expected param_dim={self.param_dim}, got {P}"

        device = baseline_dyn.device

        # 1) Node features: concat baseline_dyn + params_broadcast -> [B,N,4+P]
        params_expanded = params.unsqueeze(1).expand(B, N, P)        # [B,N,P]
        node_feats = torch.cat([baseline_dyn, params_expanded], dim=-1)  # [B,N,4+P]

        # 2) Encode node features
        node_feats = self.node_encoder(node_feats)                   # [B,N,H]

        # 3) Build graph using positions from baseline_dyn (x,y,z)
        pos = baseline_dyn[..., :3]  # [B,N,3]
        edge_index, batch_vec = self._build_knn_graph(pos)          # [2,E], [B*N]

        # Flatten nodes for torch_geometric: [B*N, H]
        x = node_feats.reshape(B * N, self.hidden_dim)

        # 4) Apply stacked GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x, inplace=True)

        # 5) Map back to residual dyn features
        x = self.out_mlp(x)                     # [B*N, out_dim]
        residual_dyn = x.view(B, N, self.out_dim)  # [B,N,4]

        return residual_dyn

__all__ = [
    "PointNetAutoencoder",
    "PointNetResidualDecoder",
    "PointNetGATResidualDecoder"
]

