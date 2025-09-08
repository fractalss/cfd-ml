# Saurav Mitra
# model_gnn.py  (XYZ(+time)+param[+baseline] -> field)
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.utils import to_undirected

TensorLike = Union[np.ndarray, torch.Tensor]

__all__ = [
    "ThreeLayerGCN",
    "build_gcn_model",
    "build_node_features_xyz",
    "train_gcn_model",
    "laplacian_smoothness",
]

# ---------------------------------
# Convolution factory (GCN / SAGE / GAT / GIN)
# ---------------------------------

def _make_conv(kind: str, in_c: int, out_c: int, **kw):
    k = (kind or "").lower()
    if k == "gcn":
        return GCNConv(in_c, out_c, cached=True)
    if k == "sage":
        return SAGEConv(in_c, out_c)  # no cached arg
    if k == "gat":
        return GATConv(
            in_c,
            out_c,
            heads=kw.get("heads", 4),
            concat=False,  # keep output dim = out_c
            dropout=kw.get("attn_dropout", 0.1),
        )
    if k == "gin":
        mlp = nn.Sequential(nn.Linear(in_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c))
        return GINConv(mlp)
    raise ValueError(f"Unknown conv kind: {kind}")


# ---------------------------------
# Model
# ---------------------------------

class ThreeLayerGCN(nn.Module):
    """A minimal 3-layer GNN for scalar field regression per node.

    in_dim:  node feature dimension (XYZ [+ time features] [+ params] [+ optional baseline channel])
    hidden:  hidden width for conv layers
    out_dim: number of target channels (default 1)
    dropout: dropout after conv1/conv2
    conv_type: one of {"gcn","sage","gat","gin"}. Default = "sage".
    Extra kwargs are passed to the convs (e.g., heads, attn_dropout for GAT).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_dim: int = 1,
        dropout: float = 0.1,
        conv_type: str = "sage",
        **kw,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.conv1 = _make_conv(conv_type, in_dim, hidden, **kw)
        self.conv2 = _make_conv(conv_type, hidden, hidden, **kw)
        self.conv3 = _make_conv(conv_type, hidden, out_dim, **kw)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, F)  edge_index: (2, E)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)  # (N, out_dim)
        return x


def build_gcn_model(
    in_dim: int,
    hidden: int = 64,
    out_dim: int = 1,
    dropout: float = 0.1,
    **kw,
) -> nn.Module:
    """Small helper to construct the GNN with a clean signature.

    By default this now builds a GraphSAGE-based model (conv_type="sage").
    Pass conv_type="gcn" / "gat" / "gin" via **kw to switch.
    Supported extra **kw:
        - conv_type: str
        - heads: int (for GAT)
        - attn_dropout: float (for GAT)
    """
    return ThreeLayerGCN(in_dim=in_dim, hidden=hidden, out_dim=out_dim, dropout=dropout, **kw)


# ---------------------------------
# Node feature builder (match inference.py)
# ---------------------------------

def build_node_features_xyz(nodes_df) -> torch.Tensor:
    """
    Standardize x,y,z individually and stack -> (N,3) torch.float32.
    This must match inference.py.
    """
    feats = []
    for c in ("x", "y", "z"):
        v = nodes_df[c].to_numpy(dtype=np.float32)
        mu = float(v.mean())
        sd = float(v.std()) if v.std() > 0 else 1.0
        feats.append(((v - mu) / sd).reshape(-1, 1))
    X = np.concatenate(feats, axis=1) if len(feats) else np.zeros((len(nodes_df), 0), dtype=np.float32)
    return torch.tensor(X, dtype=torch.float32)


# ---------------------------------
# (Optional) Time features
# ---------------------------------

def _time_feature_dim(mode: str, m: int) -> int:
    if mode == "none":
        return 0
    if mode == "scalar":
        return 1
    if mode == "fourier":
        return 2 * max(1, int(m))
    raise ValueError(f"Unknown time feature mode: {mode}")


def _time_block_for(
    t: float,
    N: int,
    device,
    dtype,
    mode: str,
    t_min: float,
    t_max: float,
    mu: Optional[float],
    sigma: Optional[float],
    m: int,
):
    if mode == "none":
        return None
    if mode == "scalar":
        sigma = sigma if (sigma and sigma > 0) else 1.0
        t_norm = (t - (mu if mu is not None else 0.0)) / sigma
        return torch.full((N, 1), float(t_norm), device=device, dtype=dtype)
    if mode == "fourier":
        if t_max == t_min:
            t_max = t_min + 1.0
        t01 = (t - t_min) / (t_max - t_min)
        freqs = torch.tensor([2**k for k in range(max(1, int(m)))], device=device, dtype=dtype)
        ang = 2.0 * math.pi * t01 * freqs
        s = torch.sin(ang)
        c = torch.cos(ang)
        vec = torch.cat([s, c], dim=-1)  # (2m,)
        return vec.unsqueeze(0).repeat(N, 1)
    raise ValueError(f"Unknown time feature mode: {mode}")


# ---------------------------------
# Training (graph fixed; features = XYZ(+time)+param [+ baseline] ; targets = field)
# ---------------------------------

def _broadcast_params(params_row: np.ndarray, N: int, device, dtype) -> torch.Tensor:
    """
    params_row: (param_dim,) numpy
    Returns: (N, param_dim) tensor broadcast across nodes
    """
    if params_row.ndim != 1:
        params_row = params_row.reshape(-1)
    pr = torch.from_numpy(params_row.astype(np.float32)).to(device)
    return pr.unsqueeze(0).repeat(N, 1).to(dtype)


def _concat_features(
    X_node: torch.Tensor,  # (N, F_node) static xyz features on device
    params_row: np.ndarray,  # (P,)
    device,
    *,
    add_time: bool = False,
    time_val: Optional[float] = None,
    time_mode: str = "none",
    t_min: float = 0.0,
    t_max: float = 1.0,
    t_mu: Optional[float] = None,
    t_sigma: Optional[float] = None,
    fourier_m: int = 4,
    time_gain: float = 1.0,
    extra_node_chan: Optional[torch.Tensor] = None,  # (N, C_extra) e.g., baseline channel
) -> torch.Tensor:
    """
    Build per-node features:
      [ X_node (N,F_node) | broadcast(params) (N,P) | (optional) time block (N,T) | (optional) extra_node_chan ]
    """
    N = X_node.size(0)
    dtype = X_node.dtype
    feats = [X_node]

    # Optional extra per-node channel(s), e.g. baseline (current snapshot)
    if extra_node_chan is not None:
        if extra_node_chan.dim() == 1:
            extra_node_chan = extra_node_chan.view(-1, 1)
        if extra_node_chan.size(0) != N:
            raise ValueError("extra_node_chan must have N rows to match X_node")
        feats.append(extra_node_chan.to(X_node.device, dtype=dtype))

    # Broadcasted params
    P_node = _broadcast_params(params_row, N, device, dtype)  # (N,P)
    feats.append(P_node)

    # Optional time block
    if add_time and time_mode != "none" and time_val is not None:
        tb = _time_block_for(
            time_val, N, device, dtype, time_mode, t_min, t_max, t_mu, t_sigma, fourier_m
        )
        tb = tb * float(time_gain)
        feats.append(tb)
    return torch.cat(feats, dim=1)  # (N, F_node + C_extra + P + T)


def laplacian_smoothness(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    # pred: (N, 1) or (N, C)
    src, dst = edge_index
    diff = pred[src] - pred[dst]
    return (diff ** 2).mean()


def _weighted_mse(
    pred: torch.Tensor,  # (N,1)
    target: torch.Tensor,  # (N,1)
    node_weights: Optional[torch.Tensor] = None,  # (N,)
) -> torch.Tensor:
    """Compute MSE with optional per-node weights.
    If node_weights is provided, return sum(w * err^2) / sum(w). Otherwise, mean(err^2).
    """
    err2 = (pred - target) ** 2  # (N,1)
    if node_weights is not None:
        w = node_weights.view(-1, 1)
        num = (w * err2).sum()
        den = w.sum().clamp_min(1e-12)
        return num / den
    return err2.mean()


@torch.no_grad()
def _eval_set(
    model: nn.Module,
    Y: np.ndarray,  # (S, N) -> targets (field)
    P: np.ndarray,  # (S, P)
    X_node: torch.Tensor,  # (N, F_node) on device
    edge_index: torch.Tensor,
    device,
    *,
    add_time: bool = False,
    times: Optional[np.ndarray] = None,
    time_mode: str = "none",
    t_min: float = 0.0,
    t_max: float = 1.0,
    t_mu: Optional[float] = None,
    t_sigma: Optional[float] = None,
    fourier_m: int = 4,
    time_gain: float = 1.0,
    node_weights: Optional[torch.Tensor] = None,  # (N,)
    sample_weights: Optional[np.ndarray] = None,  # (S,)
    baseline_eval: Optional[np.ndarray] = None,  # (S, N) or (S,N,1) optional per-sample baseline channels
) -> float:
    model.eval()

    losses = []
    for s in range(Y.shape[0]):
        extra = None
        if baseline_eval is not None:
            be = baseline_eval[s]
            if be.ndim == 2 and be.shape[1] == 1:
                be = be.reshape(-1)
            extra = torch.from_numpy(be.astype(np.float32)).to(device).view(-1, 1)
        feats = _concat_features(
            X_node,
            P[s],
            device,
            add_time=add_time,
            time_val=(times[s] if (add_time and times is not None) else None),
            time_mode=time_mode,
            t_min=t_min,
            t_max=t_max,
            t_mu=t_mu,
            t_sigma=t_sigma,
            fourier_m=fourier_m,
            time_gain=time_gain,
            extra_node_chan=extra,
        )
        # Defensive check against feature-width drift
        if hasattr(model, "in_dim") and feats.size(1) != model.in_dim:
            raise RuntimeError(
                f"[VAL] Feature width {feats.size(1)} != model.in_dim {model.in_dim}. "
                f"(extra={'yes' if extra is not None else 'no'}, P={P.shape[1]}, "
                f"time_dim={_time_feature_dim(time_mode, fourier_m) if add_time else 0})"
            )
        y_true = torch.from_numpy(Y[s].astype(np.float32)).to(device).reshape(-1, 1)
        y_pred = model(feats, edge_index)
        loss = _weighted_mse(y_pred, y_true, node_weights)
        if sample_weights is not None:
            loss = loss * float(sample_weights[s])
        losses.append(loss.item())
    # average over samples (already weighted by sample_weights above)
    return float(np.mean(losses)) if losses else float("nan")


def _squeeze_baseline_arr(arr: Optional[np.ndarray], N: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim != 2 or a.shape[1] != N:
        raise ValueError(f"baseline array must be (S,N) or (S,N,1); got {arr.shape}")
    return a


def train_gcn_model(
    Y_train: np.ndarray,  # (S_train, N)   targets (field)
    Y_val: np.ndarray,    # (S_val,   N)   targets (field)
    P_train: np.ndarray,  # (S_train, P)
    P_val: np.ndarray,    # (S_val,   P)
    edge_index: np.ndarray | torch.Tensor,
    *,
    # Static node features (xyz standardized)  pass as np.ndarray or torch.Tensor
    X_node: TensorLike | None = None,  # (N, F_node)
    # Optimization
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 64,
    dropout: float = 0.1,
    lambda_smooth: float = 0.0,
    device: Optional[str] = None,
    shuffle: bool = True,
    shuffle_seed: Optional[int] = None,
    # Optional time conditioning
    add_time: bool = False,
    times_train: Optional[np.ndarray] = None,
    times_test: Optional[np.ndarray] = None,
    time_mode: str = "none",
    fourier_m: int = 4,
    time_gain: float = 1.0,
    # Data pruning
    ignore_head_frac: float = 0.0,
    # --- NEW: weighting ---
    node_weights: Optional[np.ndarray] = None,  # (N,)
    sample_weights: Optional[np.ndarray] = None,  # (S_train,)
    # --- NEW: optional baseline channel per sample ---
    baseline_train: Optional[np.ndarray] = None,  # (S_train, N) or (S_train,N,1)
    baseline_val: Optional[np.ndarray] = None,    # (S_val,   N) or (S_val,  N,1)
    **kw,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Graph-first training:
      inputs = [X_node | (baseline?) | params | (time?)]  --> predict field (residual or full)

    Parameters:
      ignore_head_frac : float in [0,1)
          If > 0, drop the first fraction of TRAIN snapshots (by current order in
          Y_train/P_train/times_train) before training. This does not touch validation.
          Useful to exclude early transients from model fitting.
      node_weights : per-node weights (N,) applied to the MSE reduction.
      sample_weights : optional per-snapshot weights (S_train,). If provided, the
          per-snapshot loss is multiplied by sample_weights[s] before averaging
          across snapshots each epoch.
      baseline_* : optional per-sample baseline channels (S, N) or (S,N,1). If provided, one
          scalar channel per node will be concatenated to the input features.
    """
    assert Y_train.ndim == 2 and Y_val.ndim == 2, "Y_* must be (S, N)"
    assert P_train.ndim == 2 and P_val.ndim == 2, "P_* must be (S, P)"
    assert (
        Y_train.shape[0] == P_train.shape[0] and Y_val.shape[0] == P_val.shape[0]
    ), "Snapshot count mismatch"

    # --- Optionally drop the first fraction of training snapshots ---
    if ignore_head_frac and float(ignore_head_frac) > 0.0:
        S0 = int(Y_train.shape[0])
        k = max(0, min(S0, int(np.floor(float(ignore_head_frac) * S0))))
        if k > 0:
            Y_train = Y_train[k:]
            P_train = P_train[k:]
            if times_train is not None:
                times_train = times_train[k:]
            if sample_weights is not None and len(sample_weights) == S0:
                sample_weights = sample_weights[k:]
            if baseline_train is not None and baseline_train.shape[0] == S0:
                baseline_train = baseline_train[k:]
            print(f"[PRUNE] Ignored first {k}/{S0} (~{(100.0*k/max(1,S0)):.1f}%) training snapshots (ignore_head_frac={ignore_head_frac}).")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Edge index to device (undirected)
    if isinstance(edge_index, np.ndarray):
        ei = torch.from_numpy(edge_index).long().to(dev)
    else:
        ei = edge_index.to(dev)
    ei = to_undirected(ei, num_nodes=int(Y_train.shape[1]))

    N = Y_train.shape[1]
    param_dim = P_train.shape[1]

    # Node features to device
    if X_node is None:
        raise ValueError(
            "X_node (static node features) must be provided and match inference.py (xyz standardized)."
        )
    if isinstance(X_node, np.ndarray):
        X_node = torch.tensor(X_node, dtype=torch.float32, device=dev)
    else:
        X_node = X_node.to(dev, dtype=torch.float32)
    if X_node.shape[0] != N:
        raise ValueError(f"X_node has N={X_node.shape[0]} but targets have N={N}")

    # Normalize baseline array shapes to (S,N)
    baseline_train = _squeeze_baseline_arr(baseline_train, N)
    baseline_val   = _squeeze_baseline_arr(baseline_val, N)

    # Determine extra feature dims (baseline channel)
    extra_dim = 1 if baseline_train is not None else 0

    t_feat_dim = _time_feature_dim(time_mode, fourier_m) if add_time else 0
    in_dim = X_node.shape[1] + extra_dim + param_dim + t_feat_dim

    model = build_gcn_model(in_dim=in_dim, hidden=hidden, out_dim=1, dropout=dropout, **kw).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # convert weights to tensors on device
    node_w_t = None
    if node_weights is not None:
        if isinstance(node_weights, np.ndarray):
            node_w_t = torch.from_numpy(node_weights.astype(np.float32)).to(dev)
        else:
            node_w_t = node_weights.to(dev, dtype=torch.float32)
        if node_w_t.numel() != N:
            raise ValueError(f"node_weights length {node_w_t.numel()} does not match N={N}")

    sample_w = None
    if sample_weights is not None:
        sample_w = np.asarray(sample_weights, dtype=np.float32)
        if sample_w.shape[0] != Y_train.shape[0]:
            raise ValueError("sample_weights must have length S_train")

    # --- Early stopping config (from kwargs / pipeline) ---
    early_stopping = bool(kw.pop("early_stopping", True))
    es_patience = int(kw.pop("es_patience", 30))
    es_min_delta = float(kw.pop("es_min_delta", 0.0))
    es_restore_best = bool(kw.pop("es_restore_best", True))

    # Time stats
    if add_time and times_train is not None and len(times_train) == Y_train.shape[0]:
        t_mu = float(np.mean(times_train))
        t_sigma = float(np.std(times_train)) if np.std(times_train) > 0 else 1.0
        t_min = float(np.min(times_train))
        t_max = float(np.max(times_train)) if float(np.max(times_train)) > t_min else (t_min + 1.0)
    else:
        t_mu = t_sigma = None
        t_min, t_max = 0.0, 1.0

    hist = {"train_mse": [], "val_mse": [], "val_rmse": []}
    # --- Early stopping state ---
    best_val = float("inf")
    best_state = None
    no_improve = 0
    # --- RNG for shuffling ---
    if shuffle_seed is not None:
        np.random.seed(int(shuffle_seed))
        torch.manual_seed(int(shuffle_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(shuffle_seed))
    print(f"[DBG] TRAIN Y mean/std {Y_train.mean():.3e}/{Y_train.std():.3e} | "
          f"VAL Y mean/std {Y_val.mean():.3e}/{Y_val.std():.3e}")

    for epoch in range(1, epochs + 1):
        model.train()

        order = np.random.permutation(Y_train.shape[0]) if shuffle else np.arange(Y_train.shape[0])

        epoch_losses = []
        for idx, s in enumerate(order):
            extra = None
            if baseline_train is not None:
                extra = torch.from_numpy(baseline_train[s].astype(np.float32)).to(dev).view(-1, 1)
            feats = _concat_features(
                X_node,
                P_train[s],
                dev,
                add_time=add_time,
                time_val=(times_train[s] if (add_time and times_train is not None) else None),
                time_mode=time_mode,
                t_min=t_min,
                t_max=t_max,
                t_mu=t_mu,
                t_sigma=t_sigma,
                fourier_m=fourier_m,
                time_gain=time_gain,
                extra_node_chan=extra,
            )
            # Guard against feature-width drift
            if hasattr(model, "in_dim") and feats.size(1) != model.in_dim:
                raise RuntimeError(
                    f"[TRAIN] Feature width {feats.size(1)} != model.in_dim {model.in_dim}. "
                    f"(extra={'yes' if extra is not None else 'no'}, P={P_train.shape[1]}, "
                    f"time_dim={_time_feature_dim(time_mode, fourier_m) if add_time else 0})"
                )

            y_true = torch.from_numpy(Y_train[s].astype(np.float32)).to(dev).reshape(-1, 1)

            opt.zero_grad(set_to_none=True)
            y_pred = model(feats, ei)
            mse = _weighted_mse(y_pred, y_true, node_w_t)
            reg = laplacian_smoothness(y_pred, ei) * lambda_smooth if lambda_smooth > 0 else 0.0
            loss = mse + (reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=dev, dtype=y_pred.dtype))
            if sample_w is not None:
                loss = loss * float(sample_w[s])
            loss.backward()
            opt.step()
            epoch_losses.append(mse.detach().item())

        train_mse = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_mse = _eval_set(
            model,
            Y_val,
            P_val,
            X_node,
            ei,
            dev,
            add_time=add_time,
            times=times_test,
            time_mode=time_mode,
            t_min=t_min,
            t_max=t_max,
            t_mu=t_mu,
            t_sigma=t_sigma,
            fourier_m=fourier_m,
            time_gain=time_gain,
            node_weights=node_w_t,
            sample_weights=None,
            baseline_eval=baseline_val,
        )
        val_rmse = math.sqrt(val_mse) if val_mse == val_mse else float("nan")  # guard NaN

        hist["train_mse"].append(train_mse)
        hist["val_mse"].append(val_mse)
        hist["val_rmse"].append(val_rmse)

        print(
            f"[EPOCH {epoch:03d}] train_MSE={train_mse:.6e}  val_MSE={val_mse:.6e}  val_RMSE={val_rmse:.6e}"
        )

        # --- Early stopping check ---
        improved = (val_mse + es_min_delta) < best_val
        if improved:
            best_val = val_mse
            no_improve = 0
            if es_restore_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if early_stopping and no_improve >= es_patience:
                print(f"[ES] Early stopping at epoch {epoch} (best val_MSE={best_val:.6e})")
                if es_restore_best and best_state is not None:
                    model.load_state_dict(best_state)
                    print("[ES] Restored best model weights")
                break

    return model, hist
