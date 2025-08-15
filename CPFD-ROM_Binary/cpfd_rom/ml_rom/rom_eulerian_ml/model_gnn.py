# Saurav Mitra
# model_gnn.py  (XYZ(+time)+param -> field)
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

TensorLike = Union[np.ndarray, torch.Tensor]

# ---------------------------------
# Model
# ---------------------------------

class ThreeLayerGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)
        self.conv2 = GCNConv(hidden, hidden, cached=True)
        self.conv3 = GCNConv(hidden, out_dim, cached=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, F)  edge_index: (2, E)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)  # (N, 1)
        return x

def build_gcn_model(in_dim: int, hidden: int = 64, out_dim: int = 1, dropout: float = 0.1) -> nn.Module:
    """Small helper to construct the GCN with a clean signature."""
    return ThreeLayerGCN(in_dim=in_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)

# ---------------------------------
# Node feature builder (match inference.py)
# ---------------------------------

def build_node_features_xyz(nodes_df) -> torch.Tensor:
    """
    Standardize x,y,z individually and stack -> (N,3) torch.float32.
    This must match inference.py.
    """
    feats = []
    for c in ('x', 'y', 'z'):
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
    if mode == 'none':
        return 0
    if mode == 'scalar':
        return 1
    if mode == 'fourier':
        return 2 * max(1, int(m))
    raise ValueError(f"Unknown time feature mode: {mode}")

def _time_block_for(t: float, N: int, device, dtype, mode: str,
                    t_min: float, t_max: float, mu: Optional[float], sigma: Optional[float], m: int):
    if mode == 'none':
        return None
    if mode == 'scalar':
        sigma = sigma if (sigma and sigma > 0) else 1.0
        t_norm = (t - (mu if mu is not None else 0.0)) / sigma
        return torch.full((N, 1), float(t_norm), device=device, dtype=dtype)
    if mode == 'fourier':
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
# Training (graph fixed; features = XYZ(+time)+param ; targets = field)
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
    X_node: torch.Tensor,          # (N, F_node) static xyz features on device
    params_row: np.ndarray,        # (P,)
    device,
    *,
    add_time: bool = False,
    time_val: Optional[float] = None,
    time_mode: str = 'none',
    t_min: float = 0.0,
    t_max: float = 1.0,
    t_mu: Optional[float] = None,
    t_sigma: Optional[float] = None,
    fourier_m: int = 4
) -> torch.Tensor:
    """
    Build per-node features:
      [ X_node (N,F_node) | broadcast(params) (N,P) | (optional) time block (N,T) ]
    """
    N = X_node.size(0)
    dtype = X_node.dtype
    P_node = _broadcast_params(params_row, N, device, dtype)  # (N,P)
    feats = [X_node, P_node]
    if add_time and time_mode != 'none' and time_val is not None:
        tb = _time_block_for(time_val, N, device, dtype, time_mode, t_min, t_max, t_mu, t_sigma, fourier_m)
        feats.append(tb)
    return torch.cat(feats, dim=1)  # (N, F_node+P+T)

def laplacian_smoothness(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    # pred: (N, 1)
    src, dst = edge_index
    diff = pred[src] - pred[dst]
    return (diff ** 2).mean()

@torch.no_grad()
def _eval_set(model: nn.Module,
              Y: np.ndarray,                 # (S, N) -> targets (field)
              P: np.ndarray,                 # (S, P)
              X_node: torch.Tensor,          # (N, F_node) on device
              edge_index: torch.Tensor,
              device,
              *,
              add_time: bool = False,
              times: Optional[np.ndarray] = None,
              time_mode: str = 'none',
              t_min: float = 0.0,
              t_max: float = 1.0,
              t_mu: Optional[float] = None,
              t_sigma: Optional[float] = None,
              fourier_m: int = 4) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses = []
    for s in range(Y.shape[0]):
        feats = _concat_features(
            X_node, P[s], device,
            add_time=add_time,
            time_val=(times[s] if (add_time and times is not None) else None),
            time_mode=time_mode, t_min=t_min, t_max=t_max, t_mu=t_mu, t_sigma=t_sigma, fourier_m=fourier_m
        )
        y_true = torch.from_numpy(Y[s].astype(np.float32)).to(device).reshape(-1, 1)
        y_pred = model(feats, edge_index)
        losses.append(loss_fn(y_pred, y_true).item())
    return float(np.mean(losses)) if losses else float("nan")

def train_gcn_model(
    Y_train: np.ndarray,                 # (S_train, N)   targets (field)
    Y_test:  np.ndarray,                 # (S_val,   N)   targets (field)
    P_train: np.ndarray,                 # (S_train, P)
    P_test:  np.ndarray,                 # (S_val,   P)
    edge_index: np.ndarray | torch.Tensor,
    *,
    # Static node features (xyz standardized)  pass as np.ndarray or torch.Tensor
    X_node: TensorLike | None = None,    # (N, F_node), if None you must build in pipeline and pass tensor there
    # Optimization
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 64,
    dropout: float = 0.1,
    lambda_smooth: float = 0.0,
    device: Optional[str] = None,
    # Optional time conditioning
    add_time: bool = False,
    times_train: Optional[np.ndarray] = None,
    times_test: Optional[np.ndarray] = None,
    time_mode: str = 'none',
    fourier_m: int = 4,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Graph-first training:
      inputs = [X_node | params | (time?)]  --> predict field
      targets = Y_* (field)
    """
    assert Y_train.ndim == 2 and Y_test.ndim == 2, "Y_* must be (S, N)"
    assert P_train.ndim == 2 and P_test.ndim == 2, "P_* must be (S, P)"
    assert Y_train.shape[0] == P_train.shape[0] and Y_test.shape[0] == P_test.shape[0], "Snapshot count mismatch"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Edge index to device (undirected for GCNConv)
    if isinstance(edge_index, np.ndarray):
        ei = torch.from_numpy(edge_index).long().to(dev)
    else:
        ei = edge_index.to(dev)
    ei = to_undirected(ei)

    N = Y_train.shape[1]
    param_dim = P_train.shape[1]

    # Node features to device
    if X_node is None:
        raise ValueError("X_node (static node features) must be provided and match inference.py (xyz standardized).")
    if isinstance(X_node, np.ndarray):
        X_node = torch.tensor(X_node, dtype=torch.float32, device=dev)
    else:
        X_node = X_node.to(dev, dtype=torch.float32)
    if X_node.shape[0] != N:
        raise ValueError(f"X_node has N={X_node.shape[0]} but targets have N={N}")

    t_feat_dim = _time_feature_dim(time_mode, fourier_m) if add_time else 0
    in_dim = X_node.shape[1] + param_dim + t_feat_dim

    model = build_gcn_model(in_dim=in_dim, hidden=hidden, out_dim=1, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(Y_train.shape[0])
        epoch_loss = 0.0

        for s in order:
            feats = _concat_features(
                X_node, P_train[s], dev,
                add_time=add_time,
                time_val=(times_train[s] if (add_time and times_train is not None) else None),
                time_mode=time_mode, t_min=t_min, t_max=t_max, t_mu=t_mu, t_sigma=t_sigma, fourier_m=fourier_m
            )
            y_true = torch.from_numpy(Y_train[s].astype(np.float32)).to(dev).reshape(-1, 1)

            opt.zero_grad(set_to_none=True)
            y_pred = model(feats, ei)
            mse = loss_fn(y_pred, y_true)
            reg = laplacian_smoothness(y_pred, ei) * lambda_smooth if lambda_smooth > 0 else 0.0
            loss = mse + (reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=dev, dtype=y_pred.dtype))
            loss.backward()
            opt.step()
            epoch_loss += mse.item()

        train_mse = epoch_loss / max(1, Y_train.shape[0])
        val_mse = _eval_set(
            model, Y_test, P_test, X_node, ei, dev,
            add_time=add_time, times=times_test, time_mode=time_mode,
            t_min=t_min, t_max=t_max, t_mu=t_mu, t_sigma=t_sigma, fourier_m=fourier_m
        )
        val_rmse = math.sqrt(val_mse) if val_mse == val_mse else float("nan")  # guard NaN

        hist["train_mse"].append(train_mse)
        hist["val_mse"].append(val_mse)
        hist["val_rmse"].append(val_rmse)

        print(f"[EPOCH {epoch:03d}] train_MSE={train_mse:.6e}  val_MSE={val_mse:.6e}  val_RMSE={val_rmse:.6e}")

    return model, hist
