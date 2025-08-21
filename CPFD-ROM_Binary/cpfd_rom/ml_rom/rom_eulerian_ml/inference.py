# Saurav Mitra
# inference.py  GCN inference helpers (XYZ(+time)+param -> field)
from __future__ import annotations
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Uses feature builder from the GCN module
from .model_gnn import _concat_features

__all__ = ["predict_gcn"]


# ---------------------------------
# Inference convenience
# ---------------------------------

@torch.no_grad()
def predict_gcn(
    model: nn.Module,
    edge_index: Union[torch.Tensor, np.ndarray],
    X_node: torch.Tensor,
    params_row: Union[np.ndarray, torch.Tensor],
    *,
    time_val: Optional[float] = None,
    add_time: bool = False,
    time_mode: str = "none",
    t_min: float = 0.0,
    t_max: float = 1.0,
    t_mu: Optional[float] = None,
    t_sigma: Optional[float] = None,
    fourier_m: int = 4,
    time_gain: float = 1.0,
) -> torch.Tensor:
    """Return predicted field as a tensor shaped (N,).

    Args:
        model: Trained ThreeLayerGCN (or compatible) model.
        edge_index: Graph connectivity [2, E]. Can be torch Tensor or numpy array.
        X_node: Static node features (e.g., standardized XYZ) shaped (N, F_node).
        params_row: Global parameter vector for this snapshot, shape (P,).
        time_val: Scalar time value for this snapshot, if time conditioning is enabled.
        add_time: Whether to concatenate time features.
        time_mode: "none" | "scalar" | "fourier".
        t_min, t_max, t_mu, t_sigma, fourier_m: Same statistics/settings used in training.
        time_gain: Scalar multiplier applied to the time feature block to increase its weight.

    Notes:
        Ensure the same preprocessing used during training (XYZ standardization, parameter vector,
        and time-feature configuration) is applied here for consistency.
    """
    device = next(model.parameters()).device

    # Normalize inputs
    if isinstance(params_row, torch.Tensor):
        pr_np = params_row.detach().cpu().numpy().reshape(-1)
    else:
        pr_np = np.asarray(params_row, dtype=np.float32).reshape(-1)

    if not isinstance(X_node, torch.Tensor):
        X_node = torch.as_tensor(X_node, dtype=torch.float32)

    # Optional: sanity check expected input dim vs model's first conv
    tdim = 0
    tm = (time_mode or "none").lower()
    if add_time and tm != "none":
        tdim = 1 if tm == "scalar" else 2 * max(1, int(fourier_m))
    exp_in = int(X_node.shape[1]) + int(pr_np.size) + tdim
    in_model = getattr(getattr(model, "conv1", None), "in_channels", None)
    if isinstance(in_model, int) and in_model > 0 and in_model != exp_in:
        raise ValueError(
            f"predict_gcn(): feature dim {exp_in} != model input {in_model}. "
            f"Pass the same time_mode/fourier_m/time_gain used in training (got time_mode={tm}, fourier_m={fourier_m})."
        )

    feats = _concat_features(
        X_node.to(device),
        pr_np,
        device,
        add_time=add_time,
        time_val=time_val,
        time_mode=time_mode,
        t_min=t_min,
        t_max=t_max,
        t_mu=t_mu,
        t_sigma=t_sigma,
        fourier_m=fourier_m,
        time_gain=time_gain,
    )

    ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    y = model(feats, ei).squeeze(-1)  # (N,)
    return y
