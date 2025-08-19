# Saurav Mitra
# inference.py  GCN inference helpers (XYZ(+time)+param -> field)
from __future__ import annotations
from typing import Optional

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
    edge_index: torch.Tensor,
    X_node: torch.Tensor,
    params_row: np.ndarray,
    *,
    time_val: Optional[float] = None,
    add_time: bool = False,
    time_mode: str = "none",
    t_min: float = 0.0,
    t_max: float = 1.0,
    t_mu: Optional[float] = None,
    t_sigma: Optional[float] = None,
    fourier_m: int = 4,
) -> torch.Tensor:
    """Return predicted field as a tensor shaped (N,).

    Args:
        model: Trained ThreeLayerGCN (or compatible) model.
        edge_index: Graph connectivity [2, E] on any device.
        X_node: Static node features (e.g., standardized XYZ) shaped (N, F_node).
        params_row: Global parameter vector for this snapshot, shape (P,).
        time_val: Scalar time value for this snapshot, if time conditioning is enabled.
        add_time: Whether to concatenate time features.
        time_mode: "none" | "scalar" | "fourier".
        t_min, t_max, t_mu, t_sigma, fourier_m: Same statistics/settings used in training.

    Notes:
        Ensure the same preprocessing used during training (XYZ standardization, parameter vector,
        and time-feature configuration) is applied here for consistency.
    """
    device = next(model.parameters()).device

    feats = _concat_features(
        X_node.to(device),
        params_row,
        device,
        add_time=add_time,
        time_val=time_val,
        time_mode=time_mode,
        t_min=t_min,
        t_max=t_max,
        t_mu=t_mu,
        t_sigma=t_sigma,
        fourier_m=fourier_m,
    )

    y = model(feats, edge_index.to(device)).squeeze(-1)  # (N,)
    return y
