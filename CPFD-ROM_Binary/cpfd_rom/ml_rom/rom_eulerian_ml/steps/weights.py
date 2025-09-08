# ===============================
# File: cpfd_rom/ml_rom/rom_eulerian_ml/steps/weights.py
# Purpose: Weighting strategies for training loss (variance/interface/none)
# ===============================
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


def per_node_variance_weights_from_residuals(*, Y_train: np.ndarray, nodes_df=None) -> Tuple[np.ndarray, Optional[dict]]:
    """Compute per-node weights proportional to inverse variance of residuals.

    Parameters
    ----------
    Y_train : np.ndarray
        Training residuals or targets of shape (S_train, N_nodes).
    nodes_df : Any
        Unused placeholder to keep a stable call signature.

    Returns
    -------
    (w_node, meta)
        w_node: (N_nodes,) float32, mean-normalized.
        meta: minimal diagnostics dict (may be None).
    """
    if not isinstance(Y_train, np.ndarray) or Y_train.size == 0:
        return np.ones((Y_train.shape[1],), dtype=np.float32), None

    var_j = Y_train.var(axis=0)
    eps = 1e-8 * float(np.mean(var_j))
    inv_var = 1.0 / np.maximum(var_j, eps)
    w_node = inv_var / np.mean(inv_var)
    return w_node.astype(np.float32), {"var": var_j.tolist()}


def per_sample_interface_weights(*, Y_train: np.ndarray) -> Tuple[np.ndarray, Optional[dict]]:
    """Toy example: weight snapshots by how sharp an 'interface' is (gradient magnitude).

    Parameters
    ----------
    Y_train : np.ndarray
        Training residuals or targets of shape (S_train, N_nodes).

    Returns
    -------
    (w_samp, meta)
        w_samp: (S_train,) float32, mean-normalized.
        meta: minimal diagnostics dict (may be None).
    """
    if not isinstance(Y_train, np.ndarray) or Y_train.size == 0:
        return np.ones((Y_train.shape[0],), dtype=np.float32), None

    grads = np.abs(np.diff(Y_train, axis=1)).mean(axis=1)
    w_samp = grads / np.mean(grads)
    return w_samp.astype(np.float32), {"grads": grads.tolist()}


def make_training_weights(
    *,
    Y_train: np.ndarray,
    Y_val=None,
    times_train=None,
    times_val=None,
    P_train=None,
    P_val=None,
    nodes_df=None,
    W_by_time=None,
    use_residual: bool = False,
    strategy: str = "none",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Factory to create node and/or sample weights depending on strategy.

    Notes
    -----
    - Extra keyword-only arguments are accepted for API stability but are not used
      by the current strategies.
    - Returns (node_weights, sample_weights); either may be None when not used.
    """
    node_weights: Optional[np.ndarray] = None
    sample_weights: Optional[np.ndarray] = None

    if strategy == "none" or strategy is None:
        print("[WEIGHTS] No weighting applied")
        return node_weights, sample_weights

    if strategy == "variance":
        w_node, _ = per_node_variance_weights_from_residuals(Y_train=Y_train, nodes_df=nodes_df)
        node_weights = w_node
        print(f"[WEIGHTS] variance-based per-node weights computed (mean={node_weights.mean():.3f})")

    if strategy == "interface":
        w_samp, _ = per_sample_interface_weights(Y_train=Y_train)
        sample_weights = w_samp
        print(f"[WEIGHTS] interface-based per-sample weights computed (mean={sample_weights.mean():.3f})")

    return node_weights, sample_weights
