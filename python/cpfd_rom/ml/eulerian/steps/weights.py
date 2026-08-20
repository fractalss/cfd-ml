"""Weighting strategies for Eulerian ROM training losses."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from cpfd_rom.util.logging_config import detail

logger = logging.getLogger(__name__)

__all__ = [
    "make_training_weights",
    "per_node_variance_weights_from_residuals",
    "per_sample_interface_weights",
]


def _validate_training_matrix(Y_train: np.ndarray) -> np.ndarray:
    """Return *Y_train* as an array after validating its expected shape."""
    if not isinstance(Y_train, np.ndarray):
        raise TypeError(
            "Y_train must be a NumPy array with shape (S_train, N_nodes); "
            f"got {type(Y_train).__name__}"
        )
    if Y_train.ndim != 2:
        raise ValueError(
            "Y_train must be 2D with shape (S_train, N_nodes); "
            f"got shape {Y_train.shape}"
        )
    return Y_train


def per_node_variance_weights_from_residuals(
    *,
    Y_train: np.ndarray,
    nodes_df=None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Compute mean-normalized inverse-variance weights for every node.

    Parameters
    ----------
    Y_train
        Training residuals or targets with shape ``(S_train, N_nodes)``.
    nodes_df
        Unused placeholder retained for API compatibility.

    Returns
    -------
    (w_node, meta)
        ``w_node`` has shape ``(N_nodes,)`` and dtype ``float32``. ``meta``
        contains the unscaled per-node variances.
    """
    del nodes_df
    Y_train = _validate_training_matrix(Y_train)
    n_samples, n_nodes = Y_train.shape

    if n_nodes == 0:
        logger.warning("Cannot compute variance weights: Y_train has no nodes")
        return np.ones((0,), dtype=np.float32), None
    if n_samples == 0:
        logger.warning(
            "Cannot estimate node variance from an empty training set; "
            "using uniform weights"
        )
        return np.ones((n_nodes,), dtype=np.float32), None

    var_j = np.var(Y_train, axis=0)
    mean_variance = float(np.mean(var_j))

    # Preserve inverse-variance weighting while supplying a finite floor for
    # constant data, where the original relative epsilon would also be zero.
    eps = max(1.0e-8 * mean_variance, np.finfo(np.float64).eps)
    inv_var = 1.0 / np.maximum(var_j, eps)
    inv_var_mean = float(np.mean(inv_var))

    if not np.isfinite(inv_var_mean) or inv_var_mean <= 0.0:
        logger.warning(
            "Variance weights were non-finite or degenerate; using uniform weights"
        )
        w_node = np.ones((n_nodes,), dtype=np.float32)
    else:
        w_node = (inv_var / inv_var_mean).astype(np.float32)

    return w_node, {"var": var_j.tolist()}


def per_sample_interface_weights(
    *,
    Y_train: np.ndarray,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Compute mean-normalized snapshot weights from interface sharpness.

    Interface sharpness is estimated as the mean absolute difference between
    adjacent columns of each training snapshot.
    """
    Y_train = _validate_training_matrix(Y_train)
    n_samples, n_nodes = Y_train.shape

    if n_samples == 0:
        logger.warning("Cannot compute interface weights: Y_train has no samples")
        return np.ones((0,), dtype=np.float32), None
    if n_nodes < 2:
        logger.warning(
            "Interface weighting requires at least two nodes; using uniform weights"
        )
        return np.ones((n_samples,), dtype=np.float32), None

    grads = np.abs(np.diff(Y_train, axis=1)).mean(axis=1)
    mean_gradient = float(np.mean(grads))

    if not np.isfinite(mean_gradient) or mean_gradient <= 0.0:
        logger.warning(
            "Interface gradients were zero or non-finite; using uniform weights"
        )
        w_samp = np.ones((n_samples,), dtype=np.float32)
    else:
        w_samp = (grads / mean_gradient).astype(np.float32)

    return w_samp, {"grads": grads.tolist()}


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
    """Create node and/or sample weights for the selected strategy.

    Extra keyword-only arguments are retained for API compatibility. The
    return value is ``(node_weights, sample_weights)``; an unused weight type
    is returned as ``None``.
    """
    del Y_val, times_train, times_val, P_train, P_val, W_by_time

    strategy_eff = "none" if strategy is None else str(strategy).strip().lower()
    detail(
        logger,
        "Training-loss weighting strategy=%s, residual_targets=%s",
        strategy_eff,
        use_residual,
    )

    if strategy_eff == "none":
        detail(logger, "No training-loss weighting applied")
        return None, None

    if strategy_eff == "variance":
        node_weights, _ = per_node_variance_weights_from_residuals(
            Y_train=Y_train,
            nodes_df=nodes_df,
        )
        detail(
            logger,
            "Computed variance-based node weights: count=%d, mean=%.3f",
            node_weights.size,
            float(node_weights.mean()) if node_weights.size else float("nan"),
        )
        return node_weights, None

    if strategy_eff == "interface":
        sample_weights, _ = per_sample_interface_weights(Y_train=Y_train)
        detail(
            logger,
            "Computed interface-based sample weights: count=%d, mean=%.3f",
            sample_weights.size,
            float(sample_weights.mean()) if sample_weights.size else float("nan"),
        )
        return None, sample_weights

    raise ValueError(
        f"Unknown loss-weighting strategy {strategy!r}; "
        "expected one of: 'none', 'variance', 'interface'"
    )
