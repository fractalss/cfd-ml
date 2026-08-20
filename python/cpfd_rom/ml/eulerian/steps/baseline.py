"""Baseline and residual-corrector helpers for the Eulerian ML ROM."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from cpfd_rom.util.logging_config import detail

from .align import to_file_order


logger = logging.getLogger(__name__)


def _param_scalar_series(P: np.ndarray) -> np.ndarray:
    """Extract one scalar parameter per sample from ``P``."""
    P = np.asarray(P)
    if P.ndim == 1:
        return P.astype(np.float32)
    if P.ndim == 2:
        return P[:, 0].astype(np.float32)
    if P.ndim == 3:
        return P[:, :, 0].mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported P shape {P.shape}")


def _design(p: np.ndarray, poly_deg: int) -> np.ndarray:
    """Build the design matrix for a scalar parameter array."""
    p = np.asarray(p).reshape(-1, 1)
    if poly_deg >= 2:
        return np.hstack([p**2, p, np.ones_like(p)])
    return np.hstack([p, np.ones_like(p)])


def _solve_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve a numerically stabilized multi-output ridge system."""
    Xt = X.T
    XtX = Xt @ X
    lam_floor = 1e-8 * float(np.trace(XtX)) / max(1, X.shape[1])
    lam = max(float(alpha), lam_floor)
    return np.linalg.solve(XtX + lam * np.eye(X.shape[1]), Xt @ Y)


@dataclass
class BaselineModel:
    """Lightweight wrapper around per-time linear or ridge fits."""

    W_by_time: Dict[float, np.ndarray]
    poly_deg: int
    ridge_alpha: float
    atol: float = 1e-8

    def _find_W(self, tt: float) -> np.ndarray:
        for k, W in self.W_by_time.items():
            if np.isclose(k, float(tt), atol=self.atol):
                return W
        raise KeyError(f"No baseline weights stored for t={tt}")

    def _x_star(self, pval: float) -> np.ndarray:
        if self.poly_deg >= 2:
            return np.array([[pval**2, pval, 1.0]], dtype=float)
        return np.array([[pval, 1.0]], dtype=float)

    def predict_user(self, times: np.ndarray, user_param: float) -> np.ndarray:
        """Return the baseline field at each time for ``user_param``."""
        out: list[np.ndarray] = []
        for s in range(int(len(times))):
            W = self._find_W(float(times[s]))
            xs = self._x_star(float(user_param))
            out.append((xs @ W).reshape(-1))
        return np.vstack(out).astype(np.float32)

    def predict_samples(self, times: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Return the baseline field for each time/parameter sample."""
        pcol = _param_scalar_series(P)
        out: list[np.ndarray] = []
        for s in range(int(len(times))):
            W = self._find_W(float(times[s]))
            xs = self._x_star(float(pcol[s]))
            out.append((xs @ W).reshape(-1))
        return np.vstack(out).astype(np.float32)


def make_baseline_model(
    W_by_time: Dict[float, np.ndarray],
    *,
    poly_deg: int,
    ridge_alpha: float,
    atol: float,
) -> BaselineModel:
    """Wrap an existing per-time weight dictionary."""
    return BaselineModel(
        W_by_time=W_by_time,
        poly_deg=int(poly_deg),
        ridge_alpha=float(ridge_alpha),
        atol=float(atol),
    )


def fit_user_baseline(
    *,
    user_param: float,
    ref_rev: str,
    output_dir: str,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    P_train: np.ndarray,
    P_val: np.ndarray,
    times_train: np.ndarray,
    times_val: np.ndarray,
    colmap_canon: np.ndarray,
    nodes_df: pd.DataFrame,
    field_variable: str | None,
    poly_deg: int,
    ridge_alpha: float,
    atol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a per-time baseline, predict at ``user_param``, and write output."""
    T_all = times_train if times_val is None else np.concatenate([times_train, times_val], axis=0)
    P_all = P_train if P_val is None else np.concatenate([P_train, P_val], axis=0)
    Y_all = Y_train if not Y_val.size else np.concatenate([Y_train, Y_val], axis=0)

    from ..loader import list_targets

    ref_dir = Path(output_dir) / "graph" / ref_rev
    targets_ref = list_targets(ref_dir)
    times_target = np.array([t for t, _ in targets_ref], dtype=float)

    detail(
        logger,
        "Fitting baseline for user parameter %.6g across %d target times",
        float(user_param),
        len(times_target),
    )

    p_all = _param_scalar_series(P_all)
    preds_list: list[np.ndarray] = []
    used_times: list[float] = []
    for t in times_target:
        idx = np.where(np.isclose(T_all, float(t), atol=atol))[0]
        if logger.isEnabledFor(logging.DEBUG):
            try:
                uniq_revs_total = len(np.unique(p_all))
                uniq_revs_matched = len(np.unique(p_all[idx])) if idx.size else 0
                logger.debug(
                    "Baseline coverage at t=%.6f: matches=%d, unique parameters=%d/%d, atol=%g",
                    t,
                    idx.size,
                    uniq_revs_matched,
                    uniq_revs_total,
                    atol,
                )
            except Exception as exc:
                logger.debug(
                    "Baseline coverage at t=%.6f: matches=%d, atol=%g; parameter diagnostics failed: %s",
                    t,
                    idx.size,
                    atol,
                    exc,
                )

        if idx.size < 2:
            continue
        X = _design(p_all[idx].reshape(-1, 1), poly_deg)
        W = _solve_ridge(X, Y_all[idx, :], ridge_alpha)
        x_star = _design(np.array([[user_param]], dtype=float), poly_deg)
        preds_list.append((x_star @ W).reshape(-1))
        used_times.append(float(t))

    if not preds_list:
        raise RuntimeError(
            "Baseline produced no predictions (insufficient matching times across revs)."
        )

    preds = np.vstack(preds_list).astype(np.float32)
    times = np.array(used_times, dtype=float)
    preds_file = to_file_order(preds, colmap_canon)

    inv = np.empty_like(colmap_canon)
    inv[colmap_canon] = np.arange(colmap_canon.size)
    nodes_df_file = nodes_df.iloc[inv].reset_index(drop=True)

    from ..evaluation import _write_rom_only

    out_dir = Path(output_dir) / "ML" / f"ROM_param_{float(user_param):.3f}_BASELINE"
    _write_rom_only(preds_file, times, nodes_df_file, field_variable, out_dir)
    logger.info("Baseline ROM output saved to %s", out_dir)
    return preds_file, times


def build_train_val_baseline(
    times_train,
    P_train,
    Y_train,
    times_val,
    P_val,
    Y_val,
    *,
    poly_deg: int,
    ridge_alpha: float,
    atol: float,
):
    """Fit per-time baseline weights using training data, then train+validation."""

    def _pcol(P: np.ndarray) -> np.ndarray:
        P = np.asarray(P)
        if P.ndim == 1:
            return P.reshape(-1)
        if P.ndim == 2:
            return P[:, 0]
        if P.ndim == 3:
            return P[:, :, 0].mean(axis=1)
        raise ValueError(f"Unsupported P shape {P.shape}")

    def _ensure_2d_Y(Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.ndim == 3:
            if Y.shape[2] == 1:
                return Y[..., 0]
            raise ValueError(
                f"Y has shape {Y.shape}; build_train_val_baseline expects (S,N). "
                "Fit per-channel by looping over the last dimension."
            )
        if Y.ndim != 2:
            raise ValueError(f"Y must be (S,N); got {Y.shape}")
        return Y

    T_tr = np.asarray(times_train, dtype=float).reshape(-1)
    P_tr = np.asarray(P_train, dtype=float)
    Y_tr = _ensure_2d_Y(np.asarray(Y_train, dtype=float))

    has_validation = times_val is not None and len(times_val) > 0
    if has_validation:
        T_va = np.asarray(times_val, dtype=float).reshape(-1)
        P_va = np.asarray(P_val, dtype=float)
        Y_va = _ensure_2d_Y(np.asarray(Y_val, dtype=float))
        T_all = np.concatenate([T_tr, T_va], axis=0)
        P_all = np.concatenate([P_tr, P_va], axis=0)
        Y_all = np.concatenate([Y_tr, Y_va], axis=0)
    else:
        T_all, P_all, Y_all = T_tr, P_tr, Y_tr

    def _fit_W_at_time(
        tt: float,
        T: np.ndarray,
        P: np.ndarray,
        Y: np.ndarray,
    ) -> Optional[np.ndarray]:
        idx = np.where(np.isclose(T, float(tt), atol=atol))[0]
        if logger.isEnabledFor(logging.DEBUG):
            pcol = _pcol(P)
            logger.debug(
                "Baseline fit coverage at t=%.6f: matches=%d, unique parameters=%d/%d, atol=%g",
                tt,
                idx.size,
                len(np.unique(pcol[idx])) if idx.size else 0,
                len(np.unique(pcol)),
                atol,
            )
        if idx.size < 2:
            return None
        X = _design(_pcol(P)[idx].reshape(-1, 1), poly_deg)
        return _solve_ridge(X, Y[idx, :], ridge_alpha)

    times_needed = np.unique(
        np.concatenate(
            [T_tr, T_va if has_validation else np.array([], dtype=float)]
        ).astype(float)
    )

    detail(
        logger,
        "Fitting residual baseline at %d times with polynomial degree %d and ridge alpha %g",
        len(times_needed),
        poly_deg,
        ridge_alpha,
    )

    W_by_time: Dict[float, np.ndarray] = {}
    for tt in times_needed:
        W = _fit_W_at_time(tt, T_tr, P_tr, Y_tr)
        if W is None:
            logger.debug("Using training-plus-validation fallback at t=%.6f", tt)
            W = _fit_W_at_time(tt, T_all, P_all, Y_all)
        if W is None:
            raise RuntimeError(
                f"Residual baseline: insufficient samples to fit at t={tt:.6f}. "
                "Need at least two revs (distinct params) at the same time."
            )
        W_by_time[float(tt)] = W

    meta = {
        "times_needed": times_needed.astype(float).tolist(),
        "poly_deg": int(poly_deg),
        "ridge_alpha": float(ridge_alpha),
        "atol": float(atol),
    }
    detail(logger, "Residual baseline fitted at %d times", len(W_by_time))
    return W_by_time, meta


def apply_train_val_residuals(
    Y_train,
    Y_val,
    times_train,
    times_val,
    P_train,
    P_val,
    W_by_time,
):
    """Apply scalar residual normalization for backward compatibility."""

    def _x_star(pv: float, D: int):
        if D == 3:
            return np.array([[pv**2, pv, 1.0]], dtype=float)
        return np.array([[pv, 1.0]], dtype=float)

    def _get_W(tt: float, atol=1e-8):
        for k in W_by_time:
            if np.isclose(k, tt, atol=atol):
                return W_by_time[k]
        raise KeyError(f"No baseline weights stored for t={tt}")

    y_base_train = np.zeros_like(Y_train, dtype=np.float32)
    for s in range(Y_train.shape[0]):
        W = _get_W(float(times_train[s]))
        xs = _x_star(float(P_train[s, 0]), W.shape[0])
        y_base_train[s, :] = (xs @ W).reshape(-1)

    y_base_val = np.zeros_like(Y_val, dtype=np.float32) if Y_val.size else Y_val
    if Y_val.size:
        for s in range(Y_val.shape[0]):
            W = _get_W(float(times_val[s]))
            xs = _x_star(float(P_val[s, 0]), W.shape[0])
            y_base_val[s, :] = (xs @ W).reshape(-1)

    Y_res_train = (Y_train - y_base_train).astype(np.float32)
    Y_res_val = (Y_val - y_base_val).astype(np.float32) if Y_val.size else Y_val

    res_mu = float(np.mean(Y_res_train))
    residual_std = float(np.std(Y_res_train))
    res_std = residual_std if residual_std > 0 else 1.0
    detail(logger, "Scalar residual normalization: mean=%.6e, std=%.6e", res_mu, res_std)

    Y_train_n = (Y_res_train - res_mu) / res_std
    Y_val_n = (Y_res_val - res_mu) / res_std if Y_val.size else Y_val
    return Y_train_n, Y_val_n, res_mu, res_std


def apply_train_val_residuals_pernode(
    Y_train,
    Y_val,
    times_train,
    times_val,
    P_train,
    P_val,
    W_by_time,
):
    """Apply per-node residual normalization using training statistics."""

    def _x_star(pv: float, D: int):
        if D == 3:
            return np.array([[pv**2, pv, 1.0]], dtype=float)
        return np.array([[pv, 1.0]], dtype=float)

    def _get_W(tt: float, atol=1e-8):
        for k in W_by_time:
            if np.isclose(k, tt, atol=atol):
                return W_by_time[k]
        raise KeyError(f"No baseline weights stored for t={tt}")

    y_base_train = np.zeros_like(Y_train, dtype=np.float32)
    for s in range(Y_train.shape[0]):
        W = _get_W(float(times_train[s]))
        xs = _x_star(float(P_train[s, 0]), W.shape[0])
        y_base_train[s, :] = (xs @ W).reshape(-1)

    y_base_val = np.zeros_like(Y_val, dtype=np.float32) if Y_val.size else Y_val
    if Y_val.size:
        for s in range(Y_val.shape[0]):
            W = _get_W(float(times_val[s]))
            xs = _x_star(float(P_val[s, 0]), W.shape[0])
            y_base_val[s, :] = (xs @ W).reshape(-1)

    R_tr = (Y_train - y_base_train).astype(np.float32)
    R_va = (Y_val - y_base_val).astype(np.float32) if Y_val.size else Y_val

    mu_j = R_tr.mean(axis=0)
    glob = R_tr.std() if R_tr.size else 1.0
    eps = max(1e-6, 1e-3 * float(glob))
    sig_j = R_tr.std(axis=0)
    sig_j = np.where(sig_j > eps, sig_j, eps)
    n_clamped = int((sig_j <= eps).sum())
    detail(
        logger,
        "Per-node residual normalization: epsilon=%.2e, clamped nodes=%d/%d",
        eps,
        n_clamped,
        sig_j.size,
    )

    Y_train_n = (R_tr - mu_j) / sig_j
    Y_val_n = (
        (R_va - mu_j) / sig_j
        if isinstance(R_va, np.ndarray) and R_va.size
        else R_va
    )
    return Y_train_n, Y_val_n, mu_j.astype(np.float32), sig_j.astype(np.float32)


def denorm_residual(
    pred_res_norm: np.ndarray,
    mu_j: np.ndarray,
    sig_j: np.ndarray,
) -> np.ndarray:
    """De-normalize residuals predicted in per-node normalized space."""
    return pred_res_norm * sig_j + mu_j


def summarize_baseline_quality(
    times: np.ndarray,
    P: np.ndarray,
    Y: np.ndarray,
    W_by_time: Dict[float, np.ndarray],
    *,
    poly_deg: int,
    atol: float = 1e-8,
) -> Dict[str, float]:
    """Compute per-time coverage and approximate R-squared diagnostics."""
    times = np.asarray(times, dtype=float).reshape(-1)
    P = np.asarray(P)
    pcol = P[:, 0] if P.ndim == 2 else P
    out_r2: list[float] = []

    for tt in np.unique(times):
        idx = np.where(np.isclose(times, float(tt), atol=atol))[0]
        if idx.size < 2:
            logger.warning(
                "Cannot evaluate baseline quality at t=%.6f: only %d matching sample(s)",
                tt,
                idx.size,
            )
            continue

        X = _design(pcol[idx].reshape(-1, 1), poly_deg)
        W = W_by_time.get(float(tt))
        if W is None:
            logger.warning("Cannot evaluate baseline quality at t=%.6f: no fitted weights", tt)
            continue

        Y_hat = X @ W
        y = Y[idx]
        ss_res = float(np.mean((y - Y_hat) ** 2))
        ss_tot = float(np.mean((y - y.mean(axis=0)) ** 2)) if y.size else 0.0
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
        out_r2.append(r2)
        detail(
            logger,
            "Baseline quality at t=%.6f: matches=%d, approximate R2=%.3f",
            tt,
            idx.size,
            r2,
        )

    aggregate = {
        "mean_r2": float(np.mean(out_r2)) if out_r2 else float("nan"),
        "num_times": int(len(out_r2)),
    }
    detail(
        logger,
        "Baseline quality summary: mean R2=%.3f across %d times",
        aggregate["mean_r2"],
        aggregate["num_times"],
    )
    return aggregate
