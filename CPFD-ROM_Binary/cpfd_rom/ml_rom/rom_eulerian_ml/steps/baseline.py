# ===============================
# File: cpfd_rom/ml_rom/rom_eulerian_ml/pipeline_refactored/steps/baseline.py
# Purpose: Baseline (per-time linear/ridge over parameter) & residual-corrector helpers
# ===============================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .align import to_file_order


def _param_scalar_series(P: np.ndarray) -> np.ndarray:
    """Extract a scalar parameter per sample from P.
    Accepts shapes (S,), (S,1), (S,Pd) or (S,N,Pd). Returns (S,) using the first column;
    for node-wise inputs, averages that first column across nodes.
    """
    P = np.asarray(P)
    if P.ndim == 1:
        return P.astype(np.float32)
    if P.ndim == 2:
        return P[:, 0].astype(np.float32)
    if P.ndim == 3:
        return P[:, :, 0].mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported P shape {P.shape}")


# -----------------------------------------------------------------------------
# Design matrix + numerically safe ridge solve
# -----------------------------------------------------------------------------

def _design(p: np.ndarray, poly_deg: int) -> np.ndarray:
    """Return the per-sample design row(s) for a scalar parameter array p.
    p: (R,1) or (R,) array
    poly_deg: 1 -> [p, 1], 2 -> [p^2, p, 1]
    """
    p = np.asarray(p).reshape(-1, 1)
    if poly_deg >= 2:
        return np.hstack([p**2, p, np.ones_like(p)])
    return np.hstack([p, np.ones_like(p)])


def _solve_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Solve multi-output ridge: W = (X^T X + ?I)^{-1} X^T Y
    with an automatic ? floor to avoid near-singularity.
    X: (R, D)  Y: (R, N)  ->  W: (D, N)
    """
    Xt = X.T
    XtX = Xt @ X
    # Automatic minimum ridge to improve conditioning (scale-aware)
    lam_floor = 1e-8 * float(np.trace(XtX)) / max(1, X.shape[1])
    lam = max(float(alpha), lam_floor)
    return np.linalg.solve(XtX + lam * np.eye(X.shape[1]), Xt @ Y)


# -----------------------------------------------------------------------------
# Baseline wrapper (for TRAIN/VAL features and inference reuse)
# -----------------------------------------------------------------------------

@dataclass
class BaselineModel:
    """Lightweight wrapper around per-time linear/ridge fits.

    W_by_time: dict mapping float(time) -> (D, N) weight matrix
    poly_deg:  baseline polynomial degree (1 or 2)
    ridge_alpha: ridge used at fit time (for logging only)
    atol:        time matching tolerance used to find a stored time
    """
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
        """Return baseline field for each time at the given user parameter.
        shapes: times (S,) -> (S,N)
        """
        S = int(len(times))
        out: list[np.ndarray] = []
        for s in range(S):
            W = self._find_W(float(times[s]))  # (D,N)
            xs = self._x_star(float(user_param))  # (1,D)
            out.append((xs @ W).reshape(-1))
        return np.vstack(out).astype(np.float32)

    def predict_samples(self, times: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Return baseline field for each (time, param) sample.
        shapes: times (S,), P (S,1 or S,P)-> (S,N)
        """
        pcol = _param_scalar_series(P)
        S = int(len(times))
        out: list[np.ndarray] = []
        for s in range(S):
            W = self._find_W(float(times[s]))
            xs = self._x_star(float(pcol[s]))
            out.append((xs @ W).reshape(-1))
        return np.vstack(out).astype(np.float32)


def make_baseline_model(W_by_time: Dict[float, np.ndarray], *, poly_deg: int, ridge_alpha: float, atol: float) -> BaselineModel:
    """Helper so callers can wrap an existing per-time weight dict without changing signatures."""
    return BaselineModel(W_by_time=W_by_time, poly_deg=int(poly_deg), ridge_alpha=float(ridge_alpha), atol=float(atol))


# -----------------------------------------------------------------------------
# Baseline-only path (per-time linear/ridge over param)
# -----------------------------------------------------------------------------

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
    """Fit multi-output linear/ridge per time across revs; predict at user_param; write files."""
    # collect all samples
    T_all = times_train if times_val is None else np.concatenate([times_train, times_val], axis=0)
    P_all = P_train if P_val is None else np.concatenate([P_train, P_val], axis=0)
    Y_all = Y_train if not Y_val.size else np.concatenate([Y_train, Y_val], axis=0)

    # reference times (read from ref graph directory)
    out_root = Path(output_dir) / "graph"
    ref_dir = out_root / ref_rev
    from ..loader import list_targets
    targets_ref = list_targets(ref_dir)
    times_target = np.array([t for t, _ in targets_ref], dtype=float)

    p_all = _param_scalar_series(P_all)
    preds_list, used_times = [], []
    for t in times_target:
        idx = np.where(np.isclose(T_all, float(t), atol=atol))[0]
        # --- DEBUG: how many unique revs matched at this time? ---
        try:
            uniq_revs_total = len(np.unique(p_all))
            uniq_revs_matched = len(np.unique(p_all[idx])) if idx.size else 0
            print(f"[DEBUG] t={t:.6f}, matches={idx.size}, unique_revs={uniq_revs_matched}/{uniq_revs_total} (atol={atol})")
        except Exception as _e:
            print(f"[DEBUG] t={t:.6f}, matches={idx.size} (atol={atol}) [rev debug failed: {_e}]")

        if idx.size < 2:
            continue
        X = _design(p_all[idx].reshape(-1, 1), poly_deg)   # (R,D)
        W = _solve_ridge(X, Y_all[idx, :], ridge_alpha)    # (D,N)

        up = np.array([[user_param]], dtype=float)
        x_star = _design(up, poly_deg)                      # (1,D)
        y_hat = (x_star @ W).reshape(-1)
        preds_list.append(y_hat)
        used_times.append(t)

    if not preds_list:
        raise RuntimeError("Baseline produced no predictions (insufficient matching times across revs).")

    preds = np.vstack(preds_list).astype(np.float32)
    times = np.array(used_times, dtype=float)

    # NODE(canonical) ? FILE order and write
    preds_file = to_file_order(preds, colmap_canon)
    inv = np.empty_like(colmap_canon); inv[colmap_canon] = np.arange(colmap_canon.size)
    nodes_df_file = nodes_df.iloc[inv].reset_index(drop=True)

    from ..evaluation import _write_rom_only
    out_dir = Path(output_dir) / "ML" / f"ROM_param_{float(user_param):.3f}_BASELINE"
    _write_rom_only(preds_file, times, nodes_df_file, field_variable, out_dir)
    print(f"[BASELINE] wrote ROM-only files to {out_dir}")
    return preds_file, times


# -----------------------------------------------------------------------------
# Residual corrector utilities
# -----------------------------------------------------------------------------

def build_train_val_baseline(
    times_train, P_train, Y_train,
    times_val,   P_val,   Y_val,
    *, poly_deg: int, ridge_alpha: float, atol: float,
):
    """Fit per-time linear/ridge weights over the scalar parameter and return a
    dict mapping time->W.

    Assumptions / shapes:
       times_*: (S,) or (S,1) real-valued times.
       P_* : (S,1) or (S,Pd) (we use the first column as the scalar parameter).
       Y_* : (S,N) (single scalar target per node). If you have (S,N,Od), fit per channel.

    Strategy:
      1) Try to fit W at each needed time using TRAIN-only matches.
      2) If TRAIN is insufficient (<2 matches across revs) at that time, fall back to TRAIN+VAL.
      3) If still insufficient, raise with a clear message.
    """
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

    if times_val is not None and len(times_val) > 0:
        T_va = np.asarray(times_val, dtype=float).reshape(-1)
        P_va = np.asarray(P_val, dtype=float)
        Y_va = _ensure_2d_Y(np.asarray(Y_val, dtype=float))
        T_all = np.concatenate([T_tr, T_va], axis=0)
        P_all = np.concatenate([P_tr, P_va], axis=0)
        Y_all = np.concatenate([Y_tr, Y_va], axis=0)
    else:
        T_all, P_all, Y_all = T_tr, P_tr, Y_tr

    def _fit_W_at_time(tt: float, T: np.ndarray, P: np.ndarray, Y: np.ndarray) -> Optional[np.ndarray]:
        idx = np.where(np.isclose(T, float(tt), atol=atol))[0]
        # Coverage debug
        uniq_revs_total = len(np.unique(_pcol(P)))
        uniq_revs_matched = len(np.unique(_pcol(P)[idx])) if idx.size else 0
        #     print(f"[BASELINE/FIT] t={tt:.6f}: matches={idx.size}, unique_revs={uniq_revs_matched}/{uniq_revs_total} (atol={atol})")
        # except Exception as e:
        #     print(f"[BASELINE/FIT] t={tt:.6f}: matches={idx.size} (debug err: {e})")
        if idx.size < 2:
            return None
        X = _design(_pcol(P)[idx].reshape(-1, 1), poly_deg)  # (R,D)
        return _solve_ridge(X, Y[idx, :], ridge_alpha)        # (D,N)

    times_needed = np.unique(
        np.concatenate([
            T_tr if T_tr is not None else np.array([], float),
            (T_va if (times_val is not None and len(times_val) > 0) else np.array([], float))
        ]).astype(float)
    )

    W_by_time: Dict[float, np.ndarray] = {}
    for tt in times_needed:
        W = _fit_W_at_time(tt, T_tr, P_tr, Y_tr)        # try TRAIN only first
        if W is None:
            W = _fit_W_at_time(tt, T_all, P_all, Y_all)    # fallback TRAIN+VAL
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
    return W_by_time, meta


def apply_train_val_residuals(Y_train, Y_val, times_train, times_val, P_train, P_val, W_by_time):
    """Return (Y_train_n, Y_val_n, res_mu, res_std) where targets are residuals to baseline.
    NOTE: This function keeps the original scalar normalization for backward compatibility.
    Prefer `apply_train_val_residuals_pernode` for better conditioning.
    """
    def _x_star(pv: float, D: int):
        if D == 3:
            return np.array([[pv**2, pv, 1.0]], dtype=float)
        return np.array([[pv, 1.0]], dtype=float)

    def _get_W(tt: float, atol=1e-8):
        for k in W_by_time.keys():
            if np.isclose(k, tt, atol=atol):
                return W_by_time[k]
        raise KeyError(f"No baseline weights stored for t={tt}")

    y_base_train = np.zeros_like(Y_train, dtype=np.float32)
    for s in range(Y_train.shape[0]):
        tt = float(times_train[s]); W = _get_W(tt); pv = float(P_train[s, 0])
        xs = _x_star(pv, W.shape[0])
        y_base_train[s, :] = (xs @ W).reshape(-1)

    y_base_val = np.zeros_like(Y_val, dtype=np.float32) if Y_val.size else Y_val
    if Y_val.size:
        for s in range(Y_val.shape[0]):
            tt = float(times_val[s]); W = _get_W(tt); pv = float(P_val[s, 0])
            xs = _x_star(pv, W.shape[0])
            y_base_val[s, :] = (xs @ W).reshape(-1)

    Y_res_train = (Y_train - y_base_train).astype(np.float32)
    Y_res_val   = (Y_val   - y_base_val).astype(np.float32) if Y_val.size else Y_val

    res_mu  = float(np.mean(Y_res_train))
    res_std = float(np.std(Y_res_train)) if float(np.std(Y_res_train)) > 0 else 1.0
    print(f"[RESCORR-NORM] mu={res_mu:.6e} std={res_std:.6e} (residual)")

    Y_train_n = (Y_res_train - res_mu) / res_std
    Y_val_n   = (Y_res_val   - res_mu) / res_std if Y_val.size else Y_val

    return Y_train_n, Y_val_n, res_mu, res_std


# ----------------------- Preferred per-node residual normalization -----------------------

def apply_train_val_residuals_pernode(Y_train, Y_val, times_train, times_val, P_train, P_val, W_by_time):
    """Per-node residual normalization.
    Returns: (Y_train_n, Y_val_n, mu_j, sig_j)
    - mu_j, sig_j are (N,) vectors computed on TRAIN residuals only with an epsilon floor.
    """
    def _x_star(pv: float, D: int):
        if D == 3:
            return np.array([[pv**2, pv, 1.0]], dtype=float)
        return np.array([[pv, 1.0]], dtype=float)

    def _get_W(tt: float, atol=1e-8):
        for k in W_by_time.keys():
            if np.isclose(k, tt, atol=atol):
                return W_by_time[k]
        raise KeyError(f"No baseline weights stored for t={tt}")

    y_base_train = np.zeros_like(Y_train, dtype=np.float32)
    for s in range(Y_train.shape[0]):
        tt = float(times_train[s]); W = _get_W(tt); pv = float(P_train[s, 0])
        xs = _x_star(pv, W.shape[0])
        y_base_train[s, :] = (xs @ W).reshape(-1)

    y_base_val = np.zeros_like(Y_val, dtype=np.float32) if Y_val.size else Y_val
    if Y_val.size:
        for s in range(Y_val.shape[0]):
            tt = float(times_val[s]); W = _get_W(tt); pv = float(P_val[s, 0])
            xs = _x_star(pv, W.shape[0])
            y_base_val[s, :] = (xs @ W).reshape(-1)

    R_tr = (Y_train - y_base_train).astype(np.float32)      # (S_tr, N)
    R_va = (Y_val   - y_base_val).astype(np.float32) if Y_val.size else Y_val

    # Per-node stats with robust epsilon floor
    mu_j = R_tr.mean(axis=0)
    glob = R_tr.std() if R_tr.size else 1.0
    eps  = max(1e-6, 1e-3 * float(glob))
    sig_j = R_tr.std(axis=0)
    sig_j = np.where(sig_j > eps, sig_j, eps)
    n_clamped = int((sig_j <= eps).sum())
    print(f"[RESNORM] eps={eps:.2e}, clamped_nodes={n_clamped}/{sig_j.size}")

    Y_train_n = (R_tr - mu_j) / sig_j
    Y_val_n   = (R_va - mu_j) / sig_j if (isinstance(R_va, np.ndarray) and R_va.size) else R_va

    return Y_train_n, Y_val_n, mu_j.astype(np.float32), sig_j.astype(np.float32)


def denorm_residual(pred_res_norm: np.ndarray, mu_j: np.ndarray, sig_j: np.ndarray) -> np.ndarray:
    """De-normalize residuals predicted in per-node normalized space.
    Shapes: pred_res_norm (S,N), mu_j (N,), sig_j (N,) -> returns (S,N)
    """
    return pred_res_norm * sig_j + mu_j


# -----------------------------------------------------------------------------
# (Optional) Diagnostics helpers (R^2 per time, baseline coverage)
# -----------------------------------------------------------------------------

def summarize_baseline_quality(times: np.ndarray, P: np.ndarray, Y: np.ndarray, W_by_time: Dict[float, np.ndarray], *, poly_deg: int, atol: float = 1e-8) -> Dict[str, float]:
    """Compute simple diagnostics: per-time coverage and R^2 of the linear fit on provided samples.
    Returns a dict of aggregate stats. Prints per-time details.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    P = np.asarray(P)
    pcol = P[:, 0] if P.ndim == 2 else P
    out_r2 = []
    for tt in np.unique(times):
        idx = np.where(np.isclose(times, float(tt), atol=atol))[0]
        if idx.size < 2:
            print(f"[BASELINE/QUAL] t={tt:.6f}: insufficient matches ({idx.size})")
            continue
        X = _design(pcol[idx].reshape(-1, 1), poly_deg)
        W = W_by_time.get(float(tt))
        if W is None:
            print(f"[BASELINE/QUAL] t={tt:.6f}: no W fitted")
            continue
        Y_hat = X @ W
        y = Y[idx]
        ss_res = float(np.mean((y - Y_hat)**2))
        ss_tot = float(np.mean((y - y.mean(axis=0))**2)) if y.size else 0.0
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
        out_r2.append(r2)
        print(f"[BASELINE/QUAL] t={tt:.6f}: matches={idx.size}, R2~={r2:.3f}")
    agg = {
        "mean_r2": float(np.mean(out_r2)) if out_r2 else float("nan"),
        "num_times": int(len(out_r2)),
    }
    return agg