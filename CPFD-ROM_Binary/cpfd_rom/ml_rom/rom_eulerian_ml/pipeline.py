# ===============================
# File: cpfd_rom/ml_rom/rom_eulerian_ml/pipeline.py
# Purpose: Thin orchestrator that wires together modular Eulerian ML-ROM steps
#          with parameter-specific transient ROM output directories and metadata.
# ===============================
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
import torch

from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir

# graph + datasets
from .loader import prepare_graph_and_datasets, list_targets

# model + inference
from .model_gnn import build_gcn_model, train_gcn_model, _time_feature_dim, build_node_features_xyz
from .inference import predict_gcn

# alignment & ordering
from .steps.align import (
    maybe_load_coords_file_df,
    build_file_to_node_colmap,
    canonicalize_all,
    to_file_order,
)

# time configuration
from .steps.timecfg import resolve_time_features

# baseline & residual flow
from .steps.baseline import (
    fit_user_baseline,
    build_train_val_baseline,
    make_baseline_model,  # may not exist in older drops; fallback below if unavailable
)

# loss weighting
from .steps.weights import make_training_weights

# model i/o helpers
from .steps.modelio import torch_model_path, load_or_build_model, save_state_dict

# output writer
from .evaluation import _write_rom_only


# ------------------------------
# Local helpers  baseline
# ------------------------------

def _poly_design(p: float, deg: int) -> np.ndarray:
    """Return [1, p, p^2, ..., p^deg] as (D,) float32."""
    d = [1.0]
    for k in range(1, int(deg) + 1):
        d.append(float(p) ** k)
    return np.array(d, dtype=np.float32)


def _param_scalar(P: np.ndarray) -> np.ndarray:
    """Extract one scalar operating parameter per sample from P.

    Accepts:
      - P shaped (S, Pd)
      - P shaped (S, N, Pd)

    Returns:
      - p shaped (S,), using the first parameter column.
    """
    P = np.asarray(P)
    if P.ndim == 2:  # (S, Pd)
        return P[:, 0].astype(np.float32)
    if P.ndim == 3:  # (S, N, Pd)
        return P[:, :, 0].mean(axis=1).astype(np.float32)
    raise ValueError(f"Bad P shape {P.shape}; expected (S,Pd) or (S,N,Pd)")


class _BaselineFromW:
    """Lightweight baseline wrapper built from W_by_time.

    The object exposes the methods used by this pipeline:
      - predict_samples(times, P) -> (S, N)
      - predict_user(times, user_param) -> (S, N)

    W_by_time maps each time to polynomial coefficients shaped (D, N), where
    D = poly_deg + 1 and N is the number of Eulerian nodes/cells.
    """

    def __init__(self, W_by_time: dict[float, np.ndarray], poly_deg: int, atol: float = 1e-8):
        self.W_by_time = {float(k): np.asarray(v) for k, v in W_by_time.items()}
        self.poly_deg = int(poly_deg)
        self.atol = float(atol)
        self._times = np.array(sorted(self.W_by_time.keys()), dtype=np.float64)

    def _select_time(self, t: float) -> float:
        """Select exact time within tolerance, otherwise nearest available time."""
        diffs = np.abs(self._times - float(t))
        idx = int(np.argmin(diffs))
        return float(self._times[idx])

    def _eval_single(self, t: float, p_scalar: float) -> np.ndarray:
        t_key = self._select_time(t)
        W = self.W_by_time[t_key]  # (D, N)
        d = _poly_design(p_scalar, self.poly_deg).reshape(-1)  # (D,)
        return (d @ W).astype(np.float32)  # (N,)

    def predict_samples(self, times: np.ndarray, P: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float32).reshape(-1)
        p = _param_scalar(P)
        S = times.shape[0]

        y0 = self._eval_single(times[0], float(p[0]))
        out = np.empty((S, y0.shape[0]), dtype=np.float32)
        out[0] = y0

        for s in range(1, S):
            out[s] = self._eval_single(times[s], float(p[s]))

        return out

    def predict_user(self, times: np.ndarray, user_param: float) -> np.ndarray:
        times = np.asarray(times, dtype=np.float32).reshape(-1)
        S = times.shape[0]

        y0 = self._eval_single(times[0], float(user_param))
        out = np.empty((S, y0.shape[0]), dtype=np.float32)
        out[0] = y0

        for s in range(1, S):
            out[s] = self._eval_single(times[s], float(user_param))

        return out


def _maybe_build_baseline_features(
    *,
    use_feature: bool,
    baseline_model,
    times_train: Optional[np.ndarray],
    times_val: Optional[np.ndarray],
    P_train: np.ndarray,
    P_val: np.ndarray,
    user_param: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return baseline feature fields for train and validation.

    Returns:
      - B_train shaped (S_train, N) or None
      - B_val shaped (S_val, N) or None

    If baseline-as-feature is disabled or no baseline model exists, both are None.
    """
    if not use_feature or baseline_model is None:
        return None, None

    B_train = None
    if times_train is not None and len(times_train):
        B_train = baseline_model.predict_samples(times_train, P_train)

    B_val = None
    if times_val is not None and len(times_val):
        B_val = baseline_model.predict_samples(times_val, P_val)

    return B_train, B_val


# ------------------------------
# Local helpers  parameter-specific output paths and metadata
# ------------------------------

def _param_tag(p: float, ndigits: int = 3) -> str:
    """Return a filesystem-friendly operating-parameter tag.

    Examples:
      10.0   -> "10.000"
      42.25  -> "42.250"
      -5.0   -> "m5.000"
    """
    tag = f"{float(p):.{int(ndigits)}f}"
    return tag.replace("-", "m")


def _rom_param_output_dir(cfg, user_param: float) -> Path:
    """Return parameter-specific ROM output directory.

    This path is used by transient ML / residual-corrector ROM inference, so
    baseline+ transient ROM data is also associated with user_parameter.
    """
    ndigits = int(getattr(cfg, "param_tag_digits", 3))
    tag = _param_tag(user_param, ndigits=ndigits)
    return Path(cfg.output_dir) / "ML" / f"ROM_param_{tag}"


def _resolve_user_parameters(cfg) -> list[float]:
    """Resolve one or many inference parameters from config.

    Supported config forms:

      # Single operating point
      user_parameter: 45.0

      # Multiple operating points, legacy/user style
      user_parameter:
        - 35.0
        - 40.0
        - 45.0

      # Multiple operating points, explicit style
      user_parameters:
        - 35.0
        - 40.0
        - 45.0

    If user_parameters is present, it takes precedence over user_parameter.
    """
    vals = getattr(cfg, "user_parameters", None)

    if vals is None:
        vals = getattr(cfg, "user_parameter", 0.0)

    if vals is None:
        return [0.0]

    if isinstance(vals, (float, int, str)):
        return [float(vals)]

    if isinstance(vals, Sequence):
        return [float(v) for v in vals]

    raise ValueError(
        "cfg.user_parameter/user_parameters must be a scalar or sequence of scalars. "
        f"Got type={type(vals)!r}"
    )


def _write_inference_metadata(
    *,
    out_dir: Path,
    user_param: float,
    cfg,
    baseline_mode: str,
    use_residual: bool,
    use_baseline_as_feature: bool,
    time_cfg,
    model_path_pt: Path,
    train_param_min: float,
    train_param_max: float,
    y_min: float,
    y_max: float,
    n_nodes: int,
    n_edges: int,
    n_snapshots: int,
) -> None:
    """Write small self-describing metadata files inside each ROM output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "user_parameter": float(user_param),
        "field_variable": getattr(cfg, "field_variable", None),
        "rom_type": getattr(cfg, "rom_type", None),
        "type_of_field": getattr(cfg, "type_of_field", None),
        "baseline": baseline_mode,
        "use_residual_corrector": bool(use_residual),
        "use_baseline_as_feature": bool(use_baseline_as_feature),
        "time_features": {
            "add_time": bool(getattr(time_cfg, "add_time", False)),
            "time_mode": getattr(time_cfg, "time_mode", None),
            "fourier_m": int(getattr(time_cfg, "fourier_m", 0)),
            "time_gain": float(getattr(time_cfg, "time_gain", 1.0)),
        },
        "model": {
            "model_path_pt": str(model_path_pt),
            "conv_type": getattr(time_cfg, "conv_type", getattr(cfg, "conv_type", None)),
            "hidden": int(getattr(cfg, "hidden", 256)),
            "dropout": float(getattr(cfg, "dropout", 0.0)),
        },
        "training_ranges": {
            "parameter_min": float(train_param_min),
            "parameter_max": float(train_param_max),
            "target_min": float(y_min),
            "target_max": float(y_max),
            "user_inside_train_parameter_range": bool(train_param_min <= float(user_param) <= train_param_max),
        },
        "data_shape": {
            "n_nodes": int(n_nodes),
            "n_edges": int(n_edges),
            "n_snapshots_written": int(n_snapshots),
        },
    }

    json_path = out_dir / "rom_inference_meta.json"
    txt_path = out_dir / "rom_inference_meta.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"user_parameter = {float(user_param):.12g}\n")
        f.write(f"field_variable = {getattr(cfg, 'field_variable', None)}\n")
        f.write(f"rom_type = {getattr(cfg, 'rom_type', None)}\n")
        f.write(f"type_of_field = {getattr(cfg, 'type_of_field', None)}\n")
        f.write(f"baseline = {baseline_mode}\n")
        f.write(f"use_residual_corrector = {bool(use_residual)}\n")
        f.write(f"use_baseline_as_feature = {bool(use_baseline_as_feature)}\n")
        f.write(f"add_time = {bool(getattr(time_cfg, 'add_time', False))}\n")
        f.write(f"time_mode = {getattr(time_cfg, 'time_mode', None)}\n")
        f.write(f"fourier_m = {int(getattr(time_cfg, 'fourier_m', 0))}\n")
        f.write(f"time_gain = {float(getattr(time_cfg, 'time_gain', 1.0))}\n")
        f.write(f"model_path_pt = {str(model_path_pt)}\n")
        f.write(f"train_parameter_min = {float(train_param_min):.12g}\n")
        f.write(f"train_parameter_max = {float(train_param_max):.12g}\n")
        f.write(f"user_inside_train_parameter_range = {bool(train_param_min <= float(user_param) <= train_param_max)}\n")
        f.write(f"target_min = {float(y_min):.12g}\n")
        f.write(f"target_max = {float(y_max):.12g}\n")
        f.write(f"n_nodes = {int(n_nodes)}\n")
        f.write(f"n_edges = {int(n_edges)}\n")
        f.write(f"n_snapshots_written = {int(n_snapshots)}\n")


# ------------------------------
# Main pipeline
# ------------------------------

def run_ml_rom_pipeline(cfg, log_time):
    """Main entry point for Eulerian ML-ROM pipeline.

    This orchestrator supports:
      - baseline-only ROM path
      - residual-corrector / baseline+ transient GNN path
      - optional baseline-as-feature path
      - single or multiple inference operating parameters
      - parameter-specific transient ROM output directories
      - per-ROM metadata files
    """
    # --- Setup ---
    setup_output_dir(cfg)
    setup_model_paths(cfg)
    model_path_pt = torch_model_path(cfg.model_path_eulerian)

    # --- Build canonical graph + datasets. First rev defines graph. ---
    with log_time("Preparing graph & datasets from rev_dirs"):
        (
            ref_rev,
            ref_graph_dir,
            nodes_df,
            edge_index,
            _X_node_loader,
            Y_train,
            Y_val,
            P_train,
            P_val,
            times_train,
            times_val,
        ) = prepare_graph_and_datasets(cfg.__dict__, neighbor_set="n6")

    # --- FILE -> NODE mapping based on coords_file_order, if available. ---
    coords_path = Path(ref_graph_dir) / "coords_file_order.parquet"
    coords_file_df = None

    if coords_path.exists():
        try:
            import pandas as pd

            coords_file_df = pd.read_parquet(coords_path)[["i", "j", "k"]]
            print("[ALIGN] Using coords_file_order.parquet for FILE->NODE mapping")
        except Exception as e:
            print(f"[ALIGN] WARNING: failed to read {coords_path.name}: {e}")

    if coords_file_df is None:
        coords_file_df = maybe_load_coords_file_df(
            cfg,
            Path(ref_graph_dir) if ref_graph_dir else None,
            N_cols=(Y_train.shape[1] if Y_train.size else Y_val.shape[1]),
        )

    colmap_file_to_node = build_file_to_node_colmap(coords_file_df, nodes_df, Y_train)

    # --- Canonicalize (i,j,k) ordering and rebuild X_node accordingly. ---
    with log_time("Canonicalizing (i,j,k) order for nodes, Y_*, and edges"):
        nodes_df, Y_train, Y_val, edge_index, X_node, colmap_canon = canonicalize_all(
            nodes_df,
            Y_train,
            Y_val,
            edge_index,
            build_node_features_xyz,
            colmap_file_to_node,
        )

    # --- Dataset stats and parameter range. ---
    N = int(Y_train.shape[1]) if Y_train.size else 0
    E = int(edge_index.shape[1])
    S_tr, S_va = int(Y_train.shape[0]), int(Y_val.shape[0])
    P_dim = int(P_train.shape[1]) if P_train.size else 0

    print(f"[DATA] N(nodes)={N}  E(edges)={E}  S_train={S_tr}  S_val={S_va}  P_dim={P_dim}")
    print(
        f"[DATA] X_node shape={tuple(X_node.shape)}  "
        f"Y_train shape={tuple(Y_train.shape)}  P_train shape={tuple(P_train.shape)}"
    )

    user_parameters = _resolve_user_parameters(cfg)
    user_param = float(user_parameters[0])
    pmin, pmax = float(P_train.min()), float(P_train.max())
    print(f"[PARAM] train_range=[{pmin:.6f},{pmax:.6f}]  user={user_param:.6f}  inside_range={pmin <= user_param <= pmax}")
    print(f"[PARAM] inference user_parameters={user_parameters}")

    # --- Normalization: target and parameter. ---
    y_mu = float(np.mean(Y_train))
    y_std_raw = float(np.std(Y_train))
    y_std = y_std_raw if y_std_raw > 0 else 1.0

    print(f"[Y-NORM] mu={y_mu:.6e} std={y_std:.6e} (scalar)")

    Y_train_n_scalar = (Y_train - y_mu) / y_std
    Y_val_n_scalar = (Y_val - y_mu) / y_std if Y_val.size else Y_val

    p_mu = P_train.mean(axis=0)
    p_std = P_train.std(axis=0)
    p_std = np.where(p_std > 0, p_std, 1.0)

    P_train_n = (P_train - p_mu) / p_std
    P_val_n = (P_val - p_mu) / p_std

    print(f"[PARAM NORM] mu={p_mu.ravel().tolist()}  std={p_std.ravel().tolist()}")

    # --- Time feature configuration. ---
    time_cfg = resolve_time_features(cfg, times_train)

    # --- Flags. ---
    baseline_mode = str(getattr(cfg, "baseline", getattr(cfg, "baseline_mode", "none")) or "none").lower()
    use_residual = bool(getattr(cfg, "use_residual_corrector", False))
    use_baseline_as_feature = bool(getattr(cfg, "use_baseline_as_feature", False))

    # --- Optional baseline-only path. ---
    # Existing behavior is preserved. This path already attaches user_parameter through the
    # baseline writer/helper stack in older usage. The transient GNN path below now also
    # uses parameter-specific ROM directories.
    if baseline_mode in ("linear", "linear_over_param", "ridge") and not use_residual:
        with log_time("Baseline: linear over parameter (per time)"):
            _preds_file, _times = fit_user_baseline(
                user_param=user_param,
                ref_rev=ref_rev,
                output_dir=cfg.output_dir,
                Y_train=Y_train,
                Y_val=Y_val,
                P_train=P_train,
                P_val=P_val,
                times_train=times_train,
                times_val=times_val,
                colmap_canon=colmap_canon,
                nodes_df=nodes_df,
                field_variable=getattr(cfg, "field_variable", None),
                poly_deg=int(getattr(cfg, "baseline_poly_degree", 1)),
                ridge_alpha=float(getattr(cfg, "baseline_ridge_alpha", 0.0)),
                atol=float(getattr(cfg, "times_match_atol", 1e-8)),
            )
            return None, None

    # --- Residual-corrector path: fit baseline, train GNN on residuals. ---
    baseline_model = None
    base_tr_fields = None
    base_va_fields = None

    if use_residual:
        with log_time("Residual corrector: building per-time baseline on TRAIN/VAL"):
            poly_deg = int(getattr(cfg, "baseline_poly_degree", 1))
            ridge_alpha = float(getattr(cfg, "baseline_ridge_alpha", 0.0))
            atol = float(getattr(cfg, "times_match_atol", 1e-8))

            ret = build_train_val_baseline(
                times_train,
                P_train,
                Y_train,
                times_val,
                P_val,
                Y_val,
                poly_deg=poly_deg,
                ridge_alpha=ridge_alpha,
                atol=atol,
            )

            if isinstance(ret, tuple) and len(ret) == 3:
                W_by_time, baseline_model, _meta = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                W_by_time, _meta = ret
                try:
                    baseline_model = make_baseline_model(
                        W_by_time,
                        poly_deg=poly_deg,
                        ridge_alpha=ridge_alpha,
                        atol=atol,
                    )
                except Exception:
                    baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)
            else:
                W_by_time = ret
                try:
                    baseline_model = make_baseline_model(
                        W_by_time,
                        poly_deg=poly_deg,
                        ridge_alpha=ridge_alpha,
                        atol=atol,
                    )
                except Exception:
                    baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)

            base_tr_fields = None
            if times_train is not None and len(times_train):
                base_tr_fields = baseline_model.predict_samples(times_train, P_train)

            base_va_fields = None
            if times_val is not None and len(times_val):
                base_va_fields = baseline_model.predict_samples(times_val, P_val)

            print(f"[BASELINE] base_tr_fields shape={None if base_tr_fields is None else base_tr_fields.shape}")
            print(f"[BASELINE] base_va_fields shape={None if base_va_fields is None else base_va_fields.shape}")

            if base_tr_fields is None:
                raise RuntimeError("Baseline fields for training could not be constructed.")

            if base_va_fields is None:
                print("[WARN] No baseline fields for validation; using zeros as dummy baseline.")
                base_va_fields = np.zeros_like(Y_val, dtype=np.float32)

            R_tr = (Y_train - base_tr_fields).astype(np.float32)
            R_va = (Y_val - base_va_fields).astype(np.float32)

            print(f"[RESIDUAL RAW] Train mean/std {R_tr.mean():.3e}/{R_tr.std():.3e}")
            print(f"[RESIDUAL RAW] Val   mean/std {R_va.mean():.3e}/{R_va.std():.3e}")
            print(f"[RESIDUAL RAW] Max node std train={np.max(R_tr.std(axis=0)):.3e}, val={np.max(R_va.std(axis=0)):.3e}")
            print(f"[RESIDUAL RAW] Min node std train={np.min(R_tr.std(axis=0)):.3e}, val={np.min(R_va.std(axis=0)):.3e}")

            snap_tr_std = np.std(R_tr, axis=1)
            snap_va_std = np.std(R_va, axis=1)
            print(f"[RESIDUAL SNAP] Train snapshot std range {snap_tr_std.min():.3e}-{snap_tr_std.max():.3e}")
            print(f"[RESIDUAL SNAP] Val   snapshot std range {snap_va_std.min():.3e}-{snap_va_std.max():.3e}")

            pernode = bool(getattr(cfg, "residual_pernode_norm", True))
            eps = 1e-8

            if pernode:
                mu = R_tr.mean(axis=0)  # (N,)
                std = R_tr.std(axis=0) + eps  # (N,)

                print(f"[RESIDUAL NODES] std min={std.min():.3e}, max={std.max():.3e}, mean={std.mean():.3e}")
                small_nodes = int(np.sum(std < 1e-2))
                print(f"[RESIDUAL NODES] nodes with std<1e-2: {small_nodes} out of {std.size}")

                std_safe = np.where(std < 1e-2, 1e-2, std)

                Y_train_n = (R_tr - mu) / std_safe
                Y_val_n = (R_va - mu) / std_safe
                res_mu = mu.astype(np.float32)
                res_std = std_safe.astype(np.float32)

                print("[RESIDUAL] Residual corrector enabled: training GNN on residuals (per-node normalized)")
                print(f"[NORM STATS] std_safe min={std_safe.min():.3e}, max={std_safe.max():.3e}, mean={std_safe.mean():.3e}")
            else:
                mu = float(R_tr.mean())
                std = float(R_tr.std() + eps)

                Y_train_n = (R_tr - mu) / std
                Y_val_n = (R_va - mu) / std
                res_mu = np.array(mu, dtype=np.float32)
                res_std = np.array(std, dtype=np.float32)

                print("[RESIDUAL] Residual corrector enabled: training GNN on residuals (scalar normalized)")

            print(f"[NORM-CHECK] Train residuals mean/std {Y_train_n.mean():.3e}/{Y_train_n.std():.3e}")
            print(f"[NORM-CHECK] Val residuals   mean/std {Y_val_n.mean():.3e}/{Y_val_n.std():.3e}")

    else:
        # No residual path: train directly on scalar-normalized target fields.
        Y_train_n, Y_val_n = Y_train_n_scalar, Y_val_n_scalar
        res_mu = y_mu
        res_std = y_std

        print(f"[NORM-CHECK] Train targets mean/std {Y_train_n.mean():.3e}/{Y_train_n.std():.3e}")
        if Y_val_n is not None and isinstance(Y_val_n, np.ndarray) and Y_val_n.size:
            print(f"[NORM-CHECK] Val targets   mean/std {Y_val_n.mean():.3e}/{Y_val_n.std():.3e}")

    # --- Ensure baseline exists if baseline-as-feature is requested, even without residuals. ---
    if baseline_model is None and use_baseline_as_feature:
        poly_deg = int(getattr(cfg, "baseline_poly_degree", 1))
        ridge_alpha = float(getattr(cfg, "baseline_ridge_alpha", 0.0))
        atol = float(getattr(cfg, "times_match_atol", 1e-8))

        ret = build_train_val_baseline(
            times_train,
            P_train,
            Y_train,
            times_val,
            P_val,
            Y_val,
            poly_deg=poly_deg,
            ridge_alpha=ridge_alpha,
            atol=atol,
        )

        if isinstance(ret, tuple) and len(ret) == 3:
            W_by_time, baseline_model, _meta = ret
        else:
            W_by_time = ret[0] if isinstance(ret, tuple) else ret
            try:
                baseline_model = make_baseline_model(
                    W_by_time,
                    poly_deg=poly_deg,
                    ridge_alpha=ridge_alpha,
                    atol=atol,
                )
            except Exception:
                baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)

    # --- Baseline-as-feature channels for TRAIN/VAL. ---
    B_tr, B_va = _maybe_build_baseline_features(
        use_feature=use_baseline_as_feature,
        baseline_model=baseline_model,
        times_train=times_train,
        times_val=times_val,
        P_train=P_train,
        P_val=P_val,
        user_param=user_param,
    )

    # Model training expects optional baseline arrays shaped (S, N, 1).
    if B_tr is not None and B_tr.ndim == 2:
        B_tr = B_tr[..., None]
    if B_va is not None and B_va.ndim == 2:
        B_va = B_va[..., None]

    # --- Training weights. ---
    node_weights = None
    sample_weights = None
    loss_weighting = str(getattr(cfg, "loss_weighting", "none")).strip().lower()

    if loss_weighting not in (None, "", "none"):
        node_weights, sample_weights = make_training_weights(
            strategy=loss_weighting,
            Y_train=Y_train,
            Y_val=Y_val,
            times_train=times_train,
            times_val=times_val,
            P_train=P_train,
            P_val=P_val,
            W_by_time=None,
        )

    # --- Train or load model. ---
    in_dim = X_node.shape[1] + P_train_n.shape[1] + (time_cfg.t_feat_dim if time_cfg.add_time else 0)
    if B_tr is not None:
        in_dim += 1  # baseline scalar field as extra node feature

    print(
        f"[CHECK] in_dim={in_dim}  X={X_node.shape[1]}  P={P_train_n.shape[1]}  "
        f"t={(time_cfg.t_feat_dim if time_cfg.add_time else 0)}  "
        f"baseline={'yes' if B_tr is not None else 'no'}"
    )

    model = load_or_build_model(
        model_path_pt=model_path_pt,
        in_dim=in_dim,
        conv_type=str(getattr(cfg, "conv_type", "sage")).lower(),
        hidden=int(getattr(cfg, "hidden", 256)),
        dropout=float(getattr(cfg, "dropout", 0.0)),
        gat_heads=int(getattr(cfg, "gat_heads", 4)),
        attn_drop=float(getattr(cfg, "attn_dropout", 0.1)),
        early_stopping=bool(getattr(cfg, "early_stopping", True)),
        es_patience=int(getattr(cfg, "es_patience", 10)),
        es_min_delta=float(getattr(cfg, "es_min_delta", 0.0)),
        es_restore_best=bool(getattr(cfg, "es_restore_best", True)),
        skip_training=bool(getattr(cfg, "skip_training", False)),
    )

    if not (bool(getattr(cfg, "skip_training", False)) and os.path.exists(model_path_pt)):
        with log_time("Training GNN (graph-first)"):
            Y_tr = Y_train_n
            Y_va = Y_val_n

            assert isinstance(Y_tr, np.ndarray) and Y_tr.ndim == 2, (
                f"Y_tr must be (S_train, N); got {None if Y_tr is None else Y_tr.shape}"
            )
            assert isinstance(Y_va, np.ndarray) and Y_va.ndim == 2, (
                f"Y_va must be (S_val, N); got {None if Y_va is None else Y_va.shape}"
            )
            assert Y_tr.shape[1] == Y_va.shape[1], (
                f"Node dim mismatch: Y_tr N={Y_tr.shape[1]} vs Y_va N={Y_va.shape[1]}"
            )

            print(
                f"[DBG] pipeline(normalized): Y_tr mean/std {Y_tr.mean():.3e}/{Y_tr.std():.3e} | "
                f"Y_va mean/std {Y_va.mean():.3e}/{Y_va.std():.3e}"
            )

            shuffle_seed = getattr(cfg, "shuffle_seed", None)

            model, hist = train_gcn_model(
                Y_tr,
                Y_va,
                P_train_n,
                P_val_n,
                edge_index,
                X_node=X_node,
                epochs=int(getattr(cfg, "epochs", 10)),
                lr=float(getattr(cfg, "lr", 1e-3)),
                hidden=int(getattr(cfg, "hidden", 256)),
                dropout=float(getattr(cfg, "dropout", 0.0)),
                lambda_smooth=float(getattr(cfg, "lambda_smooth", 0.0)),
                ignore_head_frac=float(getattr(cfg, "ignore_head_frac", 0.0)),
                shuffle=True,
                shuffle_seed=shuffle_seed,
                device=None,
                add_time=time_cfg.add_time,
                times_train=times_train,
                times_test=times_val,
                time_mode=time_cfg.time_mode,
                fourier_m=time_cfg.fourier_m,
                time_gain=time_cfg.time_gain,
                conv_type=time_cfg.conv_type,
                heads=time_cfg.gat_heads,
                attn_dropout=time_cfg.attn_drop,
                baseline_train=B_tr,
                baseline_val=B_va,
                node_weights=node_weights,
                sample_weights=sample_weights,
            )

            save_state_dict(model, model_path_pt, hist)

    # --- Inference at one or many user parameters. ---
    with log_time("Inference at user_parameter(s): parameter-specific ROM outputs"):
        out_root = Path(cfg.output_dir) / "graph"
        ref_dir = out_root / ref_rev
        targets = list_targets(ref_dir)
        times = np.array([t for t, _ in targets], dtype=float)

        print(f"[INFER] predicting {len(times)} snapshots for user_parameters={user_parameters}")

        # Time statistics must match the training time feature construction.
        if time_cfg.add_time and times_train is not None and len(times_train) > 0:
            t_mu = float(np.mean(times_train))
            t_sigma = float(np.std(times_train)) if np.std(times_train) > 0 else 1.0
            t_min = float(np.min(times_train))
            t_max_train = float(np.max(times_train))
            t_max = t_max_train if t_max_train > t_min else (t_min + 1.0)
        else:
            t_mu = None
            t_sigma = None
            t_min, t_max = 0.0, 1.0

        ei_torch = torch.as_tensor(edge_index, dtype=torch.long)
        X_node_t = X_node.clone() if hasattr(X_node, "clone") else torch.tensor(X_node, dtype=torch.float32)

        # FILE-order node dataframe is independent of operating parameter.
        nodes_df_file = to_file_order(nodes_df, colmap_canon, dataframe=True)

        y_lo = float(np.min(Y_train))
        y_hi = float(np.max(Y_train))
        print(f"[CLIP] global [{y_lo:.6e}, {y_hi:.6e}]")

        for user_param_i in user_parameters:
            print(f"[INFER] user_parameter={user_param_i:.6f}")

            params_row = ((np.array([user_param_i], dtype=np.float32) - p_mu) / p_std).astype(np.float32)

            # Optional baseline feature channel at inference.
            # This is required for baseline+ transient ROMs trained with baseline-as-feature.
            baseline_feats = None
            if use_baseline_as_feature and (baseline_model is not None):
                baseline_feats = baseline_model.predict_user(times, user_param_i)  # (S, N)

            preds_list = []

            for s, t in enumerate(times):
                b_feat_row = baseline_feats[s] if baseline_feats is not None else None

                y = predict_gcn(
                    model,
                    edge_index=ei_torch,
                    X_node=X_node_t,
                    params_row=params_row,
                    add_time=time_cfg.add_time,
                    time_val=(float(t) if time_cfg.add_time else None),
                    time_mode=time_cfg.time_mode,
                    t_min=t_min,
                    t_max=t_max,
                    t_mu=t_mu,
                    t_sigma=t_sigma,
                    fourier_m=time_cfg.fourier_m,
                    time_gain=time_cfg.time_gain,
                    baseline_chan=b_feat_row,
                )

                preds_list.append(y.detach().cpu().numpy())

            preds = np.stack(preds_list, axis=0)  # (S, N)

            # De-normalize. If residual mode is enabled, this is the predicted residual field.
            preds = preds * res_std + res_mu

            # If residual mode, add the user-parameter-specific baseline field back.
            if use_residual and (baseline_model is not None):
                base_user = baseline_model.predict_user(times, user_param_i)
                preds = preds + base_user

            # Clip to training target range.
            np.clip(preds, y_lo, y_hi, out=preds)

            # NODE canonical order -> FILE order.
            preds_file = to_file_order(preds, colmap_canon)

            # Parameter-specific ROM directory for transient ML / baseline+ output.
            out_dir = _rom_param_output_dir(cfg, user_param_i)
            print(f"[WRITE] user_parameter={user_param_i:.6f} -> {out_dir}")

            _write_rom_only(
                preds_file,
                times,
                nodes_df_file,
                getattr(cfg, "field_variable", None),
                out_dir,
            )

            _write_inference_metadata(
                out_dir=out_dir,
                user_param=user_param_i,
                cfg=cfg,
                baseline_mode=baseline_mode,
                use_residual=use_residual,
                use_baseline_as_feature=use_baseline_as_feature,
                time_cfg=time_cfg,
                model_path_pt=Path(model_path_pt),
                train_param_min=pmin,
                train_param_max=pmax,
                y_min=y_lo,
                y_max=y_hi,
                n_nodes=N,
                n_edges=E,
                n_snapshots=len(times),
            )

    return None, None
