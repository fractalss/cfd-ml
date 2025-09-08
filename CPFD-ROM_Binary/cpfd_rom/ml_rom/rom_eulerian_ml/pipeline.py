# ===============================
# File: cpfd_rom/ml_rom/rom_eulerian_ml/pipeline_refactored/pipeline.py
# Purpose: Thin orchestrator that wires together modular steps
# ===============================
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple
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
    make_baseline_model,  # may not exist in older drops; we'll fall back if missing
)
# loss weighting
from .steps.weights import make_training_weights
# model i/o helpers
from .steps.modelio import torch_model_path, load_or_build_model, save_state_dict
from .evaluation import _write_rom_only


# ------------------------------
# Local helpers  baseline (robust to API variants)
# ------------------------------

def _poly_design(p: float, deg: int) -> np.ndarray:
    """Return [1, p, p^2, ..., p^deg] as (D,) float32."""
    d = [1.0]
    for k in range(1, int(deg) + 1):
        d.append(float(p) ** k)
    return np.array(d, dtype=np.float32)


def _param_scalar(P: np.ndarray) -> np.ndarray:
    """Extract a scalar parameter per-sample from P.
    Accepts (S,Pd) or (S,N,Pd). Returns (S,) float32 using the first column; for node-wise,
    averages the first column across nodes.
    """
    P = np.asarray(P)
    if P.ndim == 2:  # (S,Pd)
        return P[:, 0].astype(np.float32)
    if P.ndim == 3:  # (S,N,Pd)
        return P[:, :, 0].mean(axis=1).astype(np.float32)
    raise ValueError(f"Bad P shape {P.shape}; expected (S,Pd) or (S,N,Pd)")


class _BaselineFromW:
    """Lightweight baseline wrapper built from W_by_time (per-time polynomial fit).
    Exposes the two methods used by this pipeline:
       predict_samples(times, P) -> (S,N)
       predict_user(times, user_param) -> (S,N)
    """
    def __init__(self, W_by_time: dict[float, np.ndarray], poly_deg: int, atol: float = 1e-8):
        self.W_by_time = {float(k): np.asarray(v) for k, v in W_by_time.items()}
        self.poly_deg = int(poly_deg)
        self.atol = float(atol)
        self._times = np.array(sorted(self.W_by_time.keys()), dtype=np.float64)

    def _select_time(self, t: float) -> float:
        # Prefer exact (within atol); otherwise nearest neighbor
        diffs = np.abs(self._times - float(t))
        idx = int(np.argmin(diffs))
        if diffs[idx] <= self.atol:
            return float(self._times[idx])
        return float(self._times[idx])

    def _eval_single(self, t: float, p_scalar: float) -> np.ndarray:
        t_key = self._select_time(t)
        W = self.W_by_time[t_key]  # (D,N)
        d = _poly_design(p_scalar, self.poly_deg).reshape(-1)  # (D,)
        return (d @ W).astype(np.float32)  # (N,)

    def predict_samples(self, times: np.ndarray, P: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float32).reshape(-1)
        p = _param_scalar(P)
        S = times.shape[0]
        # infer N from first eval
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
    """Return (B_train, B_val) where each is (S, N) baseline field or None.
    If use_feature is False or baseline_model is None, returns (None, None).
    """
    if not use_feature or baseline_model is None:
        return None, None

    B_train = baseline_model.predict_samples(times_train, P_train) if (times_train is not None and len(times_train)) else None
    B_val = None
    if times_val is not None and len(times_val):
        B_val = baseline_model.predict_samples(times_val, P_val)
    return B_train, B_val


def run_ml_rom_pipeline(cfg, log_time):
    """Main entry point kept intentionally thin; heavy lifting dispatched to modules."""
    # --- Setup ---
    setup_output_dir(cfg)
    setup_model_paths(cfg)
    model_path_pt = torch_model_path(cfg.model_path_eulerian)

    # --- Build canonical graph + datasets (first rev defines graph) ---
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

    # --- FILE?NODE mapping based on coords_file_order (if available) ---
    coords_path = Path(ref_graph_dir) / "coords_file_order.parquet"
    coords_file_df = None
    if coords_path.exists():
        try:
            import pandas as pd
            coords_file_df = pd.read_parquet(coords_path)[["i","j","k"]]
            print("[ALIGN] Using coords_file_order.parquet for FILE?NODE mapping")
        except Exception as e:
            print(f"[ALIGN] WARNING: failed to read {coords_path.name}: {e}")

    if coords_file_df is None:
        # fallback: try cfg-specified paths under ref_graph_dir
        coords_file_df = maybe_load_coords_file_df(
            cfg,
            Path(ref_graph_dir) if ref_graph_dir else None,
            N_cols=(Y_train.shape[1] if Y_train.size else Y_val.shape[1])
        )

    colmap_file_to_node = build_file_to_node_colmap(coords_file_df, nodes_df, Y_train)

    # --- Canonicalize (i,j,k) ordering and rebuild X_node accordingly ---
    with log_time("Canonicalizing (i,j,k) order for nodes, Y_*, and edges"):
        nodes_df, Y_train, Y_val, edge_index, X_node, colmap_canon = canonicalize_all(
            nodes_df,
            Y_train,
            Y_val,
            edge_index,
            build_node_features_xyz,
            colmap_file_to_node,
        )

    # --- Dataset stats & parameter range ---
    N = int(Y_train.shape[1]) if Y_train.size else 0
    E = int(edge_index.shape[1])
    S_tr, S_va = int(Y_train.shape[0]), int(Y_val.shape[0])
    P_dim = int(P_train.shape[1]) if P_train.size else 0
    print(f"[DATA] N(nodes)={N}  E(edges)={E}  S_train={S_tr}  S_val={S_va}  P_dim={P_dim}")
    print(f"[DATA] X_node shape={tuple(X_node.shape)}  Y_train shape={tuple(Y_train.shape)}  P_train shape={tuple(P_train.shape)}")

    user_param = float(getattr(cfg, "user_parameter", 0.0))
    pmin, pmax = float(P_train.min()), float(P_train.max())
    print(f"[PARAM] train_range=[{pmin:.6f},{pmax:.6f}]  user={user_param:.6f}  inside_range={pmin <= user_param <= pmax}")

    # --- Normalization (target + param) ---
    y_mu = float(np.mean(Y_train))
    y_std = float(np.std(Y_train)) if float(np.std(Y_train)) > 0 else 1.0
    print(f"[Y-NORM] mu={y_mu:.6e} std={y_std:.6e} (scalar)")
    Y_train_n_scalar = (Y_train - y_mu) / y_std
    Y_val_n_scalar   = (Y_val   - y_mu) / y_std if Y_val.size else Y_val

    p_mu  = P_train.mean(axis=0)
    p_std = P_train.std(axis=0)
    p_std = np.where(p_std > 0, p_std, 1.0)
    P_train_n = (P_train - p_mu) / p_std
    P_val_n   = (P_val   - p_mu) / p_std
    print(f"[PARAM NORM] mu={p_mu.ravel().tolist()}  std={p_std.ravel().tolist()}")

    # --- Time feature configuration ---
    time_cfg = resolve_time_features(cfg, times_train)

    # --- Flags ---
    baseline_mode = str(getattr(cfg, "baseline", getattr(cfg, "baseline_mode", "none")) or "none").lower()
    use_residual  = bool(getattr(cfg, "use_residual_corrector", False))
    use_baseline_as_feature = bool(getattr(cfg, "use_baseline_as_feature", False))

    # --- Optional baseline-only path ---
    if baseline_mode in ("linear", "linear_over_param", "ridge") and not use_residual:
        with log_time("Baseline: linear over parameter (per time)"):
            preds_file, times = fit_user_baseline(
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

    # --- Residual-corrector (fit baseline on train/val, train GNN on residuals) ---
    baseline_model = None
    base_tr_fields = None
    base_va_fields = None
    if bool(getattr(cfg, "use_residual_corrector", False)):
        with log_time("Residual corrector: building per-time baseline on TRAIN/VAL"):
            poly_deg = int(getattr(cfg, "baseline_poly_degree", 1))
            ridge_alpha = float(getattr(cfg, "baseline_ridge_alpha", 0.0))
            atol = float(getattr(cfg, "times_match_atol", 1e-8))

            # Updated baseline helper returns dict or (dict, model, meta). Support all.
            ret = build_train_val_baseline(
                times_train, P_train, Y_train,
                times_val,   P_val,   Y_val,
                poly_deg=poly_deg,
                ridge_alpha=ridge_alpha,
                atol=atol,
            )
            if isinstance(ret, tuple) and len(ret) == 3:
                W_by_time, baseline_model, _meta = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                W_by_time, _meta = ret
                try:
                    baseline_model = make_baseline_model(W_by_time, poly_deg=poly_deg, ridge_alpha=ridge_alpha, atol=atol)
                except Exception:
                    baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)
            else:  # just dict
                W_by_time = ret
                try:
                    baseline_model = make_baseline_model(W_by_time, poly_deg=poly_deg, ridge_alpha=ridge_alpha, atol=atol)
                except Exception:
                    baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)

            # Compute residuals directly (robust to baseline API variants)
            base_tr_fields = baseline_model.predict_samples(times_train, P_train) if (times_train is not None and len(times_train)) else None
            base_va_fields = baseline_model.predict_samples(times_val,   P_val)   if (times_val   is not None and len(times_val))   else None
            print(f"[BASELINE] base_tr_fields shape={None if base_tr_fields is None else base_tr_fields.shape}")
            print(f"[BASELINE] base_va_fields shape={None if base_va_fields is None else base_va_fields.shape}")

            if base_tr_fields is None:
                raise RuntimeError("Baseline fields for training could not be constructed.")
            if base_va_fields is None:
                print("[WARN] No baseline fields for validation; using zeros as dummy baseline.")
                base_va_fields = np.zeros_like(Y_val, dtype=np.float32)
            R_tr = (Y_train - base_tr_fields).astype(np.float32)
            R_va = (Y_val   - base_va_fields).astype(np.float32) if base_va_fields is not None else Y_val
            # --- Extra debugging ---
            print(f"[RESIDUAL RAW] Train mean/std {R_tr.mean():.3e}/{R_tr.std():.3e}")
            print(f"[RESIDUAL RAW] Val   mean/std {R_va.mean():.3e}/{R_va.std():.3e}")
            print(
                f"[RESIDUAL RAW] Max node std train={np.max(R_tr.std(axis=0)):.3e}, val={np.max(R_va.std(axis=0)):.3e}")
            print(
                f"[RESIDUAL RAW] Min node std train={np.min(R_tr.std(axis=0)):.3e}, val={np.min(R_va.std(axis=0)):.3e}")
            # Per-snapshot variance check
            snap_tr_std = np.std(R_tr, axis=1)
            snap_va_std = np.std(R_va, axis=1)
            print(f"[RESIDUAL SNAP] Train snapshot std range {snap_tr_std.min():.3e}{snap_tr_std.max():.3e}")
            print(f"[RESIDUAL SNAP] Val   snapshot std range {snap_va_std.min():.3e}{snap_va_std.max():.3e}")

            # Choose normalization strategy
            pernode = bool(getattr(cfg, "residual_pernode_norm", True))
            eps = 1e-8
            if pernode:
                mu = R_tr.mean(axis=0)         # (N,)
                std = R_tr.std(axis=0) + eps   # (N,)
                # Log distribution of stds
                print(f"[RESIDUAL NODES] std min={std.min():.3e}, max={std.max():.3e}, mean={std.mean():.3e}")
                small_nodes = np.sum(std < 1e-2)
                print(f"[RESIDUAL NODES] nodes with std<1e-2: {small_nodes} out of {std.size}")

                # Apply floor to std to avoid division by tiny values
                std_safe = np.where(std < 1e-2, 1e-2, std)

                Y_train_n = (R_tr - mu) / std_safe
                Y_val_n = (R_va - mu) / std_safe
                res_mu = mu.astype(np.float32)
                res_std = std_safe.astype(np.float32)
                # Y_train_n = (R_tr - mu) / std
                # Y_val_n   = (R_va - mu) / std
                # res_mu = mu.astype(np.float32)
                # res_std = std.astype(np.float32)
                print(" Residual corrector enabled: training GNN on residuals (per-node normalized)")
                # Debugging after applying floor
                print(f"[NORM STATS] std_safe min={std_safe.min():.3e}, max={std_safe.max():.3e}, mean={std_safe.mean():.3e}")
            else:
                mu = float(R_tr.mean())
                std = float(R_tr.std() + eps)
                Y_train_n = (R_tr - mu) / std
                Y_val_n   = (R_va - mu) / std
                res_mu = np.array(mu, dtype=np.float32)
                res_std = np.array(std, dtype=np.float32)
                print(" Residual corrector enabled: training GNN on residuals (scalar normalized)")
            print(f"[NORM-CHECK] Train residuals mean/std {Y_train_n.mean():.3e}/{Y_train_n.std():.3e}")
            print(f"[NORM-CHECK] Val residuals   mean/std {Y_val_n.mean():.3e}/{Y_val_n.std():.3e}")
    else:
        # No residual path: use scalar normalized targets directly
        Y_train_n, Y_val_n = Y_train_n_scalar, Y_val_n_scalar
        res_mu = y_mu
        res_std = y_std
        # Post-normalization check for scalar mode
        print(f"[NORM-CHECK] Train targets mean/std {Y_train_n.mean():.3e}/{Y_train_n.std():.3e}")
        if Y_val_n is not None and isinstance(Y_val_n, np.ndarray) and Y_val_n.size:
            print(f"[NORM-CHECK] Val targets   mean/std {Y_val_n.mean():.3e}/{Y_val_n.std():.3e}")

    # --- Ensure baseline model exists when baseline-as-feature is requested (even if not using residuals) ---
    if baseline_model is None and use_baseline_as_feature:
        poly_deg = int(getattr(cfg, "baseline_poly_degree", 1))
        ridge_alpha = float(getattr(cfg, "baseline_ridge_alpha", 0.0))
        atol = float(getattr(cfg, "times_match_atol", 1e-8))
        ret = build_train_val_baseline(
            times_train, P_train, Y_train,
            times_val,   P_val,   Y_val,
            poly_deg=poly_deg,
            ridge_alpha=ridge_alpha,
            atol=atol,
        )
        if isinstance(ret, tuple) and len(ret) == 3:
            W_by_time, baseline_model, _meta = ret
        else:
            W_by_time = ret[0] if isinstance(ret, tuple) else ret
            try:
                baseline_model = make_baseline_model(W_by_time, poly_deg=poly_deg, ridge_alpha=ridge_alpha, atol=atol)
            except Exception:
                baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)

    # --- Baseline-as-feature channels for TRAIN/VAL (optional) ---
    B_tr, B_va = _maybe_build_baseline_features(
        use_feature=use_baseline_as_feature,
        baseline_model=baseline_model,
        times_train=times_train,
        times_val=times_val,
        P_train=P_train,
        P_val=P_val,
        user_param=user_param,
    )

    # Ensure shapes are (S,N,1) if we will pass into model_gnn
    if B_tr is not None and B_tr.ndim == 2:
        B_tr = B_tr[..., None]
    if B_va is not None and B_va.ndim == 2:
        B_va = B_va[..., None]

    # --- Training weights (optional) ---
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
            W_by_time=None,  # weights module doesn't require baseline weights currently
        )

    # --- Train or load model ---
    in_dim = X_node.shape[1] + P_train_n.shape[1] + (time_cfg.t_feat_dim if time_cfg.add_time else 0)
    if B_tr is not None:
        in_dim += 1  # baseline scalar field as an extra channel per node

    print(
        f"[CHECK] in_dim={in_dim}  X={X_node.shape[1]}  P={P_train_n.shape[1]}  "
        f"t={(time_cfg.t_feat_dim if time_cfg.add_time else 0)}  baseline={'yes' if B_tr is not None else 'no'}"
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
            Y_va = Y_val_n  # always use normalized VAL
            # Correct shape checks: allow different #samples, require same N
            assert isinstance(Y_tr,
                              np.ndarray) and Y_tr.ndim == 2, f"Y_tr must be (S_train, N); got {None if Y_tr is None else Y_tr.shape}"
            assert isinstance(Y_va,
                              np.ndarray) and Y_va.ndim == 2, f"Y_va must be (S_val, N); got {None if Y_va is None else Y_va.shape}"
            assert Y_tr.shape[1] == Y_va.shape[
                1], f"Node dim mismatch: Y_tr N={Y_tr.shape[1]} vs Y_va N={Y_va.shape[1]}"

            print(f"[DBG] pipeline(normalized): Y_tr mean/std {Y_tr.mean():.3e}/{Y_tr.std():.3e} | "
                  f"Y_va mean/std {Y_va.mean():.3e}/{Y_va.std():.3e}")

            shuffle_seed = getattr(cfg, "shuffle_seed", None)
            model, hist = train_gcn_model(
                Y_tr,
                Y_va,
                P_train_n,
                P_val_n,
                edge_index,
                X_node=X_node,
                epochs=int(getattr(cfg, "epochs", 300)),
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
                # new extras
                baseline_train=B_tr,
                baseline_val=B_va,
                node_weights=node_weights,
                sample_weights=sample_weights,
            )
            save_state_dict(model, model_path_pt, hist)

    # --- Inference at user parameter (using reference times) ---
    with log_time("Inference at user_parameter (ROM-only outputs)"):
        out_root = Path(cfg.output_dir) / "graph"
        ref_dir = out_root / ref_rev
        targets = list_targets(ref_dir)
        times = np.array([t for t, _ in targets], dtype=float)
        print(f"[INFER] predicting {len(times)} snapshots for user_parameter={user_param:.6f}")

        # time stats for embedding
        if time_cfg.add_time and times_train is not None and len(times_train) > 0:
            t_mu = float(np.mean(times_train))
            t_sigma = float(np.std(times_train)) if np.std(times_train) > 0 else 1.0
            t_min = float(np.min(times_train))
            t_max = float(np.max(times_train)) if float(np.max(times_train)) > t_min else (t_min + 1.0)
        else:
            t_mu = t_sigma = None
            t_min, t_max = 0.0, 1.0

        ei_torch = torch.as_tensor(edge_index, dtype=torch.long)
        X_node_t = X_node.clone() if hasattr(X_node, "clone") else torch.tensor(X_node, dtype=torch.float32)
        params_row = ((np.array([user_param], dtype=np.float32) - p_mu) / p_std).astype(np.float32)

        # Optional baseline feature channel at inference
        baseline_feats = None
        if use_baseline_as_feature and (baseline_model is not None):
            baseline_feats = baseline_model.predict_user(times, user_param)  # (S, N)

        preds_list = []
        for s, t in enumerate(times):
            # optional baseline feature for this time
            b_feat_row = None
            if baseline_feats is not None:
                b_feat_row = baseline_feats[s]

                # Call predict_gcn without unsupported kwargs
                if b_feat_row is not None:
                    # Concatenate baseline row manually as extra node feature if needed
                    X_node_aug = torch.cat([
                        X_node_t,
                        torch.tensor(b_feat_row, dtype=torch.float32).unsqueeze(1)
                    ], dim=1)
                else:
                    X_node_aug = X_node_t

            y = predict_gcn(
                model,
                # X_node=X_node_t,
                X_node=X_node_aug,
                params_row=params_row,
                edge_index=ei_torch,
                add_time=time_cfg.add_time,
                time_val=(float(t) if time_cfg.add_time else None),
                time_mode=time_cfg.time_mode,
                t_min=t_min,
                t_max=t_max,
                t_mu=t_mu,
                t_sigma=t_sigma,
                fourier_m=time_cfg.fourier_m,
                time_gain=time_cfg.time_gain,
                # baseline_row=b_feat_row,
            )
            preds_list.append(y.detach().cpu().numpy())

        preds = np.stack(preds_list, axis=0)  # (S,N) residual (normalized space if residual mode)

        # De/renormalize
        preds = preds * res_std + res_mu

        # If residual mode, add back baseline field to get final field
        if use_residual and (baseline_model is not None):
            base_user = baseline_model.predict_user(times, user_param)
            preds = preds + base_user

        # clip to training target range
        y_lo = float(np.min(Y_train))
        y_hi = float(np.max(Y_train))
        print(f"[CLIP] global [{y_lo:.6e}, {y_hi:.6e}]")
        np.clip(preds, y_lo, y_hi, out=preds)

        # NODE(canonical) ? FILE order & write
        preds_file = to_file_order(preds, colmap_canon)
        nodes_df_file = to_file_order(nodes_df, colmap_canon, dataframe=True)
        out_dir = Path(cfg.output_dir) / "ML" / f"ROM_param_{float(cfg.user_parameter):.3f}"
        _write_rom_only(preds_file, times, nodes_df_file, getattr(cfg, "field_variable", None), out_dir)

    return None, None
