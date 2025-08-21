# GCN-based Eulerian ROM pipeline (PyTorch + PyG)
# Graph-first flow:
#  - Build ONE canonical graph from a reference training directory (first of rev_dirs)
#  - Assemble training targets from ALL rev_dirs with their parameter values
#  - Train GNN on [XYZ(+time)+param] -> field
#  - Inference for user_parameter using the same canonical graph & the reference times

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir

# Loaders & helpers (graph-first)
from .loader import prepare_graph_and_datasets, list_targets

# Model + training + inference
from cpfd_rom.ml_rom.rom_eulerian_ml.model_gnn import (
    train_gcn_model,
    build_gcn_model,
    _time_feature_dim,
    build_node_features_xyz,
)
from cpfd_rom.ml_rom.rom_eulerian_ml.inference import predict_gcn
from cpfd_rom.util.utils_alignment import (
    canonicalize_by_ijk,
    build_colmap_by_join,
    to_file_order,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _torch_model_path(base_path: str) -> str:
    if base_path.endswith(".keras"):
        return base_path[:-6] + ".pt"
    if not base_path.endswith(".pt"):
        return base_path + ".pt"
    return base_path


def _maybe_load_coords_file_df(cfg, ref_graph_dir: Optional[Path], N_cols: int) -> Optional[pd.DataFrame]:
    """Try to obtain a DataFrame describing FILE/FLATTEN order (one row per Y column).
    Expected columns: ['i','j','k'].
    Priority:
      1) cfg.coords_file_parquet / cfg.coords_file_csv (explicit path)
      2) ref_graph_dir / 'coords_file_order.parquet' or '.csv' (if exists)
    Returns None if nothing found.
    """
    # 1) explicit paths in cfg
    p_parq = getattr(cfg, 'coords_file_parquet', None)
    p_csv  = getattr(cfg, 'coords_file_csv', None)
    try_paths = []
    if p_parq:
        try_paths.append(Path(p_parq))
    if p_csv:
        try_paths.append(Path(p_csv))

    # 2) conventional filenames under ref_graph_dir
    if ref_graph_dir is not None:
        try_paths.append(Path(ref_graph_dir) / 'coords_file_order.parquet')
        try_paths.append(Path(ref_graph_dir) / 'coords_file_order.csv')

    for p in try_paths:
        try:
            if p.suffix.lower() == '.parquet' and p.exists():
                df = pd.read_parquet(p)
            elif p.suffix.lower() == '.csv' and p.exists():
                df = pd.read_csv(p)
            else:
                continue
            # basic validation
            if set(['i','j','k']).issubset(df.columns) and len(df) == N_cols:
                print(f"[ALIGN] Using coords_file_df from {p}")
                return df[['i','j','k']].copy()
        except Exception as e:
            print(f"[ALIGN] Failed to load coords_file_df from {p}: {e}")
    print("[ALIGN] WARNING: coords_file_df not provided/found; join-based reorder will be skipped.")
    return None


# -----------------------------------------------------------------------------
# Main pipeline entry
# -----------------------------------------------------------------------------


def run_ml_rom_pipeline(cfg, log_time):
    # Basic setup
    setup_output_dir(cfg)
    setup_model_paths(cfg)
    model_path_pt = _torch_model_path(cfg.model_path_eulerian)

    # Build canonical graph from FIRST rev, assemble datasets from ALL revs
    with log_time("Preparing graph & datasets from rev_dirs"):
        (
            ref_rev,
            ref_graph_dir,
            nodes_df,
            edge_index,
            X_node,
            Y_train,
            Y_val,
            P_train,
            P_val,
            times_train,
            times_val,
        ) = prepare_graph_and_datasets(cfg.__dict__, neighbor_set="n6")

        # ---------------- Alignment Step 1: FILE -> NODE mapping (no Y reordering here) ----------------
    coords_path = Path(ref_graph_dir) / "coords_file_order.parquet"
    colmap_path = Path(ref_graph_dir) / "colmap_file_to_nodes.npy"

    if colmap_path.exists():
        colmap = np.load(colmap_path)
        print(f"[ALIGN] Loaded FILE?NODE colmap from {colmap_path.name} (len={len(colmap)})")
    else:
        colmap = None
        coords_file_df = None
        if coords_path.exists():
            try:
                coords_file_df = pd.read_parquet(coords_path)[['i','j','k']]
                print("[ALIGN] Using coords_file_order.parquet for FILE?NODE mapping")
            except Exception as e:
                print(f"[ALIGN] WARNING: failed to read {coords_path.name}: {e}")
        else:
            # fallback to cfg-specified path or helper
            maybe_df = _maybe_load_coords_file_df(cfg, Path(ref_graph_dir) if ref_graph_dir else None,
                                                  N_cols=(Y_train.shape[1] if Y_train.size else Y_val.shape[1]))
            coords_file_df = maybe_df

        if coords_file_df is not None:
            colmap = build_colmap_by_join(coords_file_df, nodes_df)
            print(f"[ALIGN] Built FILE?NODE colmap via (i,j,k) join (len={len(colmap)})")
        else:
            colmap = np.arange(len(nodes_df), dtype=np.int64)
            print("[ALIGN] WARNING: No FILE?NODE mapping found; assuming identity (Y_* already in NODE order).")

    # --- ALIGNMENT: canonicalize node/target/edge order to lexicographic (i,j,k)
    with log_time("Canonicalizing (i,j,k) order for nodes, Y_*, and edges"):
        nodes_df, Y_train, edge_index, order, inv = canonicalize_by_ijk(nodes_df, Y_train, edge_index)
        if Y_val.size:
            Y_val = Y_val[:, order]
        # Compose FILE->CANONICAL mapping for later write-out
        colmap_canon = order[colmap]
        # Optional verification that FILE?NODE mapping is consistent
        try:
            coords_path = Path(ref_graph_dir) / "coords_file_order.parquet"
            if coords_path.exists():
                coords_file_df_chk = pd.read_parquet(coords_path)[['i','j','k']].reset_index(drop=True)
                inv_map = np.empty_like(colmap_canon); inv_map[colmap_canon] = np.arange(colmap_canon.size)
                nodes_df_file_chk = nodes_df.iloc[inv_map][['i','j','k']].reset_index(drop=True)
                if (coords_file_df_chk.values == nodes_df_file_chk.values).all():
                    print("[ALIGN] Verified FILE?NODE mapping via (i,j,k).")
                else:
                    print("[ALIGN] WARNING: FILE?NODE mapping verification failed; check ijk conventions and masks.")
        except Exception as e:
            print(f"[ALIGN] Mapping verification skipped: {e}")
        # Rebuild static node features AFTER reordering nodes_df
        X_node = build_node_features_xyz(nodes_df)
        print(f"[ALIGN] Applied canonical (i,j,k) order. Perm size={len(order)}. Rebuilt X_node: {tuple(X_node.shape)}")

    # ---------------------------
    # Debug & sanity: dataset stats
    # ---------------------------
    N = int(Y_train.shape[1]) if Y_train.size else 0
    E = int(edge_index.shape[1]) if isinstance(edge_index, np.ndarray) else int(edge_index.shape[1])
    S_tr, S_va = int(Y_train.shape[0]), int(Y_val.shape[0])
    P_dim = int(P_train.shape[1]) if P_train.size else 0

    print(f"[DATA] N(nodes)={N}  E(edges)={E}  S_train={S_tr}  S_val={S_va}  P_dim={P_dim}")
    print(f"[DATA] X_node shape={tuple(X_node.shape)}  Y_train shape={tuple(Y_train.shape)}  P_train shape={tuple(P_train.shape)}")

    # Parameter coverage vs user param
    user_param = float(getattr(cfg, "user_parameter", 0.0))
    pmin, pmax = float(P_train.min()), float(P_train.max())
    print(f"[PARAM] train_range=[{pmin:.6f},{pmax:.6f}]  user={user_param:.6f}  inside_range={pmin <= user_param <= pmax}")

    # Time coverage
    if times_train is None or len(times_train) == 0:
        print("[TIME] No times detected (time_mode should be 'none' or snapshots are time-agnostic)")
    else:
        tmin, tmax, tstd = float(np.min(times_train)), float(np.max(times_train)), float(np.std(times_train))
        print(f"[TIME] train min={tmin:.6f}  max={tmax:.6f}  std={tstd:.6f}")
        uniq_preview = np.unique(times_train[:min(20, len(times_train))])[:5]
        print(f"[TIME] sample train times: {uniq_preview.tolist()}")

    # Target magnitude overview
    y_glob_mu = float(np.mean(Y_train))
    y_glob_std = float(np.std(Y_train))
    print(f"[Y] global mean={y_glob_mu:.6e}  std={y_glob_std:.6e}")

    # ---- Option A: Target normalization (scalar) ----
    y_mu  = float(y_glob_mu)
    y_std = float(y_glob_std) if float(y_glob_std) > 0 else 1.0
    print(f"[Y-NORM] mu={y_mu:.6e} std={y_std:.6e} (scalar)")
    Y_train_n = (Y_train - y_mu) / y_std
    Y_val_n   = (Y_val   - y_mu) / y_std if Y_val.size else Y_val

    # --- normalize the parameter feature(s) using training stats ---
    p_mu  = P_train.mean(axis=0)
    p_std = P_train.std(axis=0)
    p_std = np.where(p_std > 0, p_std, 1.0)  # avoid divide-by-zero
    P_train_n = (P_train - p_mu) / p_std
    P_val_n   = (P_val   - p_mu) / p_std
    print(f"[PARAM NORM] mu={p_mu.ravel().tolist()}  std={p_std.ravel().tolist()}")

    # Baseline (mean-field) validation MSE for reference
    if S_va > 0:
        ybar = Y_train.mean(axis=0)
        baseline_val_mse = float(((Y_val - ybar) ** 2).mean())
        print(f"[BASELINE] val_MSE (predict train-mean) = {baseline_val_mse:.6e}")

    # --- GT time-variation diagnostics (A): compare mean field at tmin vs tmax in training ---
    try:
        if times_train is not None and len(times_train) > 0:
            _tmin = float(np.min(times_train)); _tmax = float(np.max(times_train))
            idx_min = np.where(np.isclose(times_train, _tmin))[0]
            idx_max = np.where(np.isclose(times_train, _tmax))[0]
            if idx_min.size and idx_max.size:
                y_min_mean = Y_train[idx_min].mean(axis=0)
                y_max_mean = Y_train[idx_max].mean(axis=0)
                gt_delta_mean = np.abs(y_min_mean - y_max_mean).mean()
                print(f"[DEBUG-GT] mean|Y(tmin)-Y(tmax)| = {gt_delta_mean:.6e}  (tmin={_tmin:.3f}, tmax={_tmax:.3f}, n_min={idx_min.size}, n_max={idx_max.size})")
            else:
                print("[DEBUG-GT] missing tmin/tmax snapshots in training split for a direct compare")
    except Exception as e:
        print(f"[DEBUG-GT] time-variation diagnostic failed: {e}")

    # Configure time conditioning (Option A: robust read + raw debug)
    print("[CFG-RAW]",
          "add_time=", getattr(cfg, "add_time", None),
          "time_mode=", repr(getattr(cfg, "time_mode", None)),
          "fourier_m=", getattr(cfg, "fourier_m", None),
          "gcn=", getattr(cfg, "gcn", None))

    time_mode = str(getattr(cfg, "time_mode", "none")).strip().lower()
    fourier_m = int(getattr(cfg, "fourier_m", 4))
    _add_time_attr = getattr(cfg, "add_time", None)
    add_time = bool(_add_time_attr) if _add_time_attr is not None else (time_mode != "none")
    t_feat_dim = _time_feature_dim(time_mode, fourier_m) if add_time else 0
    # Time feature gain (amplify time block in features)
    time_gain = float(getattr(cfg, "time_gain", 1.0))

    # ----- GNN backbone selection (defaults: GraphSAGE) -----
    conv_type = getattr(cfg, "conv_type", "sage")  # "sage" | "gcn" | "gat" | "gin"
    gat_heads = getattr(cfg, "gat_heads", 4)
    attn_drop = getattr(cfg, "attn_dropout", 0.1)

    extra = {}
    if str(conv_type).lower() == "gat":
        extra.update({"heads": int(gat_heads), "attn_dropout": float(attn_drop)})

    print(f"[CFG] conv_type={conv_type}  add_time={add_time}  time_mode={time_mode}  fourier_m={fourier_m}  t_feat_dim={t_feat_dim}  time_gain={time_gain}")
    if extra:
        print(f"[CFG] GAT extras: {extra}")

    # Expected model input dim for debug
    in_dim_expected = X_node.shape[1] + P_train_n.shape[1] + t_feat_dim
    print(f"[MODEL] expected in_dim={in_dim_expected}")

    # ---- Early Stopping config (pipeline -> trainer) ----
    early_stop = bool(getattr(cfg, "early_stopping", True))
    es_patience = int(getattr(cfg, "es_patience", 20))
    es_min_delta = float(getattr(cfg, "es_min_delta", 0.0))
    es_restore_best = bool(getattr(cfg, "es_restore_best", True))
    print(f"[ES] enabled={early_stop} patience={es_patience} min_delta={es_min_delta} restore_best={es_restore_best}")

    # -----------------------------
    # Baseline: linear over parameter (per time)
    # -----------------------------
    baseline = str(getattr(cfg, "baseline", getattr(cfg, "baseline_mode", "none")) or "none").lower()
    if baseline in ("linear", "linear_over_param", "ridge"):
        with log_time("Baseline: linear over parameter (per time)"):
            # Collect all samples (train + val) for robust fits
            if times_train is None:
                raise RuntimeError("Baseline requires times_train (per-snapshot times)")
            T_all = times_train if times_val is None else np.concatenate([times_train, times_val], axis=0)
            P_all = P_train if P_val is None else np.concatenate([P_train, P_val], axis=0)  # original scale
            Y_all = Y_train if not Y_val.size else np.concatenate([Y_train, Y_val], axis=0)

            # target times = reference rev times (what we will write out)
            out_root = Path(cfg.output_dir) / "graph"
            ref_dir = out_root / ref_rev
            targets_ref = list_targets(ref_dir)
            times_target = np.array([t for t, _ in targets_ref], dtype=float)

            # design matrix options
            poly_deg = int(getattr(cfg, "baseline_poly_degree", 1))  # 1: linear, 2: quadratic
            ridge_alpha = float(getattr(cfg, "baseline_ridge_alpha", 0.0))
            atol = float(getattr(cfg, "times_match_atol", 1e-8))

            preds_list = []
            used_times = []
            for t in times_target:
                idx = np.where(np.isclose(T_all, float(t), atol=atol))[0]
                if idx.size < 2:
                    # Not enough rev coverage at this exact time -> skip
                    # (You can relax this to nearest-time pooling if desired.)
                    continue
                p = P_all[idx].reshape(-1, 1)  # (R,1)
                if poly_deg >= 2:
                    X = np.hstack([p**2, p, np.ones_like(p)])  # (R,3)
                else:
                    X = np.hstack([p, np.ones_like(p)])       # (R,2)
                Yt = Y_all[idx, :]                            # (R,N)

                # Solve multioutput ridge/linear: W = (X^T X + aI)^-1 X^T Y
                Xt = X.T
                XtX = Xt @ X
                if ridge_alpha > 0:
                    XtX = XtX + ridge_alpha * np.eye(X.shape[1])
                W = np.linalg.solve(XtX, Xt @ Yt)             # (D,N)

                # predict at user parameter (original scale)
                up = np.array([[user_param]], dtype=float)
                if poly_deg >= 2:
                    x_star = np.hstack([up**2, up, np.ones_like(up)])  # (1,3)
                else:
                    x_star = np.hstack([up, np.ones_like(up)])         # (1,2)
                y_hat = (x_star @ W).reshape(-1)                        # (N,)
                preds_list.append(y_hat)
                used_times.append(t)

            if not preds_list:
                raise RuntimeError("Baseline produced no predictions (insufficient matching times across revs).")

            preds = np.vstack(preds_list).astype(np.float32)  # (S_used, N) in CANONICAL NODE order
            times = np.array(used_times, dtype=float)

            # ---- Baseline debug: how many times used and rev matches per time ----
            print(f"[BASELINE-DBG] used_times={len(times)} of requested ~{len(times_target)}")
            atol_dbg = float(getattr(cfg, "times_match_atol", 1e-2))
            counts = [int(np.isclose(T_all, float(t), atol=atol_dbg).sum()) for t in times[:10]]
            print(f"[BASELINE-DBG] matches per time (first 10): {counts}")
            # Optional: clip to train range
            y_lo = float(np.min(Y_train)); y_hi = float(np.max(Y_train))
            print(f"[BASELINE-CLIP] global [{y_lo:.6e}, {y_hi:.6e}]")
            np.clip(preds, y_lo, y_hi, out=preds)

            # NODE (canonical) -> FILE order
            print("[BASELINE] Converting predictions and nodes to FILE/flatten order for write-out")
            preds_file = to_file_order(preds, colmap_canon)
            inv = np.empty_like(colmap_canon); inv[colmap_canon] = np.arange(colmap_canon.size)
            nodes_df_file = nodes_df.iloc[inv].reset_index(drop=True)

            from cpfd_rom.ml_rom.rom_eulerian_ml.evaluation import _write_rom_only
            out_dir = Path(cfg.output_dir) / "ML" / f"ROM_param_{float(cfg.user_parameter):.3f}_BASELINE"
            _write_rom_only(preds_file, times, nodes_df_file, getattr(cfg, "field_variable", None), out_dir)
            print(f"[BASELINE] wrote ROM-only files to {out_dir}")

            return None, None

    # Decide train vs load
    use_pretrained = getattr(cfg, "skip_training", False) and os.path.exists(model_path_pt)
    print(f"[PIPELINE] use_pretrained={use_pretrained}  model_path_pt={model_path_pt}")

    # Train or load
    if use_pretrained:
        with log_time("Loading pretrained GNN model"):
            model = build_gcn_model(
                in_dim=in_dim_expected,
                hidden=getattr(cfg, "hidden", 64),
                out_dim=1,
                dropout=getattr(cfg, "dropout", 0.1),
                conv_type=conv_type,
                # early stopping controls
                early_stopping=early_stop, es_patience=es_patience, es_min_delta=es_min_delta, es_restore_best=es_restore_best,
                **extra,
            )
            state = torch.load(model_path_pt, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])  # support rich checkpoints
            else:
                model.load_state_dict(state)
            print("[LOAD] checkpoint loaded")
    else:
        with log_time("Training GNN (graph-first)"):
            shuffle_seed = getattr(cfg, "shuffle_seed", None)
            model, hist = train_gcn_model(
                Y_train_n,
                Y_val_n,
                P_train_n,
                P_val_n,
                edge_index,
                X_node=X_node,
                epochs=getattr(cfg, "epochs", 100),
                lr=getattr(cfg, "lr", 1e-3),
                hidden=getattr(cfg, "hidden", 256),
                dropout=getattr(cfg, "dropout", 0.0),
                lambda_smooth=getattr(cfg, "lambda_smooth", 0.0),
                ignore_head_frac=getattr(cfg, "ignore_head_frac", 0.0),
                shuffle=True,
                shuffle_seed=shuffle_seed,
                device=None,
                add_time=add_time,
                times_train=times_train,
                times_test=times_val,
                time_mode=time_mode,
                fourier_m=fourier_m,
                time_gain=time_gain,
                conv_type=conv_type,
                **extra,
            )
            # Basic training diagnostics
            try:
                min_val = float(np.nanmin(np.array(hist.get("val_mse", []))))
                last_val = float(hist.get("val_mse", [np.nan])[-1])
                print(f"[TRAIN] min val_MSE={min_val:.6e}  last val_MSE={last_val:.6e}")
            except Exception:
                pass
            # --- Early stopping summary ---
            epochs_req = int(getattr(cfg, "epochs", 100))
            epochs_ran = len(hist.get("val_mse", []))
            print(f"[ES] epochs_ran={epochs_ran} / requested={epochs_req}")
            if early_stop and epochs_ran < epochs_req:
                print("[ES] Triggered early stopping" + (" (restored best weights)" if es_restore_best else ""))
            else:
                print("[ES] No early stop (ran full epochs)")
            torch.save(model.state_dict(), model_path_pt)
            print(f"[SAVE] wrote model state_dict to {model_path_pt}")

    # Optional: sanity  check time responsiveness on the same param
    if add_time and times_train is not None and len(times_train) > 0:
        try:
            device = next(model.parameters()).device
            ei_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)
            X_t  = X_node.to(device) if isinstance(X_node, torch.Tensor) else torch.tensor(X_node, dtype=torch.float32, device=device)

            # normalize param exactly like training
            pr = ((np.array([user_param], np.float32) - p_mu) / p_std).astype(np.float32)

            # training time stats used during training
            t_min_tr = float(np.min(times_train))
            t_max_tr = float(np.max(times_train))
            t_mu_tr  = float(np.mean(times_train))
            t_sigma_tr = float(np.std(times_train)) if float(np.std(times_train)) > 0 else 1.0

            # IMPORTANT: pass the exact time config (time_mode, fourier_m, and stats)
            y1 = predict_gcn(
                model,
                X_node=X_t,
                params_row=pr,
                edge_index=ei_t,
                time_val=t_min_tr,
                add_time=add_time,
                time_mode=time_mode,
                t_min=t_min_tr,
                t_max=t_max_tr,
                t_mu=t_mu_tr,
                t_sigma=t_sigma_tr,
                fourier_m=fourier_m,
                time_gain=time_gain,
            )
            y2 = predict_gcn(
                model,
                X_node=X_t,
                params_row=pr,
                edge_index=ei_t,
                time_val=t_max_tr,
                add_time=add_time,
                time_mode=time_mode,
                t_min=t_min_tr,
                t_max=t_max_tr,
                t_mu=t_mu_tr,
                t_sigma=t_sigma_tr,
                fourier_m=fourier_m,
                time_gain=time_gain,
            )
            # de-normalize for readable diagnostics
            y1d = y1 * y_std + y_mu
            y2d = y2 * y_std + y_mu
            mn1, mx1 = float(y1d.min()), float(y1d.max())
            mn2, mx2 = float(y2d.min()), float(y2d.max())
            print(f"[SANITY] tmin: min={mn1:.6e} max={mx1:.6e} | tmax: min={mn2:.6e} max={mx2:.6e}")
            print(f"[SANITY] mean|?| = {(y1d - y2d).abs().mean().item():.3e}")
        except Exception as e:
            print(f"[SANITY] time responsiveness check failed: {e}")

    # Inference for the requested user_parameter, using the canonical graph & reference times
    with log_time("Inference at user_parameter (ROM-only outputs)"):
        # Use the same reference rev graph/times
        out_root = Path(cfg.output_dir) / "graph"
        ref_dir = out_root / ref_rev
        targets = list_targets(ref_dir)
        times = np.array([t for t, _ in targets], dtype=float)
        print(f"[INFER] predicting {len(times)} snapshots for user_parameter={user_param:.6f}")

        # Compute time statistics to match training (only if used)
        if add_time and times_train is not None and len(times_train) > 0:
            t_mu = float(np.mean(times_train))
            t_sigma = float(np.std(times_train)) if np.std(times_train) > 0 else 1.0
            t_min = float(np.min(times_train))
            t_max = float(np.max(times_train)) if float(np.max(times_train)) > t_min else (t_min + 1.0)
        else:
            t_mu = t_sigma = None
            t_min, t_max = 0.0, 1.0

        # Predict for each time at the requested parameter
        ei_torch = torch.as_tensor(edge_index, dtype=torch.long)
        X_node_t = X_node.clone() if isinstance(X_node, torch.Tensor) else torch.tensor(X_node, dtype=torch.float32)
        params_row = ((np.array([user_param], dtype=np.float32) - p_mu) / p_std).astype(np.float32)

        preds_list = []
        for t in times:
            y = predict_gcn(
                model,
                X_node=X_node_t,
                params_row=params_row,
                edge_index=ei_torch,
                add_time=add_time,
                time_val=(float(t) if add_time else None),
                time_mode=time_mode,
                t_min=t_min,
                t_max=t_max,
                t_mu=t_mu,
                t_sigma=t_sigma,
                fourier_m=fourier_m,
                time_gain=time_gain,
            )
            preds_list.append(y.detach().cpu().numpy())

        preds = np.stack(preds_list, axis=0)  # (S, N) in CANONICAL NODE order

        # de-normalize predictions back to CFD scale
        preds = preds * y_std + y_mu

        # Option A: Global train-range clip based on training targets
        y_lo = float(np.min(Y_train))
        y_hi = float(np.max(Y_train))
        print(f"[CLIP] global [{y_lo:.6e}, {y_hi:.6e}]")
        np.clip(preds, y_lo, y_hi, out=preds)

        # ---------------- Alignment Step 2: NODE (canonical) -> FILE order ----------------
        print("[ALIGN] Converting predictions and nodes to FILE/flatten order for write-out")
        preds_file = to_file_order(preds, colmap_canon)  # (S, N_file)
        inv = np.empty_like(colmap_canon)
        inv[colmap_canon] = np.arange(colmap_canon.size)
        nodes_df_file = nodes_df.iloc[inv].reset_index(drop=True)

        # Write ROM-only files (x y z i j k + field) in FILE order expected by viewer
        from cpfd_rom.ml_rom.rom_eulerian_ml.evaluation import _write_rom_only
        out_dir = Path(cfg.output_dir) / "ML" / f"ROM_param_{float(cfg.user_parameter):.3f}"
        _write_rom_only(preds_file, times, nodes_df_file, getattr(cfg, "field_variable", None), out_dir)

    return None, None
