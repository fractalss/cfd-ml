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
)
from cpfd_rom.ml_rom.rom_eulerian_ml.inference import predict_gcn


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _torch_model_path(base_path: str) -> str:
    if base_path.endswith(".keras"):
        return base_path[:-6] + ".pt"
    if not base_path.endswith(".pt"):
        return base_path + ".pt"
    return base_path


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

    # Configure time conditioning
    time_mode = getattr(cfg, "time_mode", "none")
    add_time = (time_mode is not None) and (str(time_mode).lower() != "none")
    fourier_m = int(getattr(cfg, "fourier_m", 4))
    t_feat_dim = _time_feature_dim(time_mode, fourier_m) if add_time else 0

    # ----- GNN backbone selection (defaults: GraphSAGE) -----
    conv_type = getattr(cfg, "conv_type", "sage")  # "sage" | "gcn" | "gat" | "gin"
    gat_heads = getattr(cfg, "gat_heads", 4)
    attn_drop = getattr(cfg, "attn_dropout", 0.1)

    extra = {}
    if str(conv_type).lower() == "gat":
        extra.update({"heads": int(gat_heads), "attn_dropout": float(attn_drop)})

    print(f"[CFG] conv_type={conv_type}  add_time={add_time}  t_feat_dim={t_feat_dim}")
    if extra:
        print(f"[CFG] GAT extras: {extra}")

    # Expected model input dim for debug
    in_dim_expected = X_node.shape[1] + P_train_n.shape[1] + t_feat_dim
    print(f"[MODEL] expected in_dim={in_dim_expected}")

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
            model, hist = train_gcn_model(
                Y_train,
                Y_val,
                P_train_n,
                P_val_n,
                edge_index,
                X_node=X_node,
                epochs=getattr(cfg, "epochs", 200),
                lr=getattr(cfg, "lr", 3e-4),
                hidden=getattr(cfg, "hidden", 128),
                dropout=getattr(cfg, "dropout", 0.0),
                lambda_smooth=getattr(cfg, "lambda_smooth", 0.0),
                device=None,
                add_time=add_time,
                times_train=times_train,
                times_test=times_val,
                time_mode=time_mode,
                fourier_m=fourier_m,
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
            torch.save(model.state_dict(), model_path_pt)
            print(f"[SAVE] wrote model state_dict to {model_path_pt}")

    # Optional: sanity  check time responsiveness on the same param
    if add_time and times_train is not None and len(times_train) > 0:
        try:
            device = next(model.parameters()).device
            ei_t = torch.as_tensor(edge_index, dtype=torch.long, device=device)
            X_t  = X_node.to(device) if isinstance(X_node, torch.Tensor) else torch.tensor(X_node, dtype=torch.float32, device=device)
            pr   = ((np.array([user_param], np.float32) - p_mu) / p_std).astype(np.float32)
            y1 = predict_gcn(model, ei_t, X_t, pr, time_val=float(np.min(times_train)), add_time=True, time_mode=time_mode)
            y2 = predict_gcn(model, ei_t, X_t, pr, time_val=float(np.max(times_train)), add_time=True, time_mode=time_mode)
            print(f"[SANITY] mean|?y(tmin,tmax)|={(y1 - y2).abs().mean().item():.3e}")
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
                ei_torch,
                X_node_t,
                params_row,
                time_val=float(t) if add_time else None,
                add_time=add_time,
                time_mode=time_mode,
                t_min=t_min,
                t_max=t_max,
                t_mu=t_mu,
                t_sigma=t_sigma,
                fourier_m=fourier_m,
            )
            preds_list.append(y.detach().cpu().numpy())

        preds = np.stack(preds_list, axis=0)  # (S, N)

        # Option A: Global train-range clip based on training targets
        y_lo = float(np.min(Y_train))
        y_hi = float(np.max(Y_train))
        print(f"[CLIP] global [{y_lo:.6e}, {y_hi:.6e}]")
        np.clip(preds, y_lo, y_hi, out=preds)

        # Write ROM-only files (x y z i j k + field)
        from cpfd_rom.ml_rom.rom_eulerian_ml.evaluation import _write_rom_only
        out_dir = Path(cfg.output_dir) / "ML" / f"ROM_param_{float(cfg.user_parameter):.3f}"
        _write_rom_only(preds, times, nodes_df, getattr(cfg, "field_variable", None), out_dir)

    return None, None
