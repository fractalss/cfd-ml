# GCN-based Eulerian ROM pipeline (PyTorch + PyG)
# Refactored to use canonical nodes/edges from graph_build.ensure_graph_artifacts
# so node order, connectivity, and feature alignment are consistent.

from __future__ import annotations
import os
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import torch

from cpfd_rom.util import config
from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.ml_rom.rom_eulerian_ml.loader import load_and_preprocess_eulerian_data_npy
from cpfd_rom.ml_rom.rom_eulerian_ml.evaluation import compute_rmse
from cpfd_rom.ml_rom.rom_eulerian_ml import graph_build
from cpfd_rom.ml_rom.rom_eulerian_ml.model_gnn import train_gcn_model, build_gcn_model


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def _torch_model_path(base_path: str) -> str:
    """Force .keras ? .pt for checkpoints."""
    if base_path.endswith(".keras"):
        return base_path[:-6] + ".pt"
    if not base_path.endswith(".pt"):
        return base_path + ".pt"
    return base_path

def _normalize_SNx(X: np.ndarray) -> np.ndarray:
    """Coerce to (S,N) from (S,N,1) or (S,1,N)."""
    X = np.asarray(X)
    if X.ndim == 3 and X.shape[-1] == 1:
        return X[..., 0]
    if X.ndim == 3 and X.shape[1] == 1:
        return X[:, 0, :]
    if X.ndim == 2:
        return X
    raise ValueError(f"Bad shape: {X.shape}")

def _get_device(name: str | None = None) -> torch.device:
    if name in {"cpu", "cuda"}:
        return torch.device(name if (name != "cuda" or torch.cuda.is_available()) else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _predict_with_gcn(model, X_scaled, param_value, edge_index, device=None) -> np.ndarray:
    """Snapshot-wise inference to bound memory usage."""
    dev = _get_device(device)
    if isinstance(edge_index, np.ndarray):
        ei = torch.from_numpy(edge_index).long().to(dev)
    else:
        ei = edge_index.to(dev)

    X = _normalize_SNx(X_scaled)
    S, N = X.shape
    out_list = []
    P_pred = np.full((S, 1), float(param_value), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(S):
            x_node = torch.from_numpy(X[s].astype(np.float32)).to(dev).reshape(-1, 1)
            p_node = torch.from_numpy(P_pred[s]).to(dev).reshape(1, -1).repeat(N, 1)
            feats = torch.cat([x_node, p_node], dim=1)
            y = model(feats, ei).squeeze(-1).detach().cpu().numpy()
            out_list.append(y)
    return np.stack(out_list, axis=0)

def align_and_evaluate_ml_rom(X_pred, pivoted_columns, times, config, scaler):
    """Align ROM vs CFD and compute RMSE."""
    test_data = config.test_df.dropna(subset=[config.field_variable])
    pivoted_test = test_data.pivot_table(index="time", columns=["x", "y", "z"], values=config.field_variable)

    X_pred = np.squeeze(X_pred)
    pred_df_all = pd.DataFrame(X_pred, index=times, columns=list(pivoted_columns)[: X_pred.shape[1]])

    common_cols = [col for col in pivoted_columns if col in pivoted_test.columns]
    common_times = sorted(set(pivoted_test.index) & set(pred_df_all.index))
    if not common_times:
        raise ValueError("No common times between CFD and ROM predictions!")

    pred_df_all = pred_df_all.loc[common_times, common_cols]
    pivoted_test = pivoted_test.loc[common_times, common_cols]

    cfd_values, rom_values = pivoted_test.values, pred_df_all.values
    assert cfd_values.shape == rom_values.shape, f"Mismatch: CFD {cfd_values.shape}, ROM {rom_values.shape}"

    rmse_df, merged_snapshots = compute_rmse(
        cfd_values, rom_values, common_times,
        config.target_times,
        pd.MultiIndex.from_tuples(common_cols, names=["x", "y", "z"]),
        scaler,
    )
    return rmse_df, merged_snapshots


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------

def run_ml_rom_pipeline(config, log_time):
    # I/O setup
    setup_output_dir(config)
    setup_model_paths(config)
    model_path_pt = _torch_model_path(config.model_path_eulerian)

    # Load Eulerian data
    with log_time("Loading and preprocessing Eulerian data for ML"):
        X_train, X_test, P_train, P_test, scaler, pivoted_columns = load_and_preprocess_eulerian_data_npy()

    X_train, X_test = _normalize_SNx(X_train), _normalize_SNx(X_test)
    P_train = P_train[:, None] if P_train.ndim == 1 else P_train
    P_test = P_test[:, None] if P_test.ndim == 1 else P_test

    # Ensure canonical graph artifacts exist
    with log_time("Ensuring graph artifacts (nodes, edges, targets)"):
        graph_dir = graph_build.ensure_graph_artifacts(config.__dict__, config.field_variable, rebuild=False)
        nodes_df = pd.read_parquet(graph_dir / "nodes.parquet")
        # Build edges from integer (i,j,k)
        edge_index = graph_build.build_edge_index(
            nodes_df[["i", "j", "k"]],
            neighbor_set="n6",
            bidirectional=True
        )

        # Connectivity summary
        graph_build.summarize_connectivity(nodes_df, edge_index, neighbor_set="n6")

        # --- Add this undirected dedup check here ---
        u, v = edge_index[0], edge_index[1]
        uv = np.stack([np.minimum(u, v), np.maximum(u, v)], axis=1)  # canonical undirected pairs
        uniq = np.unique(uv, axis=0)
        E_undir = uniq.shape[0]


        print(f"[GRAPH] Undirected unique edges: {E_undir} (expected - 6*N minus boundaries)")
        # ----------

        # Align features to node order
        graph_build.assert_feature_alignment(X_train.shape[1], edge_index)

    # Train or load model
    if getattr(config, "skip_training", False) and os.path.exists(model_path_pt):
        with log_time("Loading pretrained GCN model"):
            in_dim = 1 + (P_train.shape[1] if P_train is not None else 1)
            model = build_gcn_model(in_dim=in_dim, hidden=getattr(config, "hidden", 64), out_dim=1, dropout=getattr(config, "dropout", 0.1))
            state = torch.load(model_path_pt, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
            model.load_state_dict(state)
            try:
                scaler = joblib.load(model_path_pt.replace(".pt", "_scaler.pkl"))
            except Exception:
                pass
    else:
        with log_time("Training GCN-based Eulerian ROM"):
            model, _ = train_gcn_model(
                X_train, X_test, P_train, P_test, edge_index,
                epochs=getattr(config, "epochs", 2),
                lr=getattr(config, "lr", 1e-3),
                hidden=getattr(config, "hidden", 64),
                dropout=getattr(config, "dropout", 0.1),
                lambda_smooth=getattr(config, "lambda_smooth", 0.1),
                add_time=False,
            )
            torch.save(model.state_dict(), model_path_pt)
            joblib.dump(scaler, model_path_pt.replace(".pt", "_scaler.pkl"))

    # Zero-shot inference on full sequence
    # --- In pipeline.py, replace the "Zero-shot inference (full sequence)" block with: ---
    with log_time("Zero-shot inference (test_dir only)"):
        # Use ONLY the test split for inference/eval so shapes match CFD
        X_scaled = _normalize_SNx(X_test)
        times = np.asarray(getattr(config, "test_times", np.arange(X_scaled.shape[0], dtype=float)))

        # Build a light test_df for eval/plots aligned to test_dir only
        X_raw = scaler.inverse_transform(X_scaled)
        test_records = (
            {
                "source": config.test_dir,
                "time": float(t),
                "x": x,
                "y": y,
                "z": z,
                config.field_variable: val,
            }
            for t, snapshot in zip(times, X_raw)
            for (x, y, z), val in zip(pivoted_columns, snapshot)
        )
        config.test_df = pd.DataFrame.from_records(test_records)

        # Predict with the trained GCN on test only
        X_pred = _predict_with_gcn(model, X_scaled, param_value=config.user_parameter, edge_index=edge_index)

    # Evaluation
    with log_time("Evaluating ML-ROM output"):
        rmse_df, merged_snapshots = align_and_evaluate_ml_rom(X_pred, pivoted_columns, times, config, scaler)

    return rmse_df, merged_snapshots
