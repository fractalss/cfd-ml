# GCN-based Eulerian ROM pipeline (PyTorch + PyG)
# Graph-first flow:
#  - Build ONE canonical graph from a reference training directory (first of rev_dirs)
#  - Assemble training targets from ALL rev_dirs with their parameter values
#  - Train GCN on [XYZ(+time)+param] -> field
#  - Inference for user_parameter using the same canonical graph & the reference times

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir, format_metadata
from cpfd_rom.ml_rom.rom_eulerian_ml import graph_build
from cpfd_rom.ml_rom.rom_eulerian_ml.model_gnn import (
    train_gcn_model,
    build_gcn_model,
    build_node_features_xyz,
)
from cpfd_rom.ml_rom.rom_eulerian_ml.inference import infer_on_param

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _torch_model_path(base_path: str) -> str:
    if base_path.endswith(".keras"):
        return base_path[:-6] + ".pt"
    if not base_path.endswith(".pt"):
        return base_path + ".pt"
    return base_path


def _list_targets(graph_dir: Path) -> List[Tuple[float, Path]]:
    snap_dir = graph_dir / 'snapshots'
    files = sorted(list(snap_dir.glob('target_*.parquet')) + list(snap_dir.glob('target_*.csv')))
    if not files:
        raise FileNotFoundError(f"No target_* files found in {snap_dir}")
    import re
    rx = re.compile(r'target_(\d+\.?\d*)s\.(?:parquet|csv)')
    times = [(float(m.group(1)), p) for p in files if (m := rx.match(p.name))]
    return sorted(times, key=lambda t: t[0])


def _load_target_vec(path: Path) -> np.ndarray:
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    arr = arr if arr.ndim > 1 else arr.reshape(-1, 1)
    return arr.reshape(-1)


def _assemble_train_val_from_revs(cfg: Dict, neighbor_set: str = 'n6', val_frac: float = 0.2):
    """Build canonical graph from the FIRST rev in cfg['rev_dirs'].
    Generate snapshot targets for ALL revs, then assemble Y_train/Y_val and P arrays.
    Returns: (nodes_df, edge_index, X_node, Y_train, Y_val, P_train, P_val, times_train, times_val)
    """
    base = Path(cfg['base_data_dir'])
    out_root = Path(cfg['output_dir']) / 'graph'

    # 1) Pick reference rev for canonical graph (node order, ijk mapping)
    revs: List[str] = list(cfg['rev_dirs'])
    if not revs:
        raise ValueError("cfg['rev_dirs'] must list training directories")
    ref_rev = revs[0]

    # Ensure artifacts for the reference rev (nodes/edges + targets)
    ref_graph_dir = graph_build.ensure_graph_artifacts(
        {**cfg, 'test_dir': ref_rev}, cfg['field_variable'], rebuild=False, neighbor_set=neighbor_set
    )

    # 2) Ensure targets for every other rev (nodes/edges are taken only from ref_rev)
    for r in revs[1:]:
        graph_build.ensure_graph_artifacts({**cfg, 'test_dir': r}, cfg['field_variable'], rebuild=False, neighbor_set=neighbor_set)

    # Load canonical nodes & edges from ref_rev
    nodes_df = pd.read_parquet(ref_graph_dir / 'nodes.parquet').sort_values('node_id').reset_index(drop=True)
    edge_index = graph_build.build_edge_index(nodes_df[['i','j','k']], neighbor_set=neighbor_set, bidirectional=True)
    graph_build.summarize_connectivity(nodes_df, edge_index, neighbor_set=neighbor_set)

    # Build static node features (standardized xyz) to match inference.py
    X_node = build_node_features_xyz(nodes_df)

    # 3) Assemble datasets
    Y_train_list: List[np.ndarray] = []
    Y_val_list:   List[np.ndarray] = []
    P_train_list: List[np.ndarray] = []
    P_val_list:   List[np.ndarray] = []
    T_train_list: List[float] = []
    T_val_list:   List[float] = []

    for r in revs:
        graph_dir = out_root / r
        targets = _list_targets(graph_dir)
        times = np.array([t for t, _ in targets], dtype=float)
        n = len(targets)
        n_train = max(1, int((1.0 - val_frac) * n))

        # Parameter value for this rev
        pval = float(cfg['param_mapping'][r])
        P_row = np.array([pval], dtype=np.float32)

        # Load to arrays in node_id order
        Ys = np.stack([_load_target_vec(p) for _, p in targets], axis=0)  # (S, N)

        Y_train_list.append(Ys[:n_train])
        Y_val_list.append(Ys[n_train:])
        P_train_list.append(np.repeat(P_row[None, :], n_train, axis=0))
        P_val_list.append(np.repeat(P_row[None, :], n - n_train, axis=0))
        T_train_list.extend(times[:n_train].tolist())
        T_val_list.extend(times[n_train:].tolist())

    Y_train = np.concatenate(Y_train_list, axis=0) if Y_train_list else np.empty((0, len(nodes_df)), dtype=np.float32)
    Y_val   = np.concatenate(Y_val_list,   axis=0) if Y_val_list   else np.empty((0, len(nodes_df)), dtype=np.float32)
    P_train = np.concatenate(P_train_list, axis=0) if P_train_list else np.empty((0, 1), dtype=np.float32)
    P_val   = np.concatenate(P_val_list,   axis=0) if P_val_list   else np.empty((0, 1), dtype=np.float32)
    times_train = np.array(T_train_list, dtype=float) if T_train_list else None
    times_val   = np.array(T_val_list,   dtype=float) if T_val_list   else None

    return ref_rev, nodes_df, edge_index, X_node, Y_train, Y_val, P_train, P_val, times_train, times_val


def _write_rom_only(preds: np.ndarray, times: np.ndarray, nodes_df: pd.DataFrame, field_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    header_cols = ['x','y','z','i','j','k', field_name]
    header_md = [format_metadata(i+1, c) for i, c in enumerate(header_cols)]
    key = nodes_df[['node_id','x','y','z','i','j','k']].sort_values('node_id').reset_index(drop=True)

    for s, t in enumerate(times):
        y = preds[s].astype(np.float64)
        df = key.copy(); df[field_name] = y
        df = df.sort_values(by=['k','j','i'], kind='mergesort')
        out_path = out_dir / f"cells_{float(t):09.3f}s.txt"
        with open(out_path, 'w') as f:
            f.write('# Zone name = "Cells"\n')
            f.write(f'# Solution time = {float(t):.6f} s\n')
            for line in header_md:
                f.write(line)
            df[['x','y','z','i','j','k', field_name]].to_csv(
                f, sep='\t', header=False, index=False, float_format='%.6e')

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
        (ref_rev, nodes_df, edge_index, X_node,
         Y_train, Y_val, P_train, P_val, times_train, times_val) = _assemble_train_val_from_revs(cfg.__dict__, neighbor_set='n6')

    # Train or load
    if getattr(cfg, 'skip_training', False) and os.path.exists(model_path_pt):
        with log_time("Loading pretrained GCN model"):
            in_dim = X_node.shape[1] + P_train.shape[1]  # no time features by default
            model = build_gcn_model(in_dim=in_dim, hidden=getattr(cfg, 'hidden', 64), out_dim=1, dropout=getattr(cfg, 'dropout', 0.1))
            state = torch.load(model_path_pt, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(state)
    else:
        with log_time("Training GCN (graph-first)"):
            model, hist = train_gcn_model(
                Y_train, Y_val, P_train, P_val, edge_index,
                X_node=X_node,
                epochs=getattr(cfg, 'epochs', 50),
                lr=getattr(cfg, 'lr', 1e-3),
                hidden=getattr(cfg, 'hidden', 64),
                dropout=getattr(cfg, 'dropout', 0.1),
                lambda_smooth=getattr(cfg, 'lambda_smooth', 0.0),
                add_time=False,
                times_train=times_train,
                times_test=times_val,
            )
            torch.save(model.state_dict(), model_path_pt)

    # Inference for the requested user_parameter, using the canonical graph & reference times
    with log_time("Inference at user_parameter (ROM-only outputs)"):
        # Build an inference cfg that points test_dir to the canonical ref_rev graph/times
        icfg = dict(cfg.__dict__)
        icfg['test_dir'] = ref_rev  # reuse reference times & graph
        preds, times, nodes_df_inf = infer_on_param(model, icfg, param_value=cfg.user_parameter, device='auto')

        # Write ROM-only files (x y z i j k + field)
        out_dir = Path(cfg.output_dir) / 'ML' / f'ROM_param_{float(cfg.user_parameter):.3f}'
        _write_rom_only(preds, times, nodes_df_inf, cfg.field_variable, out_dir)

    return None, None
