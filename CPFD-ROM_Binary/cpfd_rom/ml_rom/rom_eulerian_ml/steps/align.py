# ===============================
# File: cpfd_rom/ml_rom/rom_eulerian_ml/pipeline_refactored/steps/align.py
# Purpose: Alignment helpers split from pipeline, including FILE?NODE mapping and canonicalization
# ===============================
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from cpfd_rom.util.utils_alignment import (
    canonicalize_by_ijk,
    build_colmap_by_join,
)


def maybe_load_coords_file_df(cfg, ref_graph_dir: Optional[Path], N_cols: int) -> Optional[pd.DataFrame]:
    """Search for a coords_file_df that describes FILE/flatten order (len == N_cols)."""
    p_parq = getattr(cfg, 'coords_file_parquet', None)
    p_csv  = getattr(cfg, 'coords_file_csv', None)

    try_paths = []
    if p_parq:
        try_paths.append(Path(p_parq))
    if p_csv:
        try_paths.append(Path(p_csv))
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
            if set(['i','j','k']).issubset(df.columns) and len(df) == N_cols:
                print(f"[ALIGN] Using coords_file_df from {p}")
                return df[['i','j','k']].copy()
        except Exception as e:
            print(f"[ALIGN] Failed to load coords_file_df from {p}: {e}")
    print("[ALIGN] WARNING: coords_file_df not provided/found; join-based reorder will be skipped.")
    return None


def build_file_to_node_colmap(coords_file_df: Optional[pd.DataFrame], nodes_df: pd.DataFrame, Y_train: np.ndarray) -> np.ndarray:
    """Build FILE?NODE column map. Identity if unavailable."""
    if coords_file_df is not None:
        colmap = build_colmap_by_join(coords_file_df, nodes_df)
        print(f"[ALIGN] Built FILE?NODE colmap via (i,j,k) join (len={len(colmap)})")
        return colmap
    else:
        print("[ALIGN] No FILE?NODE mapping found; assuming identity (Y_* already in NODE order).")
        return np.arange(Y_train.shape[1], dtype=np.int64)


def canonicalize_all(
    nodes_df: pd.DataFrame,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    edge_index: np.ndarray,
    build_node_features_xyz_fn,
    colmap_file_to_node: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply canonical (i,j,k) order to nodes/targets/edges and rebuild X_node; return FILE?CANON map."""
    nodes_df, Y_train, edge_index, order, inv = canonicalize_by_ijk(nodes_df, Y_train, edge_index)
    if Y_val.size:
        Y_val = Y_val[:, order]
    colmap_canon = order[colmap_file_to_node]
    X_node = build_node_features_xyz_fn(nodes_df)
    print(f"[ALIGN] Applied canonical (i,j,k) order. Perm size={len(order)}. Rebuilt X_node: {tuple(X_node.shape)}")
    return nodes_df, Y_train, Y_val, edge_index, X_node, colmap_canon


def to_file_order(arr_or_df, colmap_canon: np.ndarray, dataframe: bool = False):
    """Reorder array (S,N) or nodes_df (N,cols) from canonical NODE order ? FILE/flatten order."""
    inv = np.empty_like(colmap_canon)
    inv[colmap_canon] = np.arange(colmap_canon.size)
    if dataframe:
        df = arr_or_df.iloc[inv].reset_index(drop=True)
        return df
    a = arr_or_df
    if a.ndim == 1:
        return a[inv]
    return a[:, inv]
