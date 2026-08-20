# utils_alignment.py   alignment helpers for Eulerian GNN pipelines
# Keep X_node, edge_index, and Y_* on the SAME node order.
#
# Usage patterns
# 1) Canonicalize to lexicographic (i,j,k):
#    nodes_df, Y_train, edge_index, perm, inv = canonicalize_by_ijk(nodes_df, Y_train, edge_index)
#    _,        Y_val,   _ ,       _,   _   = canonicalize_by_ijk(nodes_df, Y_val,   None)
#    X_node = build_node_features_xyz(nodes_df)
#
# 2) Alternatively, build Y in nodes_df order using a JOIN on (i,j,k):
#    Y_train, colmap = reorder_Y_by_join(Y_train_flat, coords_flat, nodes_df)
#    # When writing back to the original file order later:
#    Y_back = to_file_order(Y_pred_nodes_order, colmap)

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# ---------------------------------------------------------------
# Core: canonicalize everything to a single, reproducible order.
# ---------------------------------------------------------------

def canonicalize_by_ijk(
    nodes_df: pd.DataFrame,
    Y: np.ndarray,
    edge_index: Optional[np.ndarray]
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Reorder nodes, targets, and edges to lexicographic (i, j, k).

    Parameters
    ----------
    nodes_df : DataFrame with columns ['i','j','k', ...]
    Y        : array of shape (S, N) or (N,) in *current* node order
    edge_index : (2, E) ndarray of int (optional)

    Returns
    -------
    nodes2 : nodes_df sorted by (i,j,k)
    Y2     : Y with columns permuted to match nodes2
    ei2    : edge_index reindexed to nodes2 order (or None)
    order  : permutation mapping old->new (length N)
    inv    : inverse permutation mapping new->old
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    assert {'i','j','k'}.issubset(nodes_df.columns), "nodes_df must have i,j,k"
    N = nodes_df.shape[0]
    assert Y.shape[1] == N, f"Y has N={Y.shape[1]}, nodes_df has N={N}"

    order = np.lexsort((nodes_df['k'].to_numpy(),
                        nodes_df['j'].to_numpy(),
                        nodes_df['i'].to_numpy()))
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)

    nodes2 = nodes_df.iloc[order].reset_index(drop=True)
    Y2 = Y[:, order]

    ei2 = None
    if edge_index is not None:
        edge_index = np.asarray(edge_index, dtype=np.int64)
        ei2 = inv[edge_index]

    return nodes2, Y2, ei2, order, inv

# -------------------------------------------------------------------
# JOIN-based reorder: trust (i,j,k) mapping, ignore any implicit order.
# -------------------------------------------------------------------

def build_colmap_by_join(
    file_coords_df: pd.DataFrame,  # columns: i,j,k for the *file/flatten* order
    nodes_df: pd.DataFrame         # columns: i,j,k for the *desired* (nodes) order
) -> np.ndarray:
    """Return colmap such that Y_nodes = Y_file[:, colmap].

    Guarantees one-to-one by validating size and duplicates.
    """
    req = {'i','j','k'}
    assert req.issubset(file_coords_df.columns) and req.issubset(nodes_df.columns), "coords must have i,j,k"

    file_idx = file_coords_df.reset_index(drop=True).reset_index(names='file_pos')
    node_idx = nodes_df.reset_index(drop=True).reset_index(names='node_pos')

    m = file_idx.merge(node_idx, on=['i','j','k'], how='inner', validate='one_to_one')
    if len(m) != len(nodes_df):
        raise ValueError(f"JOIN size mismatch: matched {len(m)} of {len(nodes_df)} nodes. Check duplicates/missing.")

    m = m.sort_values('file_pos')
    colmap = m['node_pos'].to_numpy(dtype=np.int64)
    return colmap


def reorder_Y_by_join(
    Y_file: np.ndarray,           # (S, N_file) in file/flatten order
    file_coords_df: pd.DataFrame, # i,j,k in file order
    nodes_df: pd.DataFrame        # i,j,k in desired order
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder Y from file order to nodes_df order via (i,j,k) join.

    Returns
    -------
    Y_nodes : (S, N_nodes) in nodes_df order
    colmap  : mapping such that Y_nodes = Y_file[:, colmap]
    """
    if Y_file.ndim == 1:
        Y_file = Y_file.reshape(1, -1)
    colmap = build_colmap_by_join(file_coords_df, nodes_df)
    Y_nodes = Y_file[:, colmap]
    return Y_nodes, colmap

# ------------------------------------------------------
# Utilities for going back to the original file ordering
# ------------------------------------------------------

def to_file_order(Y_nodes: np.ndarray, colmap: np.ndarray) -> np.ndarray:
    """Inverse of reorder_Y_by_join: return columns to file order.

    If Y_nodes = Y_file[:, colmap], then Y_file = Y_nodes[:, inv_colmap].
    """
    if Y_nodes.ndim == 1:
        Y_nodes = Y_nodes.reshape(1, -1)
    inv = np.empty_like(colmap)
    inv[colmap] = np.arange(colmap.size)
    return Y_nodes[:, inv]

# ------------------------------------------------------
# Light-weight diagnostics to catch misalignment early
# ------------------------------------------------------

def make_identity_targets(nodes_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (I,J,K) identity fields of shape (1, N) each, in nodes_df order."""
    I = nodes_df['i'].to_numpy()[None, :].astype(np.float32)
    J = nodes_df['j'].to_numpy()[None, :].astype(np.float32)
    K = nodes_df['k'].to_numpy()[None, :].astype(np.float32)
    return I, J, K


def quick_axis_sanity(nodes_df: pd.DataFrame) -> None:
    """Print quick ranges for i, j, k; helpful before/after reordering."""
    for c in ('i','j','k'):
        v = nodes_df[c].to_numpy()
        print(f"[SANITY] {c}-range: [{v.min()}..{v.max()}], unique={np.unique(v).size}")
