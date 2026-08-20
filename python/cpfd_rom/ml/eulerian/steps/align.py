"""Alignment helpers for Eulerian ROM data.

This module handles FILE-to-NODE mapping and canonical ``(i, j, k)``
ordering for Eulerian targets, graph nodes, and edges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from cpfd_rom.util.logging_config import detail
from cpfd_rom.util.utils_alignment import (
    build_colmap_by_join,
    canonicalize_by_ijk,
)


logger = logging.getLogger(__name__)


def maybe_load_coords_file_df(
    cfg,
    ref_graph_dir: Optional[Path],
    N_cols: int,
) -> Optional[pd.DataFrame]:
    """Load coordinates describing FILE/flatten order, when available.

    A usable coordinate table must contain ``i``, ``j``, and ``k`` columns
    and have exactly ``N_cols`` rows.
    """
    p_parq = getattr(cfg, "coords_file_parquet", None)
    p_csv = getattr(cfg, "coords_file_csv", None)

    try_paths = []
    if p_parq:
        try_paths.append(Path(p_parq))
    if p_csv:
        try_paths.append(Path(p_csv))
    if ref_graph_dir is not None:
        ref_graph_dir = Path(ref_graph_dir)
        try_paths.extend(
            [
                ref_graph_dir / "coords_file_order.parquet",
                ref_graph_dir / "coords_file_order.csv",
            ]
        )

    required_columns = {"i", "j", "k"}

    for path in try_paths:
        try:
            suffix = path.suffix.lower()
            if suffix == ".parquet" and path.exists():
                coords_df = pd.read_parquet(path)
            elif suffix == ".csv" and path.exists():
                coords_df = pd.read_csv(path)
            else:
                logger.debug("Coordinate-order file not found or unsupported: %s", path)
                continue

            missing_columns = required_columns.difference(coords_df.columns)
            if missing_columns:
                logger.warning(
                    "Ignoring coordinate-order file %s; missing columns: %s",
                    path,
                    ", ".join(sorted(missing_columns)),
                )
                continue

            if len(coords_df) != N_cols:
                logger.warning(
                    "Ignoring coordinate-order file %s; expected %d rows, found %d",
                    path,
                    N_cols,
                    len(coords_df),
                )
                continue

            detail(logger, "Using FILE-order coordinates from %s", path)
            return coords_df[["i", "j", "k"]].copy()

        except Exception as error:
            logger.warning(
                "Failed to load FILE-order coordinates from %s: %s",
                path,
                error,
            )

    logger.warning(
        "FILE-order coordinates were not found; join-based target reordering "
        "will be skipped"
    )
    return None


def build_file_to_node_colmap(
    coords_file_df: Optional[pd.DataFrame],
    nodes_df: pd.DataFrame,
    Y_train: np.ndarray,
) -> np.ndarray:
    """Build the FILE-to-NODE column map, or use identity if unavailable."""
    if coords_file_df is not None:
        colmap = build_colmap_by_join(coords_file_df, nodes_df)
        detail(
            logger,
            "Built FILE-to-NODE column map using an (i, j, k) join (size=%d)",
            len(colmap),
        )
        return colmap

    logger.warning(
        "No FILE-to-NODE mapping is available; assuming targets are already "
        "in NODE order"
    )
    return np.arange(Y_train.shape[1], dtype=np.int64)


def canonicalize_all(
    nodes_df: pd.DataFrame,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    edge_index: np.ndarray,
    build_node_features_xyz_fn: Callable[[pd.DataFrame], np.ndarray],
    colmap_file_to_node: np.ndarray,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Canonicalize nodes, targets, and edges and rebuild node features.

    Returns the canonicalized data together with the FILE-to-CANON column map.
    """
    nodes_df, Y_train, edge_index, order, _inv = canonicalize_by_ijk(
        nodes_df,
        Y_train,
        edge_index,
    )

    if Y_val.size:
        Y_val = Y_val[:, order]

    colmap_canon = order[colmap_file_to_node]
    X_node = build_node_features_xyz_fn(nodes_df)

    detail(
        logger,
        "Applied canonical (i, j, k) ordering (permutation size=%d); "
        "rebuilt node features with shape=%s",
        len(order),
        tuple(X_node.shape),
    )
    logger.debug(
        "Canonical permutation dtype=%s; FILE-to-CANON map dtype=%s",
        order.dtype,
        colmap_canon.dtype,
    )

    return (
        nodes_df,
        Y_train,
        Y_val,
        edge_index,
        X_node,
        colmap_canon,
    )


def to_file_order(
    arr_or_df,
    colmap_canon: np.ndarray,
    dataframe: bool = False,
):
    """Reorder canonical NODE data into FILE/flatten order."""
    inv = np.empty_like(colmap_canon)
    inv[colmap_canon] = np.arange(colmap_canon.size)

    if dataframe:
        return arr_or_df.iloc[inv].reset_index(drop=True)

    array = arr_or_df
    if array.ndim == 1:
        return array[inv]
    return array[:, inv]
