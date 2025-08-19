# loader.py  graph-first dataset loader for GCN (Eulerian)
# Strictly for the gcn branch. Builds a canonical graph once and assembles
# snapshot matrices (Y) + parameter arrays (P) across all training rev_dirs.

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import torch

# Local project imports (gcn branch)
from cpfd_rom.ml_rom.rom_eulerian_ml import graph_build
from cpfd_rom.ml_rom.rom_eulerian_ml.model_gnn import build_node_features_xyz

__all__ = [
    "list_targets",
    "load_target_vec",
    "build_canonical_graph",
    "prepare_graph_and_datasets",
]

# -----------------------------------------------------------------------------
# File/system helpers
# -----------------------------------------------------------------------------

def list_targets(graph_dir: Path) -> List[Tuple[float, Path]]:
    """List snapshot files as (time, path), sorted by time.

    Accepts files named like: target_<time>s.parquet or target_<time>s.csv
    Example: target_0050.0s.parquet -> time=50.0
    """
    snap_dir = Path(graph_dir) / "snapshots"
    files = sorted(list(snap_dir.glob("target_*.parquet")) + list(snap_dir.glob("target_*.csv")))
    if not files:
        raise FileNotFoundError(f"No target_* files found in {snap_dir}")

    out: List[Tuple[float, Path]] = []
    for p in files:
        name = p.stem  # e.g., target_0050.0s
        if not (name.startswith("target_") and name.endswith("s")):
            continue
        t_str = name[len("target_"):-1]
        try:
            tt = float(t_str)
        except Exception:
            continue
        out.append((tt, p))
    return sorted(out, key=lambda t: t[0])


def load_target_vec(path: Path) -> np.ndarray:
    """Load a single snapshot vector (N,) from parquet/csv.
    Expects exactly ONE numeric target column in the file. If known index/coord
    columns are present, they are ignored. Raises if ambiguity remains.
    """
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    # Drop known non-target numeric columns if present
    drop_cols = {"node_id", "i", "j", "k", "x", "y", "z"}
    cols = [c for c in df.columns if c not in drop_cols]
    numeric = df[cols].select_dtypes(include=[np.number])

    if numeric.shape[1] == 1:
        return numeric.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)

    # Fallback: if the whole frame has exactly one numeric col, use it
    all_numeric = df.select_dtypes(include=[np.number])
    if all_numeric.shape[1] == 1:
        return all_numeric.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)

    raise ValueError(
        f"Ambiguous target columns in {path.name}: found {numeric.shape[1]} numeric columns after filtering; "
        "expected exactly one. Ensure snapshots contain only the target column."
    )


# -----------------------------------------------------------------------------
# Canonical graph + datasets
# -----------------------------------------------------------------------------

def build_canonical_graph(cfg: Dict, *, neighbor_set: str = "n6") -> Tuple[str, Path, pd.DataFrame, np.ndarray, torch.Tensor]:
    """Create/ensure graph artifacts for the FIRST training rev and return graph objects.

    Returns
    -------
    ref_rev_key : str
        Basename (e.g., "Rev1_npy") of the reference training revision.
    ref_graph_dir : Path
        Directory containing nodes.parquet and snapshots/ for the reference rev.
    nodes_df : pd.DataFrame
        Canonical node table with columns [node_id, x,y,z, i,j,k].
    edge_index : np.ndarray
        2xE undirected edge list with 0-based node indices matching nodes_df.node_id.
    X_node : torch.Tensor
        (N, F_node) static node features (standardized XYZ) as float32 tensor (CPU).
    """
    revs: List[str] = list(cfg["rev_dirs"]) if isinstance(cfg.get("rev_dirs"), (list, tuple)) else []
    if not revs:
        raise ValueError("cfg['rev_dirs'] must list training directories")

    ref_rev = revs[0]
    ref_rev_key = Path(ref_rev).name  # support absolute or relative entries

    # Ensure artifacts for reference rev (nodes, edges, targets)
    ref_graph_dir = graph_build.ensure_graph_artifacts(
        {**cfg, "test_dir": ref_rev_key}, cfg["field_variable"], rebuild=bool(cfg.get("rebuild_graph", False)), neighbor_set=neighbor_set
    )

    # Load nodes and build edges (canonical order by node_id)
    nodes_df = pd.read_parquet(Path(ref_graph_dir) / "nodes.parquet").sort_values("node_id").reset_index(drop=True)
    edge_index = graph_build.build_edge_index(nodes_df[["i", "j", "k"]], neighbor_set=neighbor_set, bidirectional=True)

    # Static node features (standardized xyz); keep on CPU here
    X_node = build_node_features_xyz(nodes_df)
    return ref_rev_key, Path(ref_graph_dir), nodes_df, edge_index, X_node


def prepare_graph_and_datasets(
    cfg: Dict,
    *,
    neighbor_set: str = "n6",
    val_frac: float = 0.2,
) -> Tuple[
    str, Path, pd.DataFrame, np.ndarray, torch.Tensor,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
]:
    """Build canonical graph (from first rev) and assemble datasets from all revs.

    Returns
    -------
    (ref_rev_key, ref_graph_dir, nodes_df, edge_index, X_node,
     Y_train, Y_val, P_train, P_val, times_train, times_val)

    Shapes:
        Y_*: (S, N)  with N = number of nodes
        P_*: (S, P)  (P often = 1; e.g., inlet velocity)
        times_*: (S,) float seconds or None if not available
    """
    out_root = Path(cfg["output_dir"]) / "graph"

    # 1) Build canonical graph
    ref_rev_key, ref_graph_dir, nodes_df, edge_index, X_node = build_canonical_graph(cfg, neighbor_set=neighbor_set)

    # 2) Ensure snapshot artifacts for all revs (only targets; nodes/edges from ref)
    revs: List[str] = list(cfg["rev_dirs"]) if isinstance(cfg.get("rev_dirs"), (list, tuple)) else []
    for r in revs[1:]:
        r_key = Path(r).name
        graph_build.ensure_graph_artifacts({**cfg, "test_dir": r_key}, cfg["field_variable"], rebuild=bool(cfg.get("rebuild_graph", False)), neighbor_set=neighbor_set)

    # 3) Assemble Y/P/times
    Y_train_list: List[np.ndarray] = []
    Y_val_list: List[np.ndarray] = []
    P_train_list: List[np.ndarray] = []
    P_val_list: List[np.ndarray] = []
    T_train_list: List[float] = []
    T_val_list: List[float] = []

    for r in revs:
        r_key = Path(r).name  # support absolute or relative
        graph_dir = out_root / r_key
        targets = list_targets(graph_dir)
        times = np.array([t for t, _ in targets], dtype=float)
        n = len(targets)
        n_train = max(1, int((1.0 - val_frac) * n))

        # Parameter value for this rev
        if "param_mapping" not in cfg or r_key not in cfg["param_mapping"]:
            raise KeyError(f"param_mapping missing for rev '{r}' (key '{r_key}') in cfg")
        pval = float(cfg["param_mapping"][r_key])
        P_row = np.array([pval], dtype=np.float32)

        # Load snapshot vectors (in canonical node order)
        Ys = np.stack([load_target_vec(p) for _, p in targets], axis=0)  # (S, N)

        Y_train_list.append(Ys[:n_train])
        Y_val_list.append(Ys[n_train:])
        P_train_list.append(np.repeat(P_row[None, :], n_train, axis=0))
        P_val_list.append(np.repeat(P_row[None, :], n - n_train, axis=0))
        T_train_list.extend(times[:n_train].tolist())
        T_val_list.extend(times[n_train:].tolist())

    Y_train = np.concatenate(Y_train_list, axis=0) if Y_train_list else np.empty((0, len(nodes_df)), dtype=np.float32)
    Y_val = np.concatenate(Y_val_list, axis=0) if Y_val_list else np.empty((0, len(nodes_df)), dtype=np.float32)
    P_train = np.concatenate(P_train_list, axis=0) if P_train_list else np.empty((0, 1), dtype=np.float32)
    P_val = np.concatenate(P_val_list, axis=0) if P_val_list else np.empty((0, 1), dtype=np.float32)
    times_train = np.array(T_train_list, dtype=float) if T_train_list else None
    times_val = np.array(T_val_list, dtype=float) if T_val_list else None

    return (
        ref_rev_key,
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
    )
