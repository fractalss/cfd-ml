# cpfd_rom/ml_rom/rom_eulerian_ml/graph_build.py

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Tuple, Union

from cpfd_rom.util.file_parsing import (
    get_simulation_time_from_json_fast,
    get_columns_from_json_cached,
)

# --------------------------------------------------------------------------------------
# Source discovery + indexing (RAW.CELL only)
# --------------------------------------------------------------------------------------

def _source_has_raw_cell(src_dir: Path) -> bool:
    return any(src_dir.glob("Raw.cell.*.npy"))

def _pair_npy_with_json(npy_path: Union[str, Path]) -> Path:
    npy_path = Path(npy_path)
    json_path = npy_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"[ERROR] Missing JSON header for {npy_path}: expected {json_path}")
    return json_path

def _list_raw_cell_npy(src_dir: Path) -> list[Path]:
    return sorted(src_dir.glob("Raw.cell.*.npy"))

def _build_cell_time_index(src_dir: Path) -> list[tuple[float, Path]]:
    npy_files = _list_raw_cell_npy(src_dir)
    if not npy_files:
        raise FileNotFoundError(f"[ERROR] No Raw.cell.*.npy files found in {src_dir}")

    entries: list[tuple[float, Path]] = []
    for npy_path in npy_files:
        json_path = _pair_npy_with_json(npy_path)
        t = get_simulation_time_from_json_fast(str(json_path))
        entries.append((t, npy_path))

    entries.sort(key=lambda x: x[0])
    return entries

# --------------------------------------------------------------------------------------
# Snapshot loading
# --------------------------------------------------------------------------------------

def _resolve_colnames_from_json(json_path: Path) -> list[str]:
    return list(get_columns_from_json_cached(str(json_path)))

def _load_raw_cell_df(npy_path: Path, json_path: Path) -> pd.DataFrame:
    arr = np.load(str(npy_path), allow_pickle=False)

    if getattr(arr, "dtype", None) is not None and arr.dtype.names is not None:
        return pd.DataFrame({name: arr[name] for name in arr.dtype.names})

    colnames = _resolve_colnames_from_json(json_path)
    if arr.ndim != 2:
        raise ValueError(f"[ERROR] Expected 2D array in {npy_path}, got shape {arr.shape}")
    if arr.shape[1] != len(colnames):
        raise ValueError(
            f"[ERROR] Column count mismatch in {npy_path}: expected {len(colnames)} (from JSON), got {arr.shape[1]}"
        )
    return pd.DataFrame(arr, columns=colnames)

def _require_cols(df: pd.DataFrame, cols: list[str], context: str = ""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"[ERROR] Missing required columns {missing}"
        if context:
            msg += f" ({context})"
        msg += f". Available columns include: {list(df.columns)[:40]} ..."
        raise ValueError(msg)

# --------------------------------------------------------------------------------------
# Graph builder (stencil on provided ijk)
# --------------------------------------------------------------------------------------
def _df_to_nodes_df(df_ref: pd.DataFrame) -> pd.DataFrame:
    """
    NODE order = sorted by Cell ID (stable across snapshots).
    Must contain: Cell ID, i, j, k, Cell center x/y/z
    Produces nodes.parquet with: node_id, Cell ID, i, j, k, x, y, z
    """
    _require_cols(
        df_ref,
        ["Cell ID", "i", "j", "k", "Cell center x", "Cell center y", "Cell center z"],
        context="build nodes",
    )

    # Force independent frame + deterministic order
    nodes = df_ref.sort_values("Cell ID").reset_index(drop=True).copy(deep=True)

    # node_id
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))

    # CoW-safe dtype coercions
    nodes.loc[:, "Cell ID"] = nodes["Cell ID"].to_numpy(dtype=np.int64)
    nodes.loc[:, "i"] = nodes["i"].to_numpy(dtype=np.int64)
    nodes.loc[:, "j"] = nodes["j"].to_numpy(dtype=np.int64)
    nodes.loc[:, "k"] = nodes["k"].to_numpy(dtype=np.int64)

    # x/y/z aliases
    nodes.loc[:, "x"] = nodes["Cell center x"].to_numpy(dtype=float)
    nodes.loc[:, "y"] = nodes["Cell center y"].to_numpy(dtype=float)
    nodes.loc[:, "z"] = nodes["Cell center z"].to_numpy(dtype=float)

    return nodes.loc[:, ["node_id", "Cell ID", "i", "j", "k", "x", "y", "z"]].copy(deep=True)


def _nodes_to_edge_index(nodes: pd.DataFrame, *, neighbor_set: str = "n6", bidirectional: bool = True) -> np.ndarray:
    required = {"node_id", "i", "j", "k"}
    if not required.issubset(nodes.columns):
        raise ValueError(f"nodes must have columns {required}")

    triplets = list(zip(nodes["i"].to_numpy(), nodes["j"].to_numpy(), nodes["k"].to_numpy()))
    ids = nodes["node_id"].to_numpy()
    lut = {ijk: nid for ijk, nid in zip(triplets, ids)}

    offsets = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                manhattan = abs(di) + abs(dj) + abs(dk)
                if neighbor_set == "n6":
                    if manhattan == 1:
                        offsets.append((di, dj, dk))
                elif neighbor_set == "n18":
                    if 1 <= manhattan <= 2:
                        offsets.append((di, dj, dk))
                elif neighbor_set == "n26":
                    if 1 <= manhattan <= 3:
                        offsets.append((di, dj, dk))
                else:
                    raise ValueError(f"Unknown neighbor_set={neighbor_set}")

    edges = set()
    for (i, j, k), src in zip(triplets, ids):
        for di, dj, dk in offsets:
            nb = (i + di, j + dj, k + dk)
            dst = lut.get(nb)
            if dst is None:
                continue
            if bidirectional:
                a, b = (src, dst) if src <= dst else (dst, src)
                edges.add((a, b))
            else:
                edges.add((src, dst))

    if bidirectional:
        dir_edges = []
        for a, b in edges:
            if a == b:
                continue
            dir_edges.append((a, b))
            dir_edges.append((b, a))
        arr = np.asarray(dir_edges, dtype=np.int64).T
    else:
        arr = np.asarray(list(edges), dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)

    if arr.size:
        N = len(nodes)
        E = arr.shape[1]
        print(f"[EDGESTATS] N={N} directed E={E} avg_out-degree={E / N:.2f} (bidirectional={bidirectional}, neighbor_set={neighbor_set})")
    else:
        print("[EDGESTATS] No edges generated.")
    return arr


def _select_reference_frames(src_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load earliest Raw.cell snapshot and return:
      - df_ref: compact df (for nodes/edges)
      - df_file: same content but preserved in FILE/FLATTEN order
    Both include: Cell ID, Cell center x/y/z, i/j/k (from solver output)
    """
    entries = _build_cell_time_index(src_dir)
    _, npy0 = entries[0]
    json0 = _pair_npy_with_json(npy0)

    df0 = _load_raw_cell_df(npy0, json0)

    # Force a real, independent frame (avoids CoW chained-assignment warnings)
    df0 = df0.copy(deep=True)

    cols = ["Cell ID", "Cell center x", "Cell center y", "Cell center z", "i", "j", "k"]
    _require_cols(df0, cols, context="reference snapshot")

    # Use .loc[:, col] writes (CoW-safe)
    df0.loc[:, "Cell ID"] = df0["Cell ID"].to_numpy(dtype=np.int64)

    for c in ("Cell center x", "Cell center y", "Cell center z"):
        df0.loc[:, c] = df0[c].to_numpy(dtype=float)

    for c in ("i", "j", "k"):
        df0.loc[:, c] = df0[c].to_numpy(dtype=np.int64)

    # Build the returned frames explicitly from df0, with copies
    df_file = df0.loc[:, cols].copy()  # FILE order
    df_ref  = df0.loc[:, cols].copy()  # same content; nodes builder sorts by Cell ID
    return df_ref, df_file

def _build_nodes_and_edges(df_ref: pd.DataFrame, df_file: pd.DataFrame, out_dir: Path, *, edge_bidir: bool = True, neighbor_set: str = "n6"):
    print(f"[GRAPH] Building nodes & edges into {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = _df_to_nodes_df(df_ref)

    # nodes.parquet
    nodes_p = out_dir / "nodes.parquet"
    nodes_p.unlink(missing_ok=True)
    nodes.to_parquet(nodes_p, index=False)
    if not nodes_p.exists():
        raise FileNotFoundError(f"[ERROR] Failed to create {nodes_p}")

    # coords_node_order.parquet
    coords_node_p = out_dir / "coords_node_order.parquet"
    coords_node_p.unlink(missing_ok=True)
    nodes[["Cell ID", "i", "j", "k", "x", "y", "z"]].to_parquet(coords_node_p, index=False)

    # coords_file_order.parquet + colmap_file_to_nodes.npy (for align pipeline)
    df_f = df_file.copy(deep=True)

    df_f.loc[:, "Cell ID"] = df_f["Cell ID"].to_numpy(dtype=np.int64)
    df_f.loc[:, "x"] = df_f["Cell center x"].to_numpy(dtype=float)
    df_f.loc[:, "y"] = df_f["Cell center y"].to_numpy(dtype=float)
    df_f.loc[:, "z"] = df_f["Cell center z"].to_numpy(dtype=float)
    coords_file_p = out_dir / "coords_file_order.parquet"
    coords_file_p.unlink(missing_ok=True)
    df_f[["i", "j", "k", "x", "y", "z"]].to_parquet(coords_file_p, index=False)
    print(f"[GRAPH] wrote coords_file_order.parquet (rows={len(df_f)})")

    # FILE index -> NODE id mapping via Cell ID
    cellid_to_nodeid = dict(zip(nodes["Cell ID"].to_numpy(), nodes["node_id"].to_numpy()))
    file_cellid = df_f["Cell ID"].to_numpy(dtype=np.int64)

    colmap = np.empty_like(file_cellid, dtype=np.int64)
    for ii, cid in enumerate(file_cellid):
        try:
            colmap[ii] = cellid_to_nodeid[int(cid)]
        except KeyError:
            raise ValueError(f"[GRAPH] Cell ID {cid} in FILE order not found in NODE map (unexpected for fixed mesh).")

    np.save(out_dir / "colmap_file_to_nodes.npy", colmap)
    print(f"[GRAPH] wrote colmap_file_to_nodes.npy (len={len(colmap)})")

    # CellID->node_id mapping
    cellmap_p = out_dir / "cellid_to_nodeid.parquet"
    cellmap_p.unlink(missing_ok=True)
    nodes[["Cell ID", "node_id"]].to_parquet(cellmap_p, index=False)

    # Edges (stencil on ijk)
    edge_index = _nodes_to_edge_index(nodes, neighbor_set=neighbor_set, bidirectional=edge_bidir)
    edges_df = pd.DataFrame({"src": edge_index[0], "dst": edge_index[1]}, dtype=np.int64)

    edges_p = out_dir / f"edges_{neighbor_set}.csv"
    edges_p.unlink(missing_ok=True)
    edges_df.to_csv(edges_p, index=False)

    print(f"[GRAPH] wrote nodes.parquet and coords_node_order.parquet (rows={len(nodes)})")
    print(f"[GRAPH] wrote cellid_to_nodeid.parquet")
    print(f"[GRAPH] nodes={len(nodes)}  edges={len(edges_df)} (directed entries)")
    summarize_connectivity(nodes, edge_index, neighbor_set=neighbor_set)

def ensure_graph_artifacts(cfg, field_var: str, rebuild: bool = False, *, neighbor_set: str = "n6") -> Path:
    base = Path(cfg["base_data_dir"])
    test_dir = cfg["test_dir"]
    src_dir = base / test_dir

    graph_dir = Path(cfg["output_dir"]) / "graph" / test_dir
    snap_dir = graph_dir / "snapshots"
    graph_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    nodes_p = graph_dir / "nodes.parquet"
    edges_p = graph_dir / f"edges_{neighbor_set}.csv"
    cellmap_p = graph_dir / "cellid_to_nodeid.parquet"
    coords_file_p = graph_dir / "coords_file_order.parquet"
    colmap_p = graph_dir / "colmap_file_to_nodes.npy"

    need_nodes_edges = rebuild or (not nodes_p.exists()) or (not edges_p.exists()) or (not cellmap_p.exists()) or (not coords_file_p.exists()) or (not colmap_p.exists())
    need_targets = rebuild or (len(list(snap_dir.glob("target_*.*"))) == 0)

    if not _source_has_raw_cell(src_dir):
        raise FileNotFoundError(f"[ERROR] No Raw.cell.*.npy files found under {src_dir}")

    if need_nodes_edges:
        print("[STEP] Creating nodes & edges")
        ref_df, file_df = _select_reference_frames(src_dir)
        _build_nodes_and_edges(ref_df, file_df, graph_dir, edge_bidir=True, neighbor_set=neighbor_set)
    else:
        print("[STEP] Nodes & edges already present  skipping build.")

    if need_targets:
        print("[STEP] Generating snapshot targets")
        nodes = pd.read_parquet(nodes_p).sort_values("node_id").reset_index(drop=True)
        cellmap = pd.read_parquet(cellmap_p)
        node_id_order = nodes[["node_id", "Cell ID"]].copy()

        entries = _build_cell_time_index(src_dir)

        first_json = _pair_npy_with_json(entries[0][1])
        colnames = _resolve_colnames_from_json(first_json)
        if field_var not in colnames:
            raise ValueError(f"[ERROR] field_var '{field_var}' not found in Raw.cell JSON columns. Available: {colnames}")

        for (t, npy_path) in tqdm(entries, total=len(entries), desc="Targets: from Raw.cell", leave=False):
            json_path = _pair_npy_with_json(npy_path)
            df = _load_raw_cell_df(npy_path, json_path)

            _require_cols(df, ["Cell ID", field_var], context=f"targets @ t={t}")

            df = df.copy(deep=True)
            df.loc[:, "Cell ID"] = df["Cell ID"].to_numpy(dtype=np.int64)


            snap = df.loc[:, ["Cell ID", field_var]].copy()
            snap = snap.merge(cellmap, on="Cell ID", how="left")

            if snap["node_id"].isna().any():
                miss = int(snap["node_id"].isna().sum())
                raise ValueError(f"[ERROR] {miss} rows in snapshot have Cell ID not found in reference mapping.")

            snap.loc[:, "node_id"] = snap["node_id"].to_numpy(dtype=np.int64)
            snap = node_id_order.merge(snap[["node_id", field_var]], on="node_id", how="left").sort_values("node_id")

            if snap[field_var].isna().any():
                miss = int(snap[field_var].isna().sum())
                raise ValueError(f"[ERROR] {miss} nodes missing '{field_var}' after alignment (unexpected for fixed mesh).")

            outp = snap_dir / f"target_{t:09.3f}s.parquet"
            outp.parent.mkdir(exist_ok=True, parents=True)
            snap[[field_var]].to_parquet(outp, index=False)
    else:
        print("[STEP] Snapshot targets already present  skipping build.")

    return graph_dir

# --------------------------------------------------------------------------------------
# Public API for in-memory edge construction (integer ijk only)
# --------------------------------------------------------------------------------------

def build_edge_index(
    coords: Union[pd.DataFrame, pd.Index, Iterable[Tuple[int, int, int]]],
    *,
    neighbor_set: str = "n6",
    bidirectional: bool = True,
) -> np.ndarray:
    if isinstance(coords, pd.DataFrame):
        if not {"i", "j", "k"}.issubset(coords.columns):
            raise TypeError("DataFrame must include columns ['i','j','k']")
        nodes = coords[["i", "j", "k"]].copy().reset_index(drop=True)
    elif isinstance(coords, (pd.MultiIndex, pd.Index)):
        vals = list(coords)
        ijk = np.asarray(vals)
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise TypeError("Index must contain (i,j,k) triplets")
        if not np.issubdtype(ijk.dtype, np.integer):
            raise TypeError("Index levels must be integer (i,j,k)")
        nodes = pd.DataFrame(ijk, columns=["i", "j", "k"])
    else:
        ijk = np.asarray(list(coords))
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise TypeError("Iterable must yield (i,j,k) triplets")
        if not np.issubdtype(ijk.dtype, np.integer):
            raise TypeError("Iterable must yield integer (i,j,k) values")
        nodes = pd.DataFrame(ijk, columns=["i", "j", "k"])

    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    return _nodes_to_edge_index(nodes, neighbor_set=neighbor_set, bidirectional=bidirectional)

# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------

def summarize_connectivity(nodes: Union[pd.DataFrame, int], edge_index: np.ndarray, *, neighbor_set: str = "n6") -> dict:
    N = len(nodes) if not isinstance(nodes, int) else nodes
    E = int(edge_index.shape[1]) if edge_index.size else 0
    deg = np.bincount(edge_index[0], minlength=N) + np.bincount(edge_index[1], minlength=N)
    stats = {
        "N": N,
        "E_directed": E,
        "deg_min": int(deg.min()) if N else 0,
        "deg_median": float(np.median(deg)) if N else 0.0,
        "deg_p75": float(np.percentile(deg, 75)) if N else 0.0,
        "deg_max": int(deg.max()) if N else 0,
        "deg_mean": float(deg.mean()) if N else 0.0,
        "expected_E_approx": (6 * N if neighbor_set == "n6" else (18 * N if neighbor_set == "n18" else 26 * N)),
        "zero_deg": int((deg == 0).sum()) if N else 0,
    }
    print(
        f"[GRAPH] Connectivity: N={stats['N']}  E={stats['E_directed']}  "
        f"deg[min/med/p75/max/avg]={stats['deg_min']}/{stats['deg_median']:.1f}/{stats['deg_p75']:.1f}/"
        f"{stats['deg_max']}/{stats['deg_mean']:.2f}  zero_deg={stats['zero_deg']}"
    )
    return stats

def assert_feature_alignment(n_features: int, edge_index: np.ndarray):
    n_graph = int(edge_index.max()) + 1 if edge_index.size else 0
    assert n_features == n_graph, (
        f"Feature nodes {n_features} != graph nodes {n_graph}. "
        f"Ensure consistent node order for features, graph, writer."
    )
