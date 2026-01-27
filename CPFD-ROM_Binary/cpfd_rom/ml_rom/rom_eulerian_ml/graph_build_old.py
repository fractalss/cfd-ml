# Saurav Mitra
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Tuple, Union

# --------------------------------------------------------------------------------------
# Source discovery helpers (from morning script)
# --------------------------------------------------------------------------------------

def _source_has_npy(src_dir: Path) -> bool:
    return (src_dir / 'columns.txt').exists() and (src_dir / 'times.csv').exists()


def _select_reference_df(src_dir: Path) -> pd.DataFrame:
    """Pick earliest NPY snapshot if present; else error (NPY expected). Enforce integer ijk.
    Returns columns: i, j, k, x, y, z (x/y/z if present in NPY).
    """
    with open(src_dir / 'columns.txt', 'r') as f:
        colnames = [ln.strip() for ln in f if ln.strip()]
    tdf = pd.read_csv(src_dir / 'times.csv').sort_values('time').reset_index(drop=True)
    arr = np.load(src_dir / tdf.iloc[0]['filename'])
    df_ref = pd.DataFrame(arr, columns=colnames)

    for c in ('i', 'j', 'k'):
        if c in df_ref.columns:
            df_ref[c] = df_ref[c].astype(np.int64)
        else:
            raise ValueError("Reference NPY must include integer grid columns i,j,k.")
    for c in ('x', 'y', 'z'):
        if c in df_ref.columns:
            df_ref[c] = df_ref[c].astype(float)

    # If any axis is 0-based, shift to 1-based to match CPFD conventions used elsewhere
    mins = df_ref[['i', 'j', 'k']].min().to_numpy()
    if (mins == 0).any():
        df_ref['i'] += (1 if mins[0] == 0 else 0)
        df_ref['j'] += (1 if mins[1] == 0 else 0)
        df_ref['k'] += (1 if mins[2] == 0 else 0)

    cols = ['i', 'j', 'k', 'x', 'y', 'z']
    present = [c for c in cols if c in df_ref.columns]
    return df_ref[present].copy()


# --------------------------------------------------------------------------------------
# Graph builder (integer ijk only) and artifact writer
# --------------------------------------------------------------------------------------

def _df_to_nodes_df(df_ref: pd.DataFrame) -> pd.DataFrame:
    """Normalize a reference snapshot into nodes with stable node_id preserving row order."""
    nodes = df_ref.copy().reset_index(drop=True)
    nodes.insert(0, 'node_id', np.arange(len(nodes), dtype=np.int64))
    return nodes


def _nodes_to_edge_index(
    nodes: pd.DataFrame,
    *,
    neighbor_set: str = "n6",
    bidirectional: bool = True,
) -> np.ndarray:
    """
    Build a 2xE edge_index from nodes[['node_id','i','j','k']].
    neighbor_set: 'n6' | 'n18' | 'n26'
    - n6 : axis-aligned neighbors only (|di|+|dj|+|dk| == 1)
    - n18: face + edge neighbors   (1 <= |di|+|dj|+|dk| <= 2)
    - n26: include corners         (1 <= |di|+|dj|+|dk| <= 3)
    """
    required = {"node_id", "i", "j", "k"}
    if not required.issubset(nodes.columns):
        raise ValueError(f"nodes must have columns {required}")

    # Build a dictionary (i,j,k) -> node_id for O(1) neighbor lookup
    triplets = list(zip(nodes["i"].to_numpy(), nodes["j"].to_numpy(), nodes["k"].to_numpy()))
    ids      = nodes["node_id"].to_numpy()
    lut = {ijk: nid for ijk, nid in zip(triplets, ids)}

    # Offsets by neighbor set
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

    # Build edges without duplicates
    edges = set()
    for (i, j, k), src in zip(triplets, ids):
        for di, dj, dk in offsets:
            nb = (i + di, j + dj, k + dk)
            dst = lut.get(nb)
            if dst is None:
                continue
            if bidirectional:
                # For undirected logic, store canonical pair once; later expand to both directions
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
        arr = np.asarray(dir_edges, dtype=np.int64).T  # shape (2, E)
    else:
        arr = np.asarray(list(edges), dtype=np.int64).T if edges else np.empty((2,0), dtype=np.int64)

    # Integrity checks (only for n6 enforce exact Manhattan distance = 1)
    if neighbor_set == "n6":
        inv = np.empty(len(nodes), dtype=np.int64)
        inv[nodes["node_id"].to_numpy()] = np.arange(len(nodes))
        ijk_arr = nodes[["i","j","k"]].to_numpy()
        if arr.size:
            s = inv[arr[0]]
            d = inv[arr[1]]
            diffs = np.abs(ijk_arr[s] - ijk_arr[d])
            manhattan = diffs.sum(axis=1)
            if not np.all(manhattan == 1):
                bad = np.where(manhattan != 1)[0][:10]
                print(f"[EDGECHK] WARNING: found {bad.size} non-n6 edges (showing up to 10). Examples manhattan={manhattan[bad]}")
        else:
            print("[EDGECHK] No edges generated.")

    # Quick stats
    if arr.size:
        N = len(nodes)
        E = arr.shape[1]
        avg_outdeg = E / N
        print(f"[EDGESTATS] N={N} directed E={E} avg_out-degree={avg_outdeg:.2f} (bidirectional={bidirectional}, neighbor_set={neighbor_set})")
    return arr



def _build_nodes_and_edges(df_ref: pd.DataFrame, out_dir: Path, edge_bidir: bool = True, *, neighbor_set: str = 'n6'):
    print(f"[GRAPH] Building nodes & edges into {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve FILE/FLATTEN order as provided by df_ref (earliest NPY snapshot)
    # 1) Write coords_file_order.parquet (exact file order used by Y columns)
    coords_cols = [c for c in ['i','j','k','x','y','z'] if c in df_ref.columns]
    coords_file_df = df_ref[coords_cols].copy().reset_index(drop=True)
    (out_dir / 'coords_file_order.parquet').unlink(missing_ok=True)
    coords_file_df.to_parquet(out_dir / 'coords_file_order.parquet', index=False)
    print(f"[GRAPH] wrote coords_file_order.parquet with columns {coords_cols} (rows={len(coords_file_df)})")

    # 2) Nodes dataframe in NODE order  currently we keep the same order as df_ref
    nodes = _df_to_nodes_df(df_ref)
    (out_dir / 'nodes.parquet').unlink(missing_ok=True)
    nodes.to_parquet(out_dir / 'nodes.parquet', index=False)

    # 3) FILE?NODE mapping (colmap); identity because nodes preserve df_ref order
    colmap = np.arange(len(nodes), dtype=np.int64)
    np.save(out_dir / 'colmap_file_to_nodes.npy', colmap)
    print(f"[GRAPH] wrote colmap_file_to_nodes.npy (identity, length={len(colmap)})")

    # 4) Edges in NODE index space
    edge_index = _nodes_to_edge_index(nodes, neighbor_set=neighbor_set, bidirectional=edge_bidir)
    edges_df = pd.DataFrame({'src': edge_index[0], 'dst': edge_index[1]}, dtype=np.int64)
    (out_dir / f'edges_{neighbor_set}.csv').unlink(missing_ok=True)
    edges_df.to_csv(out_dir / f'edges_{neighbor_set}.csv', index=False)

    print(f"[GRAPH] nodes={len(nodes)}  edges={len(edges_df)} (directed entries)")
    summarize_connectivity(nodes, edge_index, neighbor_set=neighbor_set)


def ensure_graph_artifacts(cfg, field_var: str, rebuild: bool = False, *, neighbor_set: str = 'n6') -> Path:
    """Create graph artifacts under <base>/rom_output_gnn/graph/<test_dir>/ using NPY snapshots.
    Writes:
      - nodes.parquet (NODE order)
      - edges_{neighbor_set}.csv (NODE indices)
      - coords_file_order.parquet (FILE/FLATTEN order of columns in Y; columns: i,j,k,[x,y,z])
      - colmap_file_to_nodes.npy (FILE?NODE mapping; identity if nodes keep df_ref order)
      - snapshots/target_*.parquet for field_var (in NODE order)
    """
    base = Path(cfg['base_data_dir']); test_dir = cfg['test_dir']
    src_dir = base / test_dir

    graph_dir = Path(cfg['output_dir']) / 'graph' / test_dir
    snap_dir = graph_dir / 'snapshots'
    graph_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    nodes_p = graph_dir / 'nodes.parquet'
    edges_p = graph_dir / f'edges_{neighbor_set}.csv'

    need_nodes_edges = rebuild or (not nodes_p.exists()) or (not edges_p.exists())
    need_targets = rebuild or (len(list(snap_dir.glob('target_*.*'))) == 0)

    if need_nodes_edges:
        print("[STEP] Creating nodes & edges")
        ref_df = _select_reference_df(src_dir)
        _build_nodes_and_edges(ref_df, graph_dir, edge_bidir=True, neighbor_set=neighbor_set)
    else:
        print("[STEP] Nodes & edges already present  skipping build.")

    if need_targets:
        print("[STEP] Generating snapshot targets")
        nodes = pd.read_parquet(nodes_p)
        key = nodes[['i','j','k','node_id']].copy()
        key[['i','j','k']] = key[['i','j','k']].astype(np.int64)
        if _source_has_npy(src_dir):
            times_df = pd.read_csv(src_dir / 'times.csv').sort_values('time').reset_index(drop=True)
            with open(src_dir / 'columns.txt', 'r') as f:
                colnames = [ln.strip() for ln in f if ln.strip()]
            if field_var not in colnames:
                raise ValueError(f"field_var '{field_var}' not found in columns.txt")
            for _, row in tqdm(times_df.iterrows(), total=len(times_df), desc="Targets: from NPY", leave=False):
                t = float(row['time'])
                arr = np.load(src_dir / row['filename'])
                df = pd.DataFrame(arr, columns=colnames)
                for c in ('i','j','k'):
                    if c in df.columns:
                        df[c] = df[c].astype(np.int64)
                joined = key.merge(df, on=['i','j','k'], how='left').sort_values('node_id')
                targ = joined[[field_var]]
                outp = snap_dir / f"target_{t:09.3f}s.parquet"
                outp.parent.mkdir(exist_ok=True, parents=True)
                targ.to_parquet(outp, index=False)
        else:
            raise FileNotFoundError("No NPY source found (columns.txt/times.csv required) to build targets.")
    else:
        print("[STEP] Snapshot targets already present  skipping build.")

    return graph_dir


# --------------------------------------------------------------------------------------
# Public API for in-memory graph construction (integer ijk only)
# --------------------------------------------------------------------------------------

def build_edge_index(
    coords: Union[pd.DataFrame, pd.Index, Iterable[Tuple[int,int,int]]],
    *,
    neighbor_set: str = 'n6',
    bidirectional: bool = True,
) -> np.ndarray:
    """Build edge_index (2,E) from **integer** grid triplets (i,j,k).

    Accepts:
      - DataFrame with columns ['i','j','k'] (others ignored)
      - MultiIndex/Index of (i,j,k) integers
      - Iterable of 3-int tuples
    """
    if isinstance(coords, pd.DataFrame):
        if not {'i','j','k'}.issubset(coords.columns):
            raise TypeError("DataFrame must include columns ['i','j','k']")
        nodes = coords[['i','j','k']].copy().reset_index(drop=True)
    elif isinstance(coords, (pd.MultiIndex, pd.Index)):
        vals = list(coords)
        ijk = np.asarray(vals)
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise TypeError("Index must contain (i,j,k) triplets")
        if not np.issubdtype(ijk.dtype, np.integer):
            raise TypeError("Index levels must be integer (i,j,k)")
        nodes = pd.DataFrame(ijk, columns=['i','j','k'])
    else:
        ijk = np.asarray(list(coords))
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise TypeError("Iterable must yield (i,j,k) triplets")
        if not np.issubdtype(ijk.dtype, np.integer):
            raise TypeError("Iterable must yield integer (i,j,k) values")
        nodes = pd.DataFrame(ijk, columns=['i','j','k'])

    nodes.insert(0, 'node_id', np.arange(len(nodes), dtype=np.int64))
    return _nodes_to_edge_index(nodes, neighbor_set=neighbor_set, bidirectional=bidirectional)


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------

def summarize_connectivity(nodes: Union[pd.DataFrame, int], edge_index: np.ndarray, *, neighbor_set: str = 'n6') -> dict:
    """Print & return connectivity stats to catch collapsed graphs early."""
    N = len(nodes) if not isinstance(nodes, int) else nodes
    E = int(edge_index.shape[1]) if edge_index.size else 0
    deg = np.bincount(edge_index[0], minlength=N) + np.bincount(edge_index[1], minlength=N)
    stats = {
        'N': N,
        'E_directed': E,
        'deg_min': int(deg.min()) if N else 0,
        'deg_median': float(np.median(deg)) if N else 0.0,
        'deg_p75': float(np.percentile(deg, 75)) if N else 0.0,
        'deg_max': int(deg.max()) if N else 0,
        'deg_mean': float(deg.mean()) if N else 0.0,
        'expected_E_approx': (6*N if neighbor_set=='n6' else (18*N if neighbor_set=='n18' else 26*N)),
        'zero_deg': int((deg==0).sum()) if N else 0,
    }
    print(
        f"[GRAPH] Connectivity: N={stats['N']}  E={stats['E_directed']}  "
        f"deg[min/med/p75/max/avg]={stats['deg_min']}/{stats['deg_median']:.1f}/{stats['deg_p75']:.1f}/{stats['deg_max']}/{stats['deg_mean']:.2f}  "
        f"zero_deg={stats['zero_deg']}"
    )

    return stats


def assert_feature_alignment(n_features: int, edge_index: np.ndarray):
    n_graph = int(edge_index.max()) + 1 if edge_index.size else 0
    assert n_features == n_graph, (
        f"Feature nodes {n_features} != graph nodes {n_graph}. Ensure consistent node order for features, graph, writer.")
