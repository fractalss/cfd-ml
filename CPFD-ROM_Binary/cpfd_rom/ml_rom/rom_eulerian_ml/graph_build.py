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


def _nodes_to_edge_index(nodes: pd.DataFrame, *, neighbor_set: str = 'n6', bidirectional: bool = True) -> np.ndarray:
    """Wire ±1 neighbors in integer (i,j,k). Keeps input order as node_id order. Returns (2,E) int64."""
    if not {'node_id', 'i', 'j', 'k'}.issubset(nodes.columns):
        raise ValueError("nodes DataFrame must include columns ['node_id','i','j','k']")

    if neighbor_set == 'n6':
        deltas = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    elif neighbor_set == 'n18':
        base = [-1,0,1]
        deltas = [(di,dj,dk) for di in base for dj in base for dk in base if (abs(di)+abs(dj)+abs(dk) in (1,2)) and not (di==dj==dk==0)]
    elif neighbor_set == 'n26':
        base = [-1,0,1]
        deltas = [(di,dj,dk) for di in base for dj in base for dk in base if not (di==dj==dk==0)]
    else:
        raise ValueError("neighbor_set must be one of {'n6','n18','n26'}")

    idx = {(int(r.i), int(r.j), int(r.k)): int(r.node_id) for r in nodes.itertuples(index=False)}
    src_list, dst_list = [], []

    for r in tqdm(nodes.itertuples(index=False), total=len(nodes), desc=f"Edges: wiring {neighbor_set}"):
        i, j, k = int(r.i), int(r.j), int(r.k)
        u = int(r.node_id)
        for di, dj, dk in deltas:
            v = idx.get((i+di, j+dj, k+dk))
            if v is not None:
                src_list.append(u); dst_list.append(v)
                if bidirectional and v != u:
                    src_list.append(v); dst_list.append(u)

    edge_index = np.vstack([np.asarray(src_list, dtype=np.int64), np.asarray(dst_list, dtype=np.int64)])
    return edge_index


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
