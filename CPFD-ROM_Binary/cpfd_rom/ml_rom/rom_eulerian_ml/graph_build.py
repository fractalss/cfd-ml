import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


def _build_nodes_and_edges(df_ref: pd.DataFrame, out_dir: Path, edge_bidir: bool = True):
    print(f"[GRAPH] Building nodes & edges into {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = df_ref.reset_index(drop=True).copy()
    nodes.insert(0, 'node_id', np.arange(len(nodes), dtype=np.int64))
    nodes.to_parquet(out_dir / 'nodes.parquet', index=False)

    idx = {(int(r.i), int(r.j), int(r.k)): int(r.node_id) for r in nodes.itertuples(index=False)}
    deltas = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    src_list, dst_list = [], []

    for r in tqdm(nodes.itertuples(index=False), total=len(nodes), desc="Edges: wiring 6-neighbors"):
        i,j,k = int(r.i), int(r.j), int(r.k)
        u = int(r.node_id)
        for di,dj,dk in deltas:
            v = idx.get((i+di, j+dj, k+dk))
            if v is not None:
                src_list.append(u); dst_list.append(v)
                if edge_bidir and v != u:
                    src_list.append(v); dst_list.append(u)
    edges = pd.DataFrame({'src': src_list, 'dst': dst_list}, dtype=np.int64)
    edges.to_csv(out_dir / 'edges_n6.csv', index=False)
    print(f"[GRAPH] nodes={len(nodes)}  edges={len(edges)} (directed entries)")
def ensure_graph_artifacts(cfg, field_var: str, rebuild: bool = False):
    base = Path(cfg['base_data_dir']); test_dir = cfg['test_dir']
    src_dir = base / test_dir
    graph_dir = base / 'rom_output_gnn' / 'graph' / test_dir
    snap_dir = graph_dir / 'snapshots'
    graph_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    nodes_p = graph_dir / 'nodes.parquet'
    edges_p = graph_dir / 'edges_n6.csv'

    need_nodes_edges = rebuild or (not nodes_p.exists()) or (not edges_p.exists())
    need_targets = rebuild or (len(list(snap_dir.glob('target_*.*'))) == 0)

    if need_nodes_edges:
        print("[STEP] Creating nodes & edges")
        ref_df = _select_reference_df(src_dir)
        _build_nodes_and_edges(ref_df, graph_dir, edge_bidir=True)
    else:
        print("[STEP] Nodes & edges already present  skipping build.")

    if need_targets:
        print("[STEP] Generating snapshot targets")
        nodes = pd.read_parquet(graph_dir / 'nodes.parquet')
        key = nodes[['i','j','k','node_id']].copy()
        key[['i','j','k']] = key[['i','j','k']].astype(np.int64)
        # Prefer NPY if available

        times_df = pd.read_csv(src_dir / 'times.csv').sort_values('time').reset_index(drop=True)
        for _, row in tqdm(times_df.iterrows(), total=len(times_df), desc="Targets: from NPY", leave=False):
            t = float(row['time'])
            arr = np.load(src_dir / row['filename'])
            df = pd.DataFrame(arr)
            # Ensure column names by reading columns.txt once
            with open(src_dir / 'columns.txt', 'r') as f:
                colnames = [ln.strip() for ln in f if ln.strip()]
            df.columns = colnames
            for c in ('i','j','k'):
                if c in df.columns:
                    df[c] = df[c].astype(np.int64)
            joined = key.merge(df, on=['i','j','k'], how='left').sort_values('node_id')
            targ = joined[[field_var]]
            (snap_dir / f"target_{t:09.3f}s.parquet").parent.mkdir(exist_ok=True, parents=True)
            targ.to_parquet(snap_dir / f"target_{t:09.3f}s.parquet", index=False)

    else:
        print("[STEP] Snapshot targets already present  skipping build.")

    return graph_dir
