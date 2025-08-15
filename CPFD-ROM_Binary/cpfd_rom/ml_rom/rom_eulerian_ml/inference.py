# Saurav Mitra
# inference_param.py (or inside pipeline.py)
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.utils import to_undirected

def _build_features_xyz(nodes_df: pd.DataFrame) -> torch.Tensor:
    feats = []
    for c in ('x','y','z'):
        v = nodes_df[c].to_numpy(dtype=np.float32)
        mu, sd = float(v.mean()), float(v.std() if v.std() > 0 else 1.0)
        feats.append(((v - mu) / sd).reshape(-1, 1))
    import numpy as np
    X = np.concatenate(feats, axis=1)
    return torch.tensor(X, dtype=torch.float32)

@torch.no_grad()
def infer_on_param(model, cfg, param_value: float, device='auto',
                   time_mode='none', t_stats=None, fourier_m=4):
    dev = (torch.device('cuda') if (device=='auto' and torch.cuda.is_available())
           else torch.device(device if device!='auto' else 'cpu'))

    graph_dir = Path(cfg['output_dir']) / 'graph' / cfg['test_dir']
    nodes_df = pd.read_parquet(graph_dir / 'nodes.parquet').sort_values('node_id').reset_index(drop=True)
    edges_df = pd.read_csv(graph_dir / 'edges_n6.csv')
    edge_index = torch.tensor(edges_df[['src','dst']].to_numpy().T, dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=len(nodes_df))

    # Node features [x,y,z] (standardized)
    x = _build_features_xyz(nodes_df).to(dev)

    # Optional: time features (use t_stats from training if time_mode != 'none')
    def _concat_time(x_t, t):
        if time_mode == 'none':
            return x_t
        # scalar or Fourier time block  reuse your training-time implementation
        from cpfd_rom.ml_rom.rom_eulerian_ml.model_gnn import time_block_for
        tb = time_block_for(t, x_t.size(0), dev, x_t.dtype, time_mode, t_stats, fourier_m)
        return torch.cat([x_t, tb], dim=1)

    # Append param scalar as an extra feature column
    pcol = torch.full((x.size(0), 1), float(param_value), device=dev, dtype=x.dtype)

    # Load times from test_dir (no CFD values needed)
    times_df = pd.read_csv((Path(cfg['base_data_dir']) / cfg['test_dir'] / 'times.csv')).sort_values('time')
    times = times_df['time'].to_numpy(dtype=float)

    model = model.to(dev).eval()
    preds = []
    for t in times:
        x_t = _concat_time(x, float(t))
        feats = torch.cat([x_t, pcol], dim=1)   # [x,y,z,(time),param]
        y = model(feats, edge_index).squeeze(-1).cpu().numpy()  # (N,)
        preds.append(y)
    return np.array(preds), times, nodes_df
