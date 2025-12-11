# cpfd_rom/ml_rom/rom_lagrangian_ml/data_loader.py

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import pandas as pd


def load_lagrangian_snapshots_as_graphs(
    rev_dirs,
    base_data_dir,
    param_mapping,
    field_variable="Particle volume fraction",
    radius=0.1,
    sample_ratio=0.01  # New: float in (0, 1], default 1.0 (no sampling)
):
    """
    Load Lagrangian particle snapshots from CFD runs and convert them into PyTorch Geometric (PyG) graph objects.

    Each graph represents a single snapshot with:
        - Nodes: particles
        - Node features: dynamic scalar field (e.g., particle volume fraction)
        - Positions: (x, y, z)
        - Edges: built using radius graph
        - Graph-level features: [physical param, normalized time]

    Parameters
    ----------
    rev_dirs : List[str]
        Subdirectory names of Rev*_npy folders.
    base_data_dir : str
        Path to the root directory containing all Rev folders.
    param_mapping : Dict[str, float]
        Mapping from Rev subdir names (e.g. 'Rev1_npy') to physical parameter values.
    field_variable : str
        Name of the scalar field variable to extract.
    radius : float
        Radius threshold for building the edge graph.
    sample_ratio : float
        Proportion of snapshots to use from each folder (e.g., 0.1 for 10%).

    Returns
    -------
    graphs : List[Data]
        List of PyG Data objects, each representing a graph per snapshot.
    """
    all_graphs = []
    all_times = []

    print("[DataLoader] Scanning time ranges...")

    # Pass 1: Collect time range across all directories
    for rev_dir in rev_dirs:
        dir_path = os.path.join(base_data_dir, rev_dir)
        times_csv_path = os.path.join(dir_path, "times.csv")

        if not os.path.exists(times_csv_path):
            raise FileNotFoundError(f"[DataLoader] Missing times.csv in: {dir_path}")

        times_df = pd.read_csv(times_csv_path)
        if "time" not in times_df.columns or "filename" not in times_df.columns:
            raise ValueError(f"[DataLoader] times.csv must contain 'time' and 'filename' columns in {dir_path}")

        all_times.extend(times_df["time"].tolist())

    t_min, t_max = min(all_times), max(all_times)
    print(f"[DataLoader] Global time range: t_min={t_min:.4f}, t_max={t_max:.4f}")
    if t_max == t_min:
        raise ValueError("[DataLoader] Global time range is zero  all time values are identical.")

    print("[DataLoader] Loading particle snapshots and constructing graphs...")

    # Pass 2: Load snapshots and construct PyG graphs
    for rev_dir in rev_dirs:
        dir_path = os.path.join(base_data_dir, rev_dir)
        rev_name = os.path.basename(rev_dir)
        if rev_name not in param_mapping:
            raise KeyError(f"[DataLoader] Missing param mapping for rev_dir '{rev_name}' in param_mapping.")

        param_val = param_mapping[rev_name]

        times_df = pd.read_csv(os.path.join(dir_path, "times.csv"))
        times_df = times_df.sort_values("time")

        # Sample only a fraction of snapshots
        n_total = len(times_df)
        n_sample = max(1, int(sample_ratio * n_total))
        sampled_df = times_df.iloc[:: max(1, n_total // n_sample)]

        for _, row in sampled_df.iterrows():
            fname = row["filename"]
            tval = row["time"]

            if not (fname.startswith("particles") and fname.endswith(".npy")):
                continue

            fpath = os.path.join(dir_path, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"[DataLoader] Snapshot file not found: {fpath}")

            print(f"[DataLoader] Loading: {fpath}")
            arr = np.load(fpath)
            if arr.ndim != 2 or arr.shape[1] < 6:
                raise ValueError(f"[DataLoader] Invalid shape in {fpath}, expected [N, 6+], got {arr.shape}")

            pos = torch.tensor(arr[:, 0:3], dtype=torch.float32)
            x_field = torch.tensor(arr[:, 3:4], dtype=torch.float32)
            cloud_id = torch.tensor(arr[:, 4], dtype=torch.long)
            cloud_id_base = torch.tensor(arr[:, 5], dtype=torch.long)

            print(f"[DataLoader] Constructing radius graph (N={pos.size(0)}, radius={radius})")
            edge_index = radius_graph(pos, r=radius, loop=False)
            print(f"[DataLoader] Radius graph done. Edges: {edge_index.size(1)}")

            data = Data(x=x_field, pos=pos, edge_index=edge_index)
            data.time = torch.tensor([tval], dtype=torch.float32)
            data.params = torch.tensor([param_val, (tval - t_min) / (t_max - t_min)], dtype=torch.float32)
            data.cloud_id = cloud_id
            data.cloud_id_base = cloud_id_base
            data.snapshot_name = fname
            data.rev_dir = rev_name

            all_graphs.append(data)

    print(f"[DataLoader] Total graphs loaded: {len(all_graphs)}")
    return all_graphs
