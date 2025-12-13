import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import pandas as pd


def compute_feature_stats(graphs):
    """
    Compute normalization stats from unnormalized graphs:
    - pos: StandardScaler (mean/std)
    - field: MinMaxScaler (min/max)
    """
    all_pos = torch.cat([g.pos for g in graphs], dim=0)
    all_field = torch.cat([g.x[:, -1:] for g in graphs], dim=0)
    return {
        "pos_mean": all_pos.mean(dim=0),
        "pos_std": all_pos.std(dim=0),
        "field_min": all_field.min().item(),
        "field_max": all_field.max().item(),
    }


def load_lagrangian_snapshots_as_graphs(
    rev_dirs,
    base_data_dir,
    param_mapping,
    field_variable="Particle volume fraction",
    radius=0.1,
    sample_ratio=0.01,
    feature_stats=None,
):
    all_graphs = []
    all_times = []

    print("[DataLoader] Scanning time ranges...")

    for rev_dir in rev_dirs:
        dir_path = os.path.join(base_data_dir, rev_dir)
        times_csv_path = os.path.join(dir_path, "times.csv")

        if not os.path.exists(times_csv_path):
            raise FileNotFoundError(f"Missing times.csv in: {dir_path}")

        times_df = pd.read_csv(times_csv_path)
        if "time" not in times_df.columns or "filename" not in times_df.columns:
            raise ValueError(f"'time' and 'filename' must be in {times_csv_path}")

        all_times.extend(times_df["time"].tolist())

    t_min, t_max = min(all_times), max(all_times)
    print(f"[DataLoader] Global time range: t_min={t_min:.4f}, t_max={t_max:.4f}")
    if t_max == t_min:
        raise ValueError("Zero time range  all time values are identical.")

    print("[DataLoader] Loading particle snapshots and constructing graphs...")

    for rev_dir in rev_dirs:
        dir_path = os.path.join(base_data_dir, rev_dir)
        rev_name = os.path.basename(rev_dir)
        if rev_name not in param_mapping:
            raise KeyError(f"Missing param mapping for rev_dir '{rev_name}'")

        param_val = param_mapping[rev_name]

        times_df = pd.read_csv(os.path.join(dir_path, "times.csv")).sort_values("time")
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
                raise FileNotFoundError(f"Snapshot file not found: {fpath}")

            print(f"[DataLoader] Loading: {fpath}")
            arr = np.load(fpath)
            if arr.ndim != 2 or arr.shape[1] < 10:
                raise ValueError(f"Expected shape [N, 10+], got {arr.shape} in {fpath}")

            pos = torch.tensor(arr[:, 0:3], dtype=torch.float32)                 # x, y, z
            x_field = torch.tensor(arr[:, 9:10], dtype=torch.float32)           # field
            cloud_id = torch.tensor(arr[:, 3], dtype=torch.long)
            cloud_id_base = torch.tensor(arr[:, 4], dtype=torch.long)

            if feature_stats is not None:
                pos = (pos - feature_stats["pos_mean"]) / (feature_stats["pos_std"] + 1e-8)
                x_field = (x_field - feature_stats["field_min"]) / (feature_stats["field_max"] - feature_stats["field_min"] + 1e-8)

            edge_index = radius_graph(pos, r=radius, loop=False)
            features = torch.cat([pos, x_field], dim=1)

            data = Data(x=features, pos=pos, edge_index=edge_index)
            data.y = features.clone()
            data.time = torch.tensor([tval], dtype=torch.float32)
            data.params = torch.tensor([[param_val, (tval - t_min) / (t_max - t_min)]], dtype=torch.float32)
            data.cloud_id = cloud_id
            data.cloud_id_base = cloud_id_base
            data.snapshot_name = fname
            data.rev_dir = rev_name

            all_graphs.append(data)

    print(f"[DataLoader] Total graphs loaded: {len(all_graphs)}")
    return all_graphs


def extract_scaffold_graphs(
    rev_dirs,
    base_data_dir,
    param_mapping,
    param_train_array,
    user_param_array,
    field_variable="Particle volume fraction",
    radius=0.1,
    sample_ratio=0.01,
    feature_stats=None
):
    """
    Extract scaffold graphs from the Rev closest to user_param_array in parameter space.
    """
    param_train_array = np.asarray(param_train_array)
    user_param_array = np.asarray(user_param_array).reshape(1, -1)

    if user_param_array.shape[1] != param_train_array.shape[1] - 1:
        raise ValueError("user_param_array dimensionality must match training param_dim - 1")

    # Only compare physical parameter part (not time component)
    dists = np.linalg.norm(param_train_array[:, :-1] - user_param_array, axis=1)
    nearest_idx = np.argmin(dists)

    # Identify the Rev associated with this index
    rev_dir_list = list(rev_dirs)
    if nearest_idx >= len(rev_dir_list):
        raise IndexError("Nearest parameter index exceeds rev_dirs length")

    nearest_rev = rev_dir_list[nearest_idx]
    print(f"[Scaffold] Using nearest Rev: {nearest_rev} for user_param {user_param_array.flatten()}")

    graphs = load_lagrangian_snapshots_as_graphs(
        rev_dirs=[nearest_rev],
        base_data_dir=base_data_dir,
        param_mapping=param_mapping,
        field_variable=field_variable,
        radius=radius,
        sample_ratio=sample_ratio,
        feature_stats=feature_stats,
    )

    if not graphs:
        raise ValueError(f"No graphs found for Rev dir: {nearest_rev}")

    return graphs  # all snapshots for the nearest Rev


__all__ = [
    "load_lagrangian_snapshots_as_graphs",
    "compute_feature_stats",
    "extract_scaffold_graphs"
]
