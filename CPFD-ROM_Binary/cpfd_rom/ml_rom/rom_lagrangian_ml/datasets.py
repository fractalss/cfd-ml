# cpfd_rom/ml_rom/rom_lagrangian_ml/datasets.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class SnapshotDataset(Dataset):
    """Simple dataset for (snapshot -> snapshot) autoencoder training.

    Expects a NumPy array of shape [num_snaps, n_points, n_features].
    """

    def __init__(self, data_array: np.ndarray):
        # data_array: [num_snaps, n_points, n_features]
        self.data = torch.from_numpy(data_array).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]  # [n_points, n_features]
        return x


class SnapshotParamDataset(Dataset):
    """Dataset for parameter-conditioned training.

    data_array: [num_snaps, n_points, n_features]
    params:     [num_snaps, param_dim]
    """
    def __init__(self, data_array: np.ndarray, params: np.ndarray):
        self.data = torch.from_numpy(data_array).float()
        self.params = torch.from_numpy(params).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.data[idx],          # [N, C]
            "params": self.params[idx],  # [P]
        }

# in datasets.py

def build_params_from_rev_dirs(rev_dirs, param_mapping):
    """
    Build params using times.csv only (no npy reload).
    """
    all_params = []
    for rev in tqdm(rev_dirs, desc="Building Lagrangian params"):
        rev_key = os.path.basename(os.path.normpath(rev))
        if rev_key not in param_mapping:
            raise KeyError(f"Parameter for {rev_key} (from {rev}) not found in param_mapping")
        v = param_mapping[rev_key]

        # count snapshots from times.csv
        times_csv = os.path.join(rev, "times.csv")
        if not os.path.exists(times_csv):
            raise FileNotFoundError(f"[Lagrangian] times.csv not found in {rev}")
        n = len(pd.read_csv(times_csv))  # one row per snapshot

        all_params.extend([[v]] * n)

    return np.array(all_params, dtype=np.float32)






__all__ = ["SnapshotDataset", "SnapshotParamDataset", "build_params_from_rev_dirs"]
