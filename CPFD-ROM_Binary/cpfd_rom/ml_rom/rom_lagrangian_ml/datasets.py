# cpfd_rom/ml_rom/rom_lagrangian_ml/datasets.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SnapshotDataset(Dataset):
    """Simple dataset for snapshot-wise PointNet autoencoder training.

    Expects a NumPy array of shape [num_snaps, n_points, n_features].

    For the current Lagrangian ROM, we assume:
        n_features = 6
        per-point ordering: [x, y, z, field, CloudID, CloudID_base]

    The dataset returns a single tensor per index with shape [n_points, 6].
    The training loop then applies the loss only to the first 4 channels
    (x, y, z, field), while the last 2 channels are carried through but
    not directly used in the loss.
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
    """Dataset for parameter-conditioned PointNet training.

    Parameters
    ----------
    data_array : np.ndarray
        Array of shape [num_snaps, n_points, n_features]. For the current
        ROM, n_features = 6 with per-point ordering:
            [x, y, z, field, CloudID, CloudID_base].
    params : np.ndarray
        Array of shape [num_snaps, param_dim] providing per-snapshot
        conditioning parameters (e.g., inlet velocity, temperature).

    __getitem__ returns a dict with keys:
        "x":      tensor of shape [n_points, 6]
        "params": tensor of shape [param_dim]
    """

    def __init__(self, data_array: np.ndarray, params: np.ndarray):
        self.data = torch.from_numpy(data_array).float()
        self.params = torch.from_numpy(params).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.data[idx],         # [N, C]
            "params": self.params[idx],  # [param_dim]
        }


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------

def build_params_from_rev_dirs(rev_dirs, param_mapping):
    """Build per-snapshot parameter array using times.csv only.

    Parameters
    ----------
    rev_dirs : sequence of str
        List of Rev*_npy directory paths.
    param_mapping : dict
        Mapping from rev_dir basename (e.g., "Rev1_npy") to a scalar or
        vector of parameters. Currently this helper assumes a scalar per
        rev_dir and expands it for each snapshot row in times.csv.

    Returns
    -------
    all_params : np.ndarray
        Array of shape [total_snaps, param_dim]. For the current logic,
        param_dim is typically 1 (a single scalar per snapshot), but this
        can be extended by making `param_mapping[rev_key]` a list/array.
    """
    all_params = []
    for rev in tqdm(rev_dirs, desc="Building Lagrangian params"):
        rev_key = os.path.basename(os.path.normpath(rev))
        if rev_key not in param_mapping:
            raise KeyError(
                f"Parameter for {rev_key} (from {rev}) not found in param_mapping"
            )
        v = param_mapping[rev_key]

        # ensure v is a 1D array-like for stacking
        v_arr = np.atleast_1d(v).astype(np.float32)

        # count snapshots from times.csv
        times_csv = os.path.join(rev, "times.csv")
        if not os.path.exists(times_csv):
            raise FileNotFoundError(f"[Lagrangian] times.csv not found in {rev}")
        n = len(pd.read_csv(times_csv))  # one row per snapshot

        all_params.extend([v_arr] * n)

    return np.array(all_params, dtype=np.float32)


__all__ = ["SnapshotDataset", "SnapshotParamDataset", "build_params_from_rev_dirs"]
