# cpfd_rom/ml_rom/rom_lagrangian_ml/data_loader.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def _load_one_rev_npy(directory: str):
    """
    Load times and npy snapshots from a single Rev*_npy directory.

    Expects:
      directory/
        times.csv (columns: filename, time)
        *.npy      (each of shape [n_points, n_features])
    """
    dir_path = os.path.abspath(directory)

    times_csv = os.path.join(dir_path, "times.csv")
    if not os.path.exists(times_csv):
        raise FileNotFoundError(f"[Lagrangian] times.csv not found in {dir_path}")

    times_df = pd.read_csv(times_csv)
    if "filename" not in times_df.columns or "time" not in times_df.columns:
        raise ValueError(
            f"[Lagrangian] times.csv in {dir_path} must have 'filename' and 'time' columns"
        )

    # Sort by time to ensure temporal ordering
    times_df = times_df.sort_values(by="time").reset_index(drop=True)

    times = []
    data_list = []

    from tqdm import tqdm
    for _, row in tqdm(times_df.iterrows(),
                       total=len(times_df),
                       desc=f"Loading snapshots in {os.path.basename(dir_path)}"):

        fname = row["filename"]
        t = row["time"]
        npy_path = os.path.join(dir_path, fname)
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"[Lagrangian] Expected npy file '{fname}' not found in {dir_path}"
            )

        arr = np.load(npy_path)  # shape: [n_points, n_features]
        if arr.ndim != 2:
            raise ValueError(
                f"[Lagrangian] NPY file '{fname}' in {dir_path} does not have shape [N, C]; "
                f"got {arr.shape}"
            )

        times.append(t)
        data_list.append(arr.astype(np.float32))

    return np.array(times, dtype=np.float32), np.stack(data_list, axis=0)


def load_lagrangian_snapshots(directories):
    """
    Loads and combines Lagrangian particle data from multiple Rev*_npy directories.

    Args:
        directories (list of str): List of directory paths
            (e.g., ["Rev1_npy", "Rev2_npy", ...])

    Returns:
        tuple:
          all_times : np.ndarray [total_snaps]
          all_data  : np.ndarray [total_snaps, n_points, n_features]
    """
    all_times = []
    all_data = []



    for directory in tqdm(directories, desc="Loading Lagrangian directories"):
        times, data = _load_one_rev_npy(directory)
        all_times.append(times)
        all_data.append(data)

    if not all_times:
        raise RuntimeError("[Lagrangian] No snapshots loaded; check rev_dirs and paths.")

    all_times = np.concatenate(all_times, axis=0)
    all_data = np.concatenate(all_data, axis=0)

    return all_times, all_data
