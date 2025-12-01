# cpfd_rom/ml_rom/rom_lagrangian_ml/data_loader.py
"""Lagrangian snapshot loader for PointNet ROM.

This module is intentionally lightweight: it only knows how to load
per-snapshot particle data from Rev*_npy directories and stack them
into a single array. The semantic meaning and ordering of the feature
channels is enforced at a higher level (pipeline + columns.txt), but we
assume the following for the current Lagrangian ROM:

    Each npy file has shape [n_points, n_features] with
        n_features = 6

    and the features are ordered as:
        [x, y, z, field_variable, CloudID, CloudID_base]

This matches the expectations in:
     pipeline.py   scaling only the first 4 dynamic features and
                     carrying the last 2 IDs through unchanged.
     evaluation.py  writing out the 6 columns in Tecplot-style
                      particles_*.txt using columns.txt as the
                      authoritative header.

If in future you change the number or ordering of features in the
npy snapshots, you must update both this expectation and the
corresponding logic in pipeline.py and evaluation.py.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def _load_one_rev_npy(directory: str):
    """Load times and npy snapshots from a single Rev*_npy directory.

    Expected directory layout:

        directory/
            times.csv   (columns: filename, time)
            *.npy       (each of shape [n_points, n_features])

    For the current Lagrangian ROM, each NPY must have:
        n_features = 6 ? [x, y, z, field, CloudID, CloudID_base]
    but this function does not enforce or reorder columns; it only
    verifies that each file has consistent [N, C] shape and returns
    float32 arrays. The semantic checks happen in the pipeline.
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

    for _, row in tqdm(
        times_df.iterrows(),
        total=len(times_df),
        desc=f"Loading snapshots in {os.path.basename(dir_path)}",
    ):
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
        # Store as float32 for PyTorch
        data_list.append(arr.astype(np.float32))

    # times: list[float], data_list: list[[N, C]]
    times_arr = np.array(times, dtype=np.float32)
    data_arr = np.stack(data_list, axis=0)  # [S, N, C]

    return times_arr, data_arr


def load_lagrangian_snapshots(directories):
    """Load and combine Lagrangian particle data from multiple Rev*_npy directories.

    Parameters
    ----------
    directories : sequence of str
        List of Rev*_npy directory paths, e.g. ["Rev1_npy", "Rev2_npy", ...].

    Returns
    -------
    all_times : np.ndarray
        1D array of length total_snaps with the solution time for each snapshot.
    all_data : np.ndarray
        3D array of shape [total_snaps, n_points, n_features]. For the
        current ROM, n_features should be 6 and ordering must be
        consistent across all directories.
    """
    all_times = []
    all_data = []

    for directory in tqdm(directories, desc="Loading Lagrangian directories"):
        times, data = _load_one_rev_npy(directory)
        all_times.append(times)
        all_data.append(data)

    if not all_times:
        raise RuntimeError("[Lagrangian] No snapshots loaded; check rev_dirs and paths.")

    all_times_arr = np.concatenate(all_times, axis=0)
    all_data_arr = np.concatenate(all_data, axis=0)

    return all_times_arr, all_data_arr
