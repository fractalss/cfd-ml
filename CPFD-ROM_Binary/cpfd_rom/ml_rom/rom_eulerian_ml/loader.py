# loader.py  loads snapshots, real times, and supports ordered or full zero-shot splits
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from cpfd_rom.util.io_utils import process_directory_npy
from cpfd_rom.util import config


def _load_times_map(dir_path: str) -> dict:
    """Return {filename -> time(float)} from times.csv, if present."""
    tcsv = os.path.join(dir_path, "times.csv")
    if not os.path.exists(tcsv):
        return {}
    df = pd.read_csv(tcsv)
    if not {"time", "filename"}.issubset(df.columns):
        return {}
    df = df.dropna(subset=["time", "filename"])  # clean
    return {str(row["filename"]): float(row["time"]) for _, row in df.iterrows()}


def _build_pivoted_columns(coords_xyz: np.ndarray) -> pd.MultiIndex:
    """Create a MultiIndex [(x,y,z), ...] for N nodes from an (N,3) array."""
    tuples = [tuple(map(float, row)) for row in coords_xyz]
    return pd.MultiIndex.from_tuples(tuples, names=["x", "y", "z"])


def load_and_preprocess_eulerian_data_npy():
    """
    Loads all snapshots from config.rev_dirs, reads real times from times.csv, builds
    a fixed coordinate mapping, scales snapshots, and returns train/test splits with
    optional ordered splitting or full zero-shot test over all snapshots.

    Returns
    -------
    X_train : (S_train, N, 1)
    X_test  : (S_test,  N, 1)
    P_train : (S_train, P)  (P is usually 1)
    P_test  : (S_test,  P)
    scaler  : fitted MinMaxScaler
    pivoted_columns : MultiIndex of length N with names (x,y,z)
    """
    all_X = []           # list of (S_dir, N) arrays
    all_P = []           # per-snapshot global param values
    all_times = []       # per-snapshot times (float seconds)
    coords_ref = None    # (N,3) from the first snapshot seen

    # Loader knobs with sensible defaults
    test_size_cfg = getattr(config, "test_size", 0.20)              # float in (0,1]; 1.0 ? all test
    preserve_time_order = getattr(config, "preserve_time_order", True)
    zero_shot_use_all = getattr(config, "zero_shot_use_all", False)

    # Iterate each revision directory
    for path in config.rev_dirs:
        dir_path = str(path)

        # 1) Load field snapshots (S_dir, N)
        X_dir, _ = process_directory_npy(dir_path, config.field_variable)
        S_dir, N = X_dir.shape
        all_X.append(X_dir)

        # 2) Parameter value for this directory (replicated per snapshot)
        param_value = config.param_mapping[os.path.basename(dir_path)]
        all_P.extend([param_value] * S_dir)

        # 3) Real times per snapshot (aligned to filenames)
        times_map = _load_times_map(dir_path)
        npy_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])
        # If more files than S_dir (or vice versa), only use the first S_dir
        npy_files = npy_files[:S_dir]
        for idx, fname in enumerate(npy_files):
            all_times.append(float(times_map.get(fname, idx)))  # fallback: sequential index

        # 4) Coordinates reference from the first snapshot
        if coords_ref is None:
            # Try to read coords from the first file (assumes columns x,y,z first)
            snap0 = np.load(os.path.join(dir_path, npy_files[0]))
            coords_ref = snap0[:, :3].astype(float)  # (N,3)

    # Stack to arrays
    X_all = np.vstack(all_X)                 # (S_total, N)
    S_total, N = X_all.shape
    all_P = np.asarray(all_P, dtype=np.float32).reshape(S_total, -1)  # (S_total, P)
    all_times = np.asarray(all_times, dtype=np.float64)               # (S_total,)

    # Scale to (S_total, N, 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all).reshape(S_total, N, 1)

    # Build pivoted columns from coords_ref
    if coords_ref is None:
        raise RuntimeError("No coordinates found to build pivoted_columns.")
    if coords_ref.shape[0] != N:
        # Align coordinate count with features if necessary
        coords_ref = coords_ref[:N]
    pivoted_columns = _build_pivoted_columns(coords_ref)

    # Expose times for downstream consumers
    config.all_times = all_times

    # -------------------------------
    # Split logic
    # -------------------------------
    if zero_shot_use_all or (isinstance(test_size_cfg, (int, float)) and float(test_size_cfg) >= 1.0):
        # Use ALL snapshots as test set (no training split)
        X_train = X_scaled[:0]
        P_train = all_P[:0]
        X_test  = X_scaled
        P_test  = all_P
        config.train_times = np.asarray([], dtype=np.float64)
        config.test_times  = all_times
    else:
        # Convert test_size to float in (0,1)
        if isinstance(test_size_cfg, int):
            test_size = max(1, int(test_size_cfg)) / float(S_total)
        else:
            test_size = float(test_size_cfg)
        test_size = min(max(test_size, 0.0), 0.99)  # keep a non-empty train set

        if preserve_time_order:
            # Sequential split: first (1-test_size) for train, last test_size for test
            split_at = int(round(S_total * (1.0 - test_size)))
            split_at = min(max(split_at, 1), S_total - 1)
            X_train = X_scaled[:split_at]
            P_train = all_P[:split_at]
            X_test  = X_scaled[split_at:]
            P_test  = all_P[split_at:]
            config.train_times = all_times[:split_at]
            config.test_times  = all_times[split_at:]
        else:
            # Shuffle-based split with fixed seed; keep times aligned
            idx = np.arange(S_total)
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
            n_test = int(round(S_total * test_size))
            test_idx = np.sort(idx[:n_test])
            train_idx = np.sort(idx[n_test:])
            X_train = X_scaled[train_idx]
            P_train = all_P[train_idx]
            X_test  = X_scaled[test_idx]
            P_test  = all_P[test_idx]
            config.train_times = all_times[train_idx]
            config.test_times  = all_times[test_idx]

    return X_train, X_test, P_train, P_test, scaler, pivoted_columns
