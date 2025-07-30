# Modified loader.py to support parameter injection
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from cpfd_rom.util.io_utils import process_directory_npy
from cpfd_rom.util import config
from cpfd_rom.util.data_utils import crop_to_divisible_by_4

def load_and_preprocess_eulerian_data_npy():
    all_X = []
    all_P = []
    coords_all = []

    # Load all Rev directories
    for path in config.rev_dirs:
        X_d, param_d = process_directory_npy(path, config.field_variable)
        all_X.append(X_d)

        # Assign operating parameter based on path (e.g., velocity or volume fraction)
        param_value = config.param_mapping[os.path.basename(path)]
        all_P.extend([param_value] * len(X_d))

        # Load coords for all snapshots in this directory
        npy_files = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
        for fname in npy_files:
            snapshot = np.load(os.path.join(path, fname))
            coords_all.append(snapshot[:, :3])  # shape: (num_points, 3)

    # Convert to array (num_snapshots, num_points, 3)
    coords_all = np.array(coords_all)

    # Stack all snapshots
    X_all = np.vstack(all_X)  # shape: (num_snapshots, num_points)
    X_all = X_all[..., np.newaxis]  # Add last dim for channel
    X_all = crop_to_divisible_by_4(X_all)  # Ensure compatibility with decoder

    coords_all = coords_all[:, :X_all.shape[1], :]  # Crop coords to match X_all
    coords = coords_all[0]  # Use coords from first snapshot (assumes grid fixed)

    # NOTE: No coordinate sorting applied here  keep raw order for simplicity

    # Convert parameter list to array and crop accordingly
    all_P = np.array(all_P).reshape(-1, 1)
    N = X_all.shape[0]
    all_P = all_P[:N]  # Ensure shape match after cropping

    # Normalize fields
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all.reshape(N, -1)).reshape(X_all.shape)

    # Train/test split
    X_train, X_test, P_train, P_test = train_test_split(X_scaled, all_P, test_size=0.2, random_state=42)

    # Return also the coords (MultiIndex format)
    pivoted_columns = pd.MultiIndex.from_tuples([tuple(c) for c in coords], names=['x', 'y', 'z'])
    # Save coordinate order used in training
    coord_path = config.model_path_eulerian.replace(".keras", "_coords.npy")
    np.save(coord_path, coords)
    return X_train, X_test, P_train, P_test, scaler, pivoted_columns
