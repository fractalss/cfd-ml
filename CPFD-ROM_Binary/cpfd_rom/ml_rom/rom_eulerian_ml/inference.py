import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from cpfd_rom.util import config
from cpfd_rom.util.io_utils import process_directory_npy
from cpfd_rom.util.data_utils import crop_to_divisible_by_4


def find_nearest_velocity(user_velocity, vel_mapping):
    """
    Return the closest velocity key (e.g., 'Rev1_npy') for a given user-specified velocity.
    """
    closest_rev = min(vel_mapping.items(), key=lambda item: abs(item[1] - user_velocity))[0]
    return closest_rev


def load_input_for_velocity(rev_name, field_variable, scaler=None, return_scaler=False):
    """
    Load and process CFD data from the specified training directory (nearest velocity).

    Returns:
        X_scaled: shape (n_samples, n_features, 1)
        X_raw_expanded: unscaled data, shape (n_samples, n_features, 1)
        times: list of time steps
        pivoted_columns: MultiIndex (x, y, z) for cropped points
        scaler (optional): fitted MinMaxScaler
    """
    dir_path = next((p for p in config.rev_dirs if rev_name in p), None)
    if dir_path is None:
        raise ValueError(f"[ERROR] rev_name '{rev_name}' not found in config.rev_dirs")

    # Initialize fallback
    pivoted_columns = pd.MultiIndex.from_arrays([[], [], []], names=['x', 'y', 'z'])

    # Load data
    X_array, param_array = process_directory_npy(dir_path, field_variable)
    if X_array is None or param_array is None or len(X_array) == 0:
        raise ValueError(f"[ERROR] process_directory_npy() returned empty data for {dir_path}")

    times = param_array[:, 1].astype(np.float32)
    sources = param_array[:, 0]

    # Load (x, y, z) coordinates from first npy file
    npy_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {dir_path}")

    first_npy_path = os.path.join(dir_path, npy_files[0])
    sample_snapshot = np.load(first_npy_path)
    if sample_snapshot.shape[1] < 3:
        raise ValueError(f"[ERROR] npy file '{first_npy_path}' does not contain x, y, z + fields")

    coords = [tuple(row[:3]) for row in sample_snapshot]
    coords = coords[:X_array.shape[1]]  # Ensure coords match raw shape

    # Crop X_array to divisible by 4
    X_cropped = crop_to_divisible_by_4(X_array)
    N_cropped = X_cropped.shape[1]

    if N_cropped > len(coords):
        raise ValueError(f"[ERROR] Cropped data has {N_cropped} points, but only {len(coords)} coords available")

    coords_cropped = coords[:N_cropped]  # Align with cropped features
    pivoted_columns = pd.MultiIndex.from_tuples(coords_cropped, names=['x', 'y', 'z'])

    # Add feature dimension
    X_raw_expanded = X_cropped[..., np.newaxis]

    # Fit scaler if not provided
    created_scaler = False
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(X_cropped)
        created_scaler = True

    # Normalize using scaler
    X_scaled = scaler.transform(X_cropped.reshape(X_cropped.shape[0], -1)).reshape(X_raw_expanded.shape)

    if return_scaler and created_scaler:
        return X_scaled, X_raw_expanded, times, pivoted_columns, scaler
    else:
        return X_scaled, X_raw_expanded, times, pivoted_columns


def run_inference(model, X_scaled, param_value):
    """
    Perform reconstruction using the trained conditional AutoEncoder.
    param_value: scalar float or array to be expanded as parameter input.
    """
    P = np.full((X_scaled.shape[0], 1), param_value)  # Broadcast parameter to batch
    print("[DEBUG] Unique velocity inputs to model:", np.unique(P))
    output = model.predict({"field_input": X_scaled, "parameter_input": P})
    print("[DEBUG] Std of predicted first snapshot:", np.std(output[0]))
    print("[DEBUG] Std of predicted last snapshot:", np.std(output[-1]))
    return output
