import numpy as np
from cpfd_rom.util import config
from cpfd_rom.util.io_utils import process_directory
from sklearn.preprocessing import StandardScaler
from cpfd_rom.util.data_utils import crop_to_divisible_by_4


def find_nearest_velocity(user_velocity, vel_mapping):
    """
    Return the closest velocity key (e.g., 'Rev1') for a given user-specified velocity.
    """
    closest_rev = min(vel_mapping.items(), key=lambda item: abs(item[1] - user_velocity))[0]
    return closest_rev


def load_input_for_velocity(rev_name, field_variable):
    """
    Load and process CFD data from the specified training directory (nearest velocity).

    Returns:
        X_scaled: shape (n_samples, n_features, 1)
        X_raw: unscaled data
        scaler: fitted StandardScaler
        times: list of time steps
    """
    dir_path = next(p for p in config.rev_dirs if rev_name in p)
    df = process_directory(dir_path)

    pivoted = df.pivot_table(index=['source', 'time'], columns=['x', 'y', 'z'], values=field_variable)
    # print("Sample pivoted.columns:")
    # print(pivoted.columns[:10])

    if pivoted.empty:
        raise ValueError(f"[ERROR] No usable data found in {rev_name} for field '{field_variable}'")

    X_raw = pivoted.values[..., np.newaxis]  # shape (n_samples, n_features, 1)
    X_raw = crop_to_divisible_by_4(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.reshape(X_raw.shape[0], -1)).reshape(X_raw.shape)

    times = pivoted.index.get_level_values('time').values

    return X_scaled, X_raw, scaler, times, pivoted.columns


def run_inference(model, X_scaled):
    """
    Perform reconstruction using the trained CNN AutoEncoder.

    Returns:
        X_reconstructed: same shape as X_scaled
    """
    return model.predict(X_scaled)


