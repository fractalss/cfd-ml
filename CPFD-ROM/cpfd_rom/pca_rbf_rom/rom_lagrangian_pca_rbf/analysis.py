import os
import numpy as np
import pandas as pd
from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata


def evaluate_lagrangian_rom(model, test_snapshots, test_times, user_parameter, field_variable, target_times):
    """
    Evaluate the Lagrangian ROM by comparing predictions with test snapshots.

    Args:
        model: Trained ROM with `.predict()`
        test_snapshots: np.ndarray of shape (n_snapshots, n_particles, 3)
        test_times: list or array of time points
        user_parameter: float used as fixed velocity input to ROM
        field_variable: str name of the third feature (e.g., "Particle speed")
        target_times: list of times to retain merged snapshots for later plotting

    Returns:
        rmse_df: pd.DataFrame with RMSE per feature over time
        merged_snapshots: list of (time, pd.DataFrame) tuples for target times
    """

    n_features = 4
    feature_labels = ['x', 'y', 'z', field_variable]
    rmse_per_feature = {label: [] for label in feature_labels}
    valid_times = []
    merged_snapshots = []

    for t, snapshot_file in zip(test_times, test_snapshots):
        snapshot = np.load(snapshot_file, mmap_mode='r')
        test_flat = snapshot.flatten()
        pred_flat = model.predict([[t, user_parameter]]).flatten()

        if test_flat.shape != pred_flat.shape:
            raise ValueError(f"[ERROR] Shape mismatch: prediction {pred_flat.shape}, test data {test_flat.shape}")
        valid_times.append(t)
        for i, label in enumerate(feature_labels):
            rmse = np.sqrt(((test_flat[i::n_features] - pred_flat[i::n_features]) ** 2).mean())
            rmse_per_feature[label].append(rmse)
        # Save for later visualization if it's a target time
        if any(np.isclose(t, target_t) for target_t in target_times):
            df = pd.DataFrame({
                'x_CFD': test_flat[0::n_features],
                'y_CFD': test_flat[1::n_features],
                'z_CFD': test_flat[2::n_features],
                f'{field_variable}_CFD': test_flat[3::n_features],
                'x_ROM': pred_flat[0::n_features],
                'y_ROM': pred_flat[1::n_features],
                'z_ROM': pred_flat[2::n_features],
                f'{field_variable}_ROM': pred_flat[3::n_features],
            })

            merged_snapshots.append((t, df))
        del snapshot    # Free memory

    # Ensure all RMSE lists match the length of test_times

    min_len = min(len(valid_times), *[len(rmse_per_feature[label]) for label in feature_labels])
    rmse_df = pd.DataFrame({'time': valid_times[:min_len]})
    for label in feature_labels:
        rmse_df[f'RMSE_{label}'] = rmse_per_feature[label][: min_len]


        # --- Save ROM Outputs for Lagrangian Data ---
    output_dir = os.path.join(config.output_dir, f"PCA-RBF")
    os.makedirs(output_dir, exist_ok=True)
    metadata_lines = [
        format_metadata(1, "x"),
        format_metadata(2, "y"),
        format_metadata(3, "z"),
        format_metadata(4, config.field_variable),
    ]

    for t in test_times:
        pred_raw = model.predict([[t, user_parameter]])
        pred_snapshot = pred_raw.reshape(-1, 4)
        df_rom = pd.DataFrame(pred_snapshot, columns=['x', 'y', 'z', field_variable])
        filename = os.path.join(output_dir, f"particles_{t:09.3f}s.txt")
        with open(filename, 'w') as f:

            f.write(f"# Zone name = \"Particles\"\n")
            f.write(f"# Solution time = {t:.6f} s\n")
            for line in metadata_lines:
                f.write(line)
            df_rom.to_csv(f, sep='\t', header=False, index=False, float_format="%.6e")

    print(f"[INFO] Lagrangian ROM outputs written to: {output_dir}")

    return rmse_df, merged_snapshots
