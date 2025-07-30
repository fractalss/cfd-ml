import os
import pandas as pd
import numpy as np
from .evaluation import compute_rmse
from cpfd_rom.util.output_utils import format_metadata

def evaluate_and_collect_snapshots(test_df, pivoted, model, config):
    rmse_list = []
    merged_snapshots = []

    # Create output directory once
    output_dir = os.path.join(config.output_dir, f"PCA-RBF")
    os.makedirs(output_dir, exist_ok=True)

    metadata_lines = [
        format_metadata(1, "x"),
        format_metadata(2, "y"),
        format_metadata(3, "z"),
        format_metadata(4, config.field_variable),
    ]

    coords = pd.DataFrame(pivoted.columns.tolist(), columns=['x', 'y', 'z'])

    # Extract time values from the MultiIndex
    times = test_df.index.get_level_values('time').unique()

    for t in sorted(times):
        # Predict for current time and user parameter
        param_time_pair = [[config.user_parameter, t]]
        X_reconstructed = model.predict(param_time_pair)

        if X_reconstructed is None:
            raise ValueError(f"No reconstructed output found for time {t}")

        if X_reconstructed.ndim > 1:
            X_reconstructed = X_reconstructed.flatten()

        n_spatial_points = coords.shape[0]
        n_features = 1

        # Extract test snapshot from MultiIndex DataFrame

        test_snapshot = test_df.xs(t, level="time", drop_level=False)

        test_snapshot_avg = test_snapshot.mean(axis=0)  # average across sources if multiple

        test_field = coords.copy()
        test_field[config.field_variable] = test_snapshot_avg.values
        test_field[['x', 'y', 'z']] = test_field[['x', 'y', 'z']].astype(np.float64).round(5)
        test_field = test_field.sort_values(by=['z', 'y', 'x']).reset_index(drop=True)
        # Reconstruct DataFrame using coordinates
        reconstructed_df = coords.copy()
        reconstructed_df[config.field_variable] = X_reconstructed
        reconstructed_df[['x', 'y', 'z']] = reconstructed_df[['x', 'y', 'z']].astype(np.float64).round(5)

        reconstructed_df = reconstructed_df.sort_values(by=['z', 'y', 'x']).reset_index(drop=True)
        merged_df = pd.merge(
            test_field,
            reconstructed_df,
            on=['x', 'y', 'z'],
            suffixes=('_CFD', '_ROM'),
            how='inner'
        )

        # Save ROM output to file
        filename = os.path.join(output_dir, f"cells_{t:09.3f}s.txt")
        with open(filename, 'w') as f:
            f.write(f"# Zone name = \"Cells\"\n")
            f.write(f"# Solution time = {t:.6f} s\n")
            for line in metadata_lines:
                f.write(line)
            reconstructed_df.to_csv(f, sep='\t', header=False, index=False, float_format="%.6e")

        rmse, merged_df = compute_rmse(merged_df, config.field_variable)
        rmse_list.append((t, rmse))

        if t in config.target_times:
            merged_snapshots.append((t, merged_df))

    print(f"[INFO] ROM field output files saved to: {output_dir}")
    return pd.DataFrame(rmse_list, columns=['time', 'RMSE']), merged_snapshots
