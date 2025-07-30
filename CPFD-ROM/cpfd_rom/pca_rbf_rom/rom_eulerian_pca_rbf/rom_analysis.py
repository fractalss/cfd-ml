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
    # Define metadata lines with aligned spacing using fixed-width columns

    metadata_lines = [
        format_metadata(1, "x"),
        format_metadata(2, "y"),
        format_metadata(3, "z"),
        format_metadata(4, config.field_variable),
    ]

    # # Define metadata lines once
    # metadata_lines = [
    #     '#@   1 "x"\t\t\t\t\t\t\t\t\t""\n',
    #     '#@   2 "y"\t\t\t\t\t\t\t\t\t""\n',
    #     '#@   3 "z"\t\t\t\t\t\t\t\t\t""\n',
    #     f'#@   4 "{config.field_variable}"\t\t\t\t\t\t\t\t\t""\n'
    # ]
    coords = pd.DataFrame(pivoted.columns.tolist(), columns=['x', 'y', 'z'])
    for t in sorted(test_df['time'].unique()):
        # Predict ROM output
        X_reconstructed = model.predict([[config.user_parameter, t]])
        if X_reconstructed.ndim > 1:
            X_reconstructed = X_reconstructed.flatten()

        n_spatial_points = coords.shape[0]

        n_features = 1
        if len(X_reconstructed) != n_spatial_points * n_features:
            print(
                f"[WARNING] Skipping t={t:.2f}s due to shape mismatch: expected {n_spatial_points * n_features}, got {len(X_reconstructed)}")
            continue

        #Compute RMSE
        test_snapshot = test_df[np.isclose(test_df['time'], t)]
        test_field = test_snapshot[['x', 'y', 'z', config.field_variable]].copy()
        test_field[['x', 'y', 'z']] = test_field[['x', 'y', 'z']].round(5)
        test_field = test_field.sort_values(by=['x', 'y', 'z']).reset_index(drop=True)

        # Reconstruct DataFrame using x, y, z from pivoted
        # Reconstruct DataFrame using coordinates
        reconstructed_df = coords.copy()
        reconstructed_df[config.field_variable] = X_reconstructed
        reconstructed_df[['x', 'y', 'z']] = reconstructed_df[['x', 'y', 'z']].round(5)

        # Sort reconstructed_df by x, y, z
        reconstructed_df = reconstructed_df.sort_values(by=['z', 'y', 'x']).reset_index(drop=True)
        # Match original CFD ordering
        # try:
        #     reconstructed_df = pd.merge(
        #         test_field[['x', 'y', 'z']],
        #         reconstructed_df,
        #         on=['x', 'y', 'z'],
        #         how='left'
        #     )
        # except Exception as e:
        #     print(f"[WARNING] Merge failed at t={t:.2f}s: {e}")
        #     continue
        # Save ROM output to file
        filename = os.path.join(output_dir, f"cells_{t:09.3f}s.txt")
        with open(filename, 'w') as f:
            f.write(f"# Zone name = \"Cells\"\n")
            f.write(f"# Solution time = {t:.6f} s\n")
            for line in metadata_lines:
                f.write(line)
            reconstructed_df.to_csv(f, sep='\t', header=False, index=False, float_format="%.6e")

        rmse, merged_df = compute_rmse(reconstructed_df, test_field, config.field_variable)
        rmse_list.append((t, rmse))

        if t in config.target_times:
            merged_snapshots.append((t, merged_df))

    print(f"[INFO] ROM field output files saved to: {output_dir}")
    return pd.DataFrame(rmse_list, columns=['time', 'RMSE']), merged_snapshots
