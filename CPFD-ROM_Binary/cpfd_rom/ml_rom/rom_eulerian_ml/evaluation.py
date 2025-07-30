import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata

def compute_rmse(X_true, X_pred, times, target_times, pivoted_columns, scaler):
    """
    Compute RMSE per snapshot and reconstruct snapshots for target_times.

    Args:
        X_true: numpy array (n_snapshots, n_features, 1)
        X_pred: same shape as X_true
        times: list of snapshot times
        target_times: list of target times for visualization
        pivoted_columns: MultiIndex from pivoted.columns (x, y, z)
        scaler: fitted StandardScaler used for inverse transforming the predictions

    Returns:
        rmse_df: DataFrame with time, RMSE
        merged_snapshots: list of (time, DataFrame) for visualization
    """
    # Inverse transform predictions back to physical units
    X_pred_unscaled = scaler.inverse_transform(X_pred.reshape(X_pred.shape[0], -1)).reshape(X_pred.shape)

    rmse_list = []
    merged_snapshots = []

    # Create output directory for Tecplot-formatted files
    output_dir = os.path.join(config.output_dir, f"ML")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(X_true.shape[0]):
        t = times[i]
        true_flat = X_true[i].reshape(-1)
        pred_flat = X_pred_unscaled[i].reshape(-1)

        # Clip predictions to CFD-expected range (e.g., 0 to 1 for volume fraction)
        pred_flat = np.clip(pred_flat, 0.0, 1.0)

        rmse = root_mean_squared_error(true_flat, pred_flat)
        rmse_list.append((t, rmse))

        # Collect merged snapshot for all times, not just target_times
        N = len(true_flat)
        x_coords = list(pivoted_columns.get_level_values("x"))[:N]
        y_coords = list(pivoted_columns.get_level_values("y"))[:N]
        z_coords = list(pivoted_columns.get_level_values("z"))[:N]

        if not (len(x_coords) == len(y_coords) == len(z_coords) == N):
            raise ValueError(f"[ERROR] Mismatch in spatial dimensions: x={len(x_coords)}, y={len(y_coords)}, z={len(z_coords)}, N={N}")

        df = pd.DataFrame({
            "x": x_coords,
            "y": y_coords,
            "z": z_coords,
            f"{config.field_variable}_CFD": true_flat,
            f"{config.field_variable}_ROM": pred_flat
        })
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float64).round(5)
        merged_snapshots.append((t, df))

        # Write Tecplot-compatible ROM output
        metadata_lines = [
            format_metadata(1, "x"),
            format_metadata(2, "y"),
            format_metadata(3, "z"),
            format_metadata(4, config.field_variable),
        ]
        df = df.sort_values(by=['z', 'y', 'x']).reset_index(drop=True)
        filename = os.path.join(output_dir, f"cells_{t:09.3f}s.txt")
        with open(filename, 'w') as f:
            f.write(f"# Zone name = \"Cells\"\n")
            f.write(f"# Solution time = {t:.6f} s\n")
            for line in metadata_lines:
                f.write(line)
            df[['x', 'y', 'z', f"{config.field_variable}_ROM"]].to_csv(
                f, sep='\t', header=False, index=False, float_format="%.6e")

    rmse_df = pd.DataFrame(rmse_list, columns=['time', 'RMSE'])

    # Plot RMSE vs time for all times
    plt.figure(figsize=(8, 5))
    plt.plot(rmse_df['time'], rmse_df['RMSE'], marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Time (ML ROM)')
    plt.grid(True)
    output_path = os.path.join(config.output_dir, "rmse_ml_eulerian.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()

    return rmse_df, merged_snapshots
