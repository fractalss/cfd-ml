# visualization_lagrangian.py for rom_lagrangian_ml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata


def predict_and_plot_rmse(model, scaler, test_scaled, test_array, test_times, out_file="rmse_ml_lagrangian.png"):
    pred_scaled = model.predict(test_scaled)
    pred_array = scaler.inverse_transform(pred_scaled.reshape(-1, scaler.mean_.shape[0])).reshape(pred_scaled.shape)
    rmse_x = np.sqrt(np.mean((test_array[:, :, 0] - pred_array[:, :, 0]) ** 2, axis=1))
    rmse_y = np.sqrt(np.mean((test_array[:, :, 1] - pred_array[:, :, 1]) ** 2, axis=1))
    rmse_z = np.sqrt(np.mean((test_array[:, :, 2] - pred_array[:, :, 2]) ** 2, axis=1))
    rmse_variable = np.sqrt(np.mean((test_array[:, :, 3] - pred_array[:, :, 3]) ** 2, axis=1))


    plt.figure(figsize=(10, 4))
    plt.plot(test_times, rmse_x, label='x')
    plt.plot(test_times, rmse_y, label='y')
    plt.plot(test_times, rmse_z, label='z')
    plt.plot(test_times, rmse_variable, label=f'{config.field_variable}')
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")
    plt.title("RMSE per Snapshot")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(config.output_dir, out_file)
    plt.savefig(output_path)
    plt.tight_layout()
    plt.show()

    output_dir = os.path.join(config.output_dir, f"ML")
    os.makedirs(output_dir, exist_ok=True)

    metadata_lines = [
        format_metadata(1, "x"),
        format_metadata(2, "y"),
        format_metadata(3, "z"),
        format_metadata(4, config.field_variable),
    ]

    for i, t in enumerate(test_times):
        snapshot = pred_array[i].reshape(-1, 4)  # x, y, z, field
        df_rom = pd.DataFrame(snapshot, columns=['x', 'y', 'z', config.field_variable])


        filename = os.path.join(output_dir, f"particles_{t:09.3f}s.txt")
        with open(filename, 'w') as f:
            f.write(f"# Zone name = \"Particles\"\n")
            f.write(f"# Solution time = {t:.6f} s\n")
            for line in metadata_lines:
                f.write(line)
            df_rom.to_csv(f, sep='\t', header=False, index=False, float_format="%.6e")

    print(f"[INFO] Tecplot-compatible Lagrangian ROM output saved to: {output_dir}")
    return pred_array


def plot_selected_snapshots(test_times, test_array, pred_array, target_times=[5.0, 10.0, 15.0],
                            out_file="comparison_ml_lagrangian.png"):
    indices = [np.argmin(np.abs(test_times - t)) for t in target_times]

    fig, axs = plt.subplots(len(indices), 2, figsize=(12, 6 * len(indices)))

    for row, idx in enumerate(indices):
        true_snapshot = test_array[idx]
        pred_snapshot = pred_array[idx]

        sc0 = axs[row, 0].scatter(true_snapshot[:, 0], true_snapshot[:, 2],c=true_snapshot[:, 3], cmap='viridis', s=1)
        axs[row, 0].set_title(f"True Test_2 at t={test_times[idx]:.2f}s")
        plt.colorbar(sc0, ax=axs[row, 0])

        sc1 = axs[row, 1].scatter(pred_snapshot[:, 0], pred_snapshot[:, 2],c=pred_snapshot[:, 3], cmap='viridis', s=1)
        axs[row, 1].set_title(f"PointNet Prediction at t={test_times[idx]:.2f}s")
        plt.colorbar(sc1, ax=axs[row, 1])

    plt.tight_layout()
    output_path = os.path.join(config.output_dir, out_file)
    plt.savefig(output_path)
    plt.show()
