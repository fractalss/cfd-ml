"""Visualization and Tecplot export helpers for the Lagrangian ROM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata


logger = logging.getLogger(__name__)


def predict_and_plot_rmse(
    model,
    scaler,
    test_scaled: np.ndarray,
    test_array: np.ndarray,
    test_times: Sequence[float],
    out_file: str = "rmse_ml_lagrangian.png",
) -> np.ndarray:
    """Predict the test set, plot per-snapshot RMSE, and export predictions."""
    pred_scaled = model.predict(test_scaled)
    pred_array = scaler.inverse_transform(
        pred_scaled.reshape(-1, scaler.mean_.shape[0])
    ).reshape(pred_scaled.shape)

    rmse_x = np.sqrt(
        np.mean((test_array[:, :, 0] - pred_array[:, :, 0]) ** 2, axis=1)
    )
    rmse_y = np.sqrt(
        np.mean((test_array[:, :, 1] - pred_array[:, :, 1]) ** 2, axis=1)
    )
    rmse_z = np.sqrt(
        np.mean((test_array[:, :, 2] - pred_array[:, :, 2]) ** 2, axis=1)
    )
    rmse_variable = np.sqrt(
        np.mean((test_array[:, :, 3] - pred_array[:, :, 3]) ** 2, axis=1)
    )

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(test_times, rmse_x, label="x")
    plt.plot(test_times, rmse_y, label="y")
    plt.plot(test_times, rmse_z, label="z")
    plt.plot(test_times, rmse_variable, label=config.field_variable)
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")
    plt.title("RMSE per Snapshot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / out_file)
    plt.show()

    output_dir = output_root / "ML"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines = [
        format_metadata(1, "x"),
        format_metadata(2, "y"),
        format_metadata(3, "z"),
        format_metadata(4, config.field_variable),
    ]

    for i, t in enumerate(test_times):
        snapshot = pred_array[i].reshape(-1, 4)
        df_rom = pd.DataFrame(
            snapshot,
            columns=["x", "y", "z", config.field_variable],
        )

        filename = output_dir / f"particles_{t:09.3f}s.txt"
        with filename.open("w", encoding="utf-8") as file_obj:
            file_obj.write('# Zone name = "Particles"\n')
            file_obj.write(f"# Solution time = {t:.6f} s\n")
            for line in metadata_lines:
                file_obj.write(line)
            df_rom.to_csv(
                file_obj,
                sep="\t",
                header=False,
                index=False,
                float_format="%.6e",
            )

    logger.info(
        "Tecplot-compatible Lagrangian ROM output saved to: %s",
        output_dir,
    )
    return pred_array


def plot_selected_snapshots(
    test_times: Sequence[float],
    test_array: np.ndarray,
    pred_array: np.ndarray,
    target_times: Sequence[float] | None = None,
    out_file: str = "comparison_ml_lagrangian.png",
) -> None:
    """Plot true and predicted particle fields near selected target times."""
    if target_times is None:
        target_times = (5.0, 10.0, 15.0)

    times_array = np.asarray(test_times)
    indices = [int(np.argmin(np.abs(times_array - t))) for t in target_times]

    fig, axs = plt.subplots(
        len(indices),
        2,
        figsize=(12, 6 * len(indices)),
        squeeze=False,
    )

    for row, idx in enumerate(indices):
        true_snapshot = test_array[idx]
        pred_snapshot = pred_array[idx]

        sc0 = axs[row, 0].scatter(
            true_snapshot[:, 0],
            true_snapshot[:, 2],
            c=true_snapshot[:, 3],
            cmap="viridis",
            s=1,
        )
        axs[row, 0].set_title(f"True Test_2 at t={times_array[idx]:.2f}s")
        fig.colorbar(sc0, ax=axs[row, 0])

        sc1 = axs[row, 1].scatter(
            pred_snapshot[:, 0],
            pred_snapshot[:, 2],
            c=pred_snapshot[:, 3],
            cmap="viridis",
            s=1,
        )
        axs[row, 1].set_title(
            f"PointNet Prediction at t={times_array[idx]:.2f}s"
        )
        fig.colorbar(sc1, ax=axs[row, 1])

    fig.tight_layout()
    output_path = Path(config.output_dir) / out_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.show()


__all__ = ["predict_and_plot_rmse", "plot_selected_snapshots"]
