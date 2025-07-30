# visualization_lagrangian.py
import os
import matplotlib.pyplot as plt
from cpfd_rom.util import config


def plot_lagrangian_rmse(rmse_df, field_variable, n_features=None):
    """
    Plot RMSE vs time for all predicted Lagrangian features.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(rmse_df['time'], rmse_df['RMSE_x'], label='x')
    plt.plot(rmse_df['time'], rmse_df['RMSE_y'], label='y')
    plt.plot(rmse_df['time'], rmse_df['RMSE_z'], label='z')
    plt.plot(rmse_df['time'], rmse_df[f'RMSE_{field_variable}'], label=field_variable)
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")
    plt.title("Lagrangian ROM RMSE vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(config.output_dir, "rmse_pca_rbf_lagrangian.png")
    plt.savefig(output_path)
    plt.show()


def plot_snapshot_comparison(merged_snapshots, field_variable, n_features=None):
    """
    Plot and optionally save side-by-side scatter plots of CFD vs ROM snapshots for selected times in a single figure.

    Args:
        merged_snapshots: list of (time, DataFrame) tuples
        field_variable: name of the scalar field to color by
        save_path: file path to save the figure (e.g., 'snapshot_comparison.png')
    """
    n_snapshots = len(merged_snapshots)
    fig, axs = plt.subplots(n_snapshots, 2, figsize=(12, 5 * n_snapshots), sharex=True, sharey=True)

    if n_snapshots == 1:
        axs = [axs]  # ensure iterable even for single row

    for row_idx, (t, df) in enumerate(merged_snapshots):
        ax_cfd = axs[row_idx][0]
        ax_rom = axs[row_idx][1]

        sc1 = ax_cfd.scatter(df['x_CFD'], df['z_CFD'], c=df[f'{field_variable}_CFD'], cmap='viridis', s=2)
        ax_cfd.set_title(f"CFD at t = {t:.2f}s")
        ax_cfd.set_xlabel('x')
        ax_cfd.set_ylabel('z')
        fig.colorbar(sc1, ax=ax_cfd, label=field_variable)

        sc2 = ax_rom.scatter(df['x_ROM'], df['z_ROM'], c=df[f'{field_variable}_ROM'], cmap='viridis', s=2)
        ax_rom.set_title(f"ROM at t = {t:.2f}s")
        ax_rom.set_xlabel('x')
        fig.colorbar(sc2, ax=ax_rom, label=field_variable)

    plt.suptitle(f"Snapshot Comparison at t = {t:.2f}s")
    plt.tight_layout()
    output_path = os.path.join(config.output_dir, "comparison_pca_rbf_lagrangian.png")
    plt.savefig(output_path)
    plt.show()
