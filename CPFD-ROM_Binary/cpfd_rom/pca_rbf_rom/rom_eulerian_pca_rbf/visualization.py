import matplotlib.pyplot as plt
import numpy as np
import os
from cpfd_rom.util import config


def plot_explained_variance(cumulative_variance):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_rmse_vs_time(rmse_df):
    plt.figure(figsize=(8, 4))
    plt.plot(rmse_df['time'], rmse_df['RMSE'], marker='o')
    plt.title("RMSE of ROM vs CFD at All Times")
    plt.xlabel("Time [s]")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(config.output_dir, "rmse_pca_rbf_eulerian.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()


def plot_comparisons(merged_snapshots, field_variable, target_times, user_velocity):
    num_times = len(target_times)
    fig, axes = plt.subplots(nrows=2, ncols=num_times, figsize=(7 * num_times, 12), sharex=True, sharey=True)

    for idx, (t, merged_df) in enumerate(merged_snapshots):


        sc1 = axes[0, idx].scatter(merged_df['x'], merged_df['z'], c=merged_df[f'{field_variable}_CFD'], cmap='viridis')
        axes[0, idx].set_title(f"CFD @ t={t:.1f}s")
        axes[0, idx].set_xlabel("x")
        axes[0, idx].set_ylabel("z")
        fig.colorbar(sc1, ax=axes[0, idx], label=f"{field_variable} (CFD)")

        sc2 = axes[1, idx].scatter(merged_df['x'], merged_df['z'], c=merged_df[f'{field_variable}_ROM'], cmap='viridis')
        axes[1, idx].set_title(f"ROM @ t={t:.1f}s")
        axes[1, idx].set_xlabel("x")
        axes[1, idx].set_ylabel("z")
        fig.colorbar(sc2, ax=axes[1, idx], label=f"{field_variable} (ROM)")

    plt.suptitle(f"CFD vs ROM at {user_velocity} m/s for target times", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(config.output_dir, "comparison_pca_rbf_eulerian.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()