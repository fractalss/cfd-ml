# rom_eulerian_ml/visualization.py
import os
import matplotlib.pyplot as plt
from cpfd_rom.util import config

def plot_comparisons(merged_snapshots, field_variable, target_times, user_velocity):
    num_times = len(merged_snapshots)
    if num_times == 0:
        print("[WARNING] No snapshots to plot  skipping visualization.")
        return
    fig, axes = plt.subplots(nrows=2, ncols=num_times, figsize=(6 * num_times, 10), sharex=True, sharey=True)

    for idx, (t, df) in enumerate(merged_snapshots):
        # CFD
        sc1 = axes[0, idx].scatter(df['x'], df['z'], c=df[f'{field_variable}_CFD'], cmap='viridis')
        axes[0, idx].set_title(f"CFD @ t={t:.1f}s")
        axes[0, idx].set_xlabel("x")
        axes[0, idx].set_ylabel("z")
        fig.colorbar(sc1, ax=axes[0, idx], label=f"{field_variable} (CFD)")

        # ROM
        sc2 = axes[1, idx].scatter(df['x'], df['z'], c=df[f'{field_variable}_ROM'], cmap='viridis')
        axes[1, idx].set_title(f"ROM @ t={t:.1f}s")
        axes[1, idx].set_xlabel("x")
        axes[1, idx].set_ylabel("z")
        fig.colorbar(sc2, ax=axes[1, idx], label=f"{field_variable} (ROM)")

    plt.suptitle(f"ML ROM vs CFD Comparison @ {user_velocity:.1f} m/s", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(config.output_dir, "comparison_ml_eulerian.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()