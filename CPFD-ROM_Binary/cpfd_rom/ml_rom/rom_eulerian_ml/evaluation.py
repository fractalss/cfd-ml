import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata


def _load_nodes_df() -> pd.DataFrame:
    """
    Load the canonical nodes (node_id, x, y, z, i, j, k) that graph_build.py wrote.
    Location matches graph_build.ensure_graph_artifacts():
        <config.output_dir>/graph/<config.test_dir>/nodes.parquet
    """
    nodes_path = Path(config.output_dir) / "graph" / config.test_dir / "nodes.parquet"
    if not nodes_path.exists():
        raise FileNotFoundError(
            f"[ERROR] nodes.parquet not found at {nodes_path}. "
            "Ensure graph_build.ensure_graph_artifacts() wrote it under config.output_dir/graph/<test_dir>/."
        )
    nodes_df = pd.read_parquet(nodes_path)
    required = {"node_id", "x", "y", "z", "i", "j", "k"}
    missing = required - set(nodes_df.columns)
    if missing:
        raise ValueError(f"[ERROR] nodes.parquet missing columns: {sorted(missing)}")
    # Enforce canonical node order
    nodes_df = nodes_df.sort_values("node_id").reset_index(drop=True)
    return nodes_df


def compute_rmse(X_true: np.ndarray,
                 X_pred: np.ndarray,
                 times: np.ndarray,
                 target_times,
                 pivoted_columns,  # kept for signature compatibility; not relied upon for coords anymore
                 scaler):
    """
    Compute RMSE over time and write CPFD-style TXT files for ROM output including:
      x, y, z, i, j, k, <field_variable>_ROM

    Coordinates and (i,j,k) come from nodes.parquet written by graph_build.py,
    guaranteeing the same node ordering used in training/graph construction.

    Returns (rmse_df, merged_snapshots),
      where merged_snapshots is a list of (t, DataFrame) including CFD and ROM columns.
    """
    # Shape checks
    if X_true.ndim != 2 or X_pred.ndim != 2:
        raise ValueError(f"Expected 2D arrays (S,N). Got X_true={X_true.shape}, X_pred={X_pred.shape}")
    if X_true.shape != X_pred.shape:
        raise ValueError(f"Shape mismatch: X_true={X_true.shape}, X_pred={X_pred.shape}")

    S, N = X_true.shape

    # Inverse scale (best-effort for X_true)
    try:
        X_true_phys = scaler.inverse_transform(X_true)
    except Exception:
        X_true_phys = X_true.copy()
    X_pred_phys = scaler.inverse_transform(X_pred)

    # Load canonical nodes once (ensures consistent ordering and (i,j,k))
    nodes_df = _load_nodes_df()
    if len(nodes_df) != N:
        raise ValueError(
            f"[ERROR] Node count mismatch: nodes.parquet has {len(nodes_df)} rows but predictions have N={N}."
        )

    # Prepare CPFD output dir
    output_dir = os.path.join(config.output_dir, "ML")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare metadata header (x,y,z,i,j,k,<field>)
    field_name = config.field_variable
    header_cols = ["x", "y", "z", "i", "j", "k", field_name]
    header_md = [format_metadata(idx + 1, col) for idx, col in enumerate(header_cols)]

    rmse_list = []
    merged_snapshots = []

    # Write all snapshots
    for s in tqdm(range(S), desc="Writing ROM output", unit="snapshot"):
        t = float(times[s])
        true_flat = X_true_phys[s].reshape(-1)
        pred_flat = X_pred_phys[s].reshape(-1)

        if getattr(config, "clip_predictions", False):
            lo = getattr(config, "clip_min", 0.0)
            hi = getattr(config, "clip_max", 1.0)
            pred_flat = np.clip(pred_flat, lo, hi)

        # RMSE at this time
        rmse = root_mean_squared_error(true_flat, pred_flat)
        rmse_list.append((t, rmse))

        # Assemble full snapshot dataframe in canonical node order
        df = nodes_df.copy()
        df[f"{field_name}_CFD"] = true_flat
        df[f"{field_name}_ROM"] = pred_flat

        # For file writing, round xyz (cosmetic) and sort in CPFD-friendly (k,j,i)
        df_out = df.copy()
        df_out[["x", "y", "z"]] = df_out[["x", "y", "z"]].astype(np.float64).round(5)
        df_out = df_out.sort_values(by=["k", "j", "i"], kind="mergesort").reset_index(drop=True)

        # Save CPFD-style TXT with ROM values (x y z i j k field_ROM)
        filename = os.path.join(output_dir, f"cells_{t:09.3f}s.txt")
        with open(filename, "w") as f:
            f.write('# Zone name = "Cells"\n')
            f.write(f"# Solution time = {t:.6f} s\n")
            for line in header_md:
                f.write(line)
            df_out[["x", "y", "z", "i", "j", "k", f"{field_name}_ROM"]].to_csv(
                f, sep="\t", header=False, index=False, float_format="%.6e"
            )

        # Keep a richer dataframe (with CFD & ROM) for diagnostics/plotting
        merged_snapshots.append((t, df))

    # RMSE timeline and plot
    rmse_df = pd.DataFrame(rmse_list, columns=["time", "RMSE"])
    plt.figure(figsize=(8, 5))
    plt.plot(rmse_df["time"], rmse_df["RMSE"], marker="o", linestyle="-")
    plt.xlabel("Time (s)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Time (ML ROM)")
    plt.grid(True)
    output_path = os.path.join(config.output_dir, "rmse_ml_eulerian.png")
    plt.savefig(output_path)
    plt.show()
    plt.close()

    return rmse_df, merged_snapshots
