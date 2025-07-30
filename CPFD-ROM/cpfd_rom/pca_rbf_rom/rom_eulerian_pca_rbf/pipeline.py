import os
import numpy as np
import pandas as pd
import gc
import tempfile

from cpfd_rom.pca_rbf_rom import (
    PCARBF_ROM_Generic,
    evaluate_and_collect_snapshots,
    plot_explained_variance,
    plot_rmse_vs_time,
    plot_comparisons as plot_comparisons_pca_rbf,
)
from cpfd_rom.util.io_utils import process_directory
from cpfd_rom.util.output_utils import setup_output_dir

def run_eulerian_pca_rbf_pipeline(config, log_time):
    with log_time("Loading Eulerian training data"):
        print("[INFO] Loading training data...")
        temp_files = []
        for d in config.rev_dirs:
            df = process_directory(d)
            print(f"[DEBUG] {d} memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
            df['source'] = os.path.basename(d)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".feather")
            df.reset_index(drop=True).to_feather(tmp_file.name)
            temp_files.append(tmp_file.name)
            del df
            gc.collect()

        print("[DEBUG] Concatenating dataframes...")
        all_data = pd.concat([pd.read_feather(f) for f in temp_files], ignore_index=True)
        all_data.dropna(subset=[config.field_variable], inplace=True)


    print("[DEBUG] Creating pivot table...")
    pivoted = all_data.pivot_table(index=['source', 'time'], columns=['x', 'y', 'z'], values=config.field_variable)

    # Free memory from large intermediate objects
    del all_data
    gc.collect()
    for f in temp_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"[WARNING] Could not delete temp file {f}: {e}")
    if pivoted.empty:
        raise ValueError(f"[ERROR] Pivot table is empty. Field '{config.field_variable}' has no usable data.")

    X = pivoted.values.astype(np.float32)
    param_values = np.column_stack([
        pivoted.index.get_level_values('source').map(config.param_mapping).values,
        pivoted.index.get_level_values('time').values
    ]).astype(np.float32)

    # Setup output directory once at start
    setup_output_dir(config)

    with log_time("Building and fitting Eulerian PCA-RBF model"):
        print("[INFO] Building model...")
        model = PCARBF_ROM_Generic()
        print("[INFO] Performing inference of model...")
        X_pca = model.fit(X, param_values)

    cumulative_variance = np.cumsum(model.pca.explained_variance_ratio_)
    plot_explained_variance(cumulative_variance)



    test_df = config.test_df

    with log_time("Evaluating Eulerian ROM and plotting RMSE"):
        print("[INFO] Plotting and Saving RMSE data...")
        rmse_df, merged_snapshots = evaluate_and_collect_snapshots(test_df, pivoted, model, config)
        plot_rmse_vs_time(rmse_df)
    # Free pivot table and input arrays
    del pivoted, X, param_values
    gc.collect()

    with log_time("Plotting CFD vs ROM comparison for Eulerian"):
        print("[INFO] Plotting and Saving CFD Vs. ROM comparison data...")
        plot_comparisons_pca_rbf(merged_snapshots, config.field_variable, config.target_times, config.user_parameter)
