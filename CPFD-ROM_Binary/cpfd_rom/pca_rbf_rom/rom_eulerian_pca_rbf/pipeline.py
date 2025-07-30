import os
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

from cpfd_rom.pca_rbf_rom import (
    PCARBF_ROM_Generic,
    evaluate_and_collect_snapshots,
    plot_explained_variance,
    plot_rmse_vs_time,
    plot_comparisons as plot_comparisons_pca_rbf,
)
from cpfd_rom.util.io_utils import process_directory_npy
from cpfd_rom.util.output_utils import setup_output_dir
from sklearn.preprocessing import MinMaxScaler

def run_eulerian_pca_rbf_pipeline(config, log_time):
    with log_time("Loading Eulerian training data"):
        print("[INFO] Loading training data from .npy files...")
        data_list = []
        param_list = []

        for d in config.rev_dirs:
            X_d, param_d = process_directory_npy(d, config.field_variable)
            data_list.append(X_d)
            param_list.append(param_d)

        X = np.vstack(data_list).astype(np.float32)
        param_values_raw = np.vstack(param_list)

        sources = param_values_raw[:, 0]
        times = param_values_raw[:, 1].astype(np.float32)
        source_params = np.array([config.param_mapping[src] for src in sources], dtype=np.float32)
        param_values = np.column_stack([source_params, times])

        index = pd.MultiIndex.from_arrays([sources, times], names=['source', 'time'])
        pivoted = pd.DataFrame(X, index=index)

        first_dir = config.rev_dirs[0]
        npy_files = sorted([f for f in os.listdir(first_dir) if f.endswith('.npy')])
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {first_dir}")
        first_npy_path = os.path.join(first_dir, npy_files[0])
        sample_snapshot = np.load(first_npy_path)
        coords = [tuple(row[:3]) for row in sample_snapshot]
        pivoted.columns = pd.MultiIndex.from_tuples(coords, names=['x', 'y', 'z'])

    setup_output_dir(config)

    with log_time("Building and fitting Eulerian PCA ROM"):
        print("[INFO] Building model using PCA with RBF interpolation...")
        model = PCARBF_ROM_Generic()
        model.fit(X, param_values)

    with log_time("Loading Eulerian test data"):
        print("[INFO] Loading test data from test directory...")
        test_data, test_params = process_directory_npy(config.test_directory, config.field_variable)
        sources = test_params[:, 0]
        times = test_params[:, 1].astype(np.float32)
        npy_files = sorted([f for f in os.listdir(config.test_directory) if f.endswith('.npy')])
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {config.test_directory}")
        first_npy_path = os.path.join(config.test_directory, npy_files[0])
        sample_snapshot = np.load(first_npy_path)
        coords = [tuple(row[:3]) for row in sample_snapshot]
        test_index = pd.MultiIndex.from_arrays([sources, times], names=['source', 'time'])
        test_pivoted = pd.DataFrame(test_data, index=test_index, columns=pd.MultiIndex.from_tuples(coords, names=['x', 'y', 'z']))

        config.test_df = test_pivoted
        config.test_times = np.unique(times)

    with log_time("Evaluating Eulerian ROM and plotting RMSE"):
        print("[INFO] Plotting and Saving RMSE data...")
        rmse_df, merged_snapshots = evaluate_and_collect_snapshots(config.test_df, pivoted, model, config)
        plot_rmse_vs_time(rmse_df)

    del pivoted, X, param_values
    gc.collect()

    with log_time("Plotting CFD vs ROM comparison for Eulerian"):
        print("[INFO] Plotting and Saving CFD Vs. ROM comparison data...")
        plot_comparisons_pca_rbf(merged_snapshots, config.field_variable, config.target_times, config.user_parameter)
