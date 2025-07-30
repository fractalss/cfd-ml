import os
import numpy as np
import gc
from cpfd_rom.pca_rbf_rom import (
    PCARBF_ROM_Generic,
    evaluate_lagrangian_rom,
    plot_lagrangian_rmse,
    plot_snapshot_comparison,
)
from cpfd_rom.pca_rbf_rom.rom_lagrangian_pca_rbf.loader import load_lagrangian_data
from cpfd_rom.util.lagrangian_io import process_directory_lagrangian
from cpfd_rom.util.output_utils import setup_output_dir

def run_lagrangian_pca_rbf_pipeline(config, log_time):
    with log_time("Loading Lagrangian data"):
        rbf_inputs, X = load_lagrangian_data(config.rev_dirs, config.param_mapping)

    with log_time("Training Lagrangian ROM"):
        print("[INFO] Training Lagrangian ROM...")
        model = PCARBF_ROM_Generic()
        model.fit(X, rbf_inputs)
        setup_output_dir(config)
        del X, rbf_inputs
        gc.collect()

    with log_time("Evaluating Lagrangian ROM"):
        print("[INFO] Predicting with Lagrangian ROM...")
        test_times, test_snapshot_files = process_directory_lagrangian(config.test_directory)
        rmse_df, merged_snapshots = evaluate_lagrangian_rom(
            model,
            test_snapshots=test_snapshot_files,
            test_times=test_times,
            user_parameter=config.user_parameter,
            field_variable=config.field_variable,
            target_times=config.target_times
        )
        del test_snapshot_files
        gc.collect()

    with log_time("Plotting Lagrangian ROM results"):
        print("[INFO] Plotting Lagrangian RMSE and snapshot comparisons...")
        plot_lagrangian_rmse(rmse_df, config.field_variable, n_features=4)
        plot_snapshot_comparison(merged_snapshots, config.field_variable, n_features=4)
        del merged_snapshots, rmse_df
        gc.collect()
