# Modified loader.py and inference.py to support parameter injection
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.saving import load_model, save_model
from cpfd_rom.util.io_utils import process_directory_npy
from cpfd_rom.util import config
from cpfd_rom.util.data_utils import crop_to_divisible_by_4
from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.ml_rom.rom_eulerian_ml.model import train_autoencoder
from cpfd_rom.ml_rom.rom_eulerian_ml.inference import run_inference, find_nearest_velocity, load_input_for_velocity
from cpfd_rom.ml_rom.rom_eulerian_ml.loader import load_and_preprocess_eulerian_data_npy
from cpfd_rom.ml_rom.rom_eulerian_ml.visualization import plot_comparisons as plot_comparisons_ml
from cpfd_rom.ml_rom.rom_eulerian_ml.evaluation import compute_rmse

import joblib

def align_and_evaluate_ml_rom(X_pred, pivoted_columns, times, config, scaler):
    test_data = config.test_df.dropna(subset=[config.field_variable])
    pivoted_test = test_data.pivot_table(index='time', columns=['x', 'y', 'z'], values=config.field_variable)

    X_pred = np.squeeze(X_pred)
    pred_df_all = pd.DataFrame(X_pred)
    pred_df_all.index = times
    pred_df_all.columns = list(pivoted_columns)[:X_pred.shape[1]]

    common_cols_set = set(pivoted_test.columns).intersection(set(pred_df_all.columns))
    common_cols = [col for col in pivoted_columns if col in common_cols_set]
    common_times = sorted(set(pivoted_test.index).intersection(set(pred_df_all.index)))
    if len(common_times) == 0:
        raise ValueError("[ERROR] No common time steps found between CFD and ROM predictions!")

    pred_df_all = pred_df_all.loc[common_times, common_cols]
    pivoted_test = pivoted_test.loc[common_times, common_cols]

    cfd_values = pivoted_test.values
    all_rom_values = pred_df_all.values

    print(f"[DEBUG] Final aligned CFD shape: {cfd_values.shape}")
    print(f"[DEBUG] Final aligned ROM shape: {all_rom_values.shape}")

    assert cfd_values.shape == all_rom_values.shape, f"Mismatch: CFD {cfd_values.shape}, ROM {all_rom_values.shape}"

    pivoted_columns = pd.MultiIndex.from_tuples(common_cols, names=["x", "y", "z"]) if isinstance(common_cols[0], tuple) else pivoted_columns
    rmse_df, merged_snapshots = compute_rmse(
        cfd_values, all_rom_values, common_times, config.target_times, pivoted_columns, scaler
    )

    return rmse_df, merged_snapshots

def run_ml_rom_pipeline(config, log_time):
    model = None
    scaler = None

    setup_output_dir(config)
    setup_model_paths(config)
    model_path = config.model_path_eulerian

    if getattr(config, 'skip_training', False) and os.path.exists(model_path):
        with log_time("Loading pretrained model"):
            print(f"[INFO] Loading pretrained model from {model_path}")
            model = load_model(model_path)
        scaler = joblib.load(model_path.replace(".keras", "_scaler.pkl"))
    else:
        with log_time("Loading and preprocessing Eulerian data for ML"):
            X_train, X_test, P_train, P_test, scaler, pivoted_columns = load_and_preprocess_eulerian_data_npy()

        with log_time("Training ML-based Eulerian ROM"):
            model, history = train_autoencoder(X_train, X_test, P_train, P_test)
            save_model(model, model_path)
            joblib.dump(scaler, model_path.replace(".keras", "_scaler.pkl"))

    with log_time("Inference using nearest parameter"):
        nearest_rev = find_nearest_velocity(config.user_parameter, config.param_mapping)

        # # Load training data for scaler
        # X_train_scaled, _, _, _ = load_input_for_velocity(nearest_rev, config.field_variable, scaler=None, return_scaler=False)
        # if scaler is None:
        #     from sklearn.preprocessing import MinMaxScaler
        #     scaler = MinMaxScaler()
        #     scaler.fit(X_train_scaled.reshape(-1, X_train_scaled.shape[-1]))

        # Now load test data using the trained scaler
        X_scaled, X_raw, times, pivoted_columns = load_input_for_velocity(nearest_rev, config.field_variable, scaler=scaler)

        print(f"[DEBUG] Loaded input shape: {X_scaled.shape}, Raw input shape: {X_raw.shape}, Times: {len(times)}")
        print("[DEBUG] Std of first snapshot:", np.std(X_scaled[0]))
        print("[DEBUG] Std of last snapshot:", np.std(X_scaled[-1]))
        print("[DEBUG] Max diff between snapshots:", np.max(np.abs(X_scaled[0] - X_scaled[-1])))
        print("[DEBUG] Raw field snapshot diffs:", np.linalg.norm(X_raw[0].squeeze() - X_raw[-1].squeeze()))

        dir_path = next(p for p in config.rev_dirs if nearest_rev in p)
        npy_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])

        test_records = []
        for snap_idx, snapshot in enumerate(X_raw.squeeze()):
            t = times[snap_idx]
            for coord, value in zip(pivoted_columns, snapshot):
                x, y, z = coord
                test_records.append({
                    'source': nearest_rev,
                    'time': float(t),
                    'x': x,
                    'y': y,
                    'z': z,
                    config.field_variable: value
                })
        config.test_df = pd.DataFrame(test_records)

        X_pred = run_inference(model, X_scaled, param_value=config.user_parameter)

    with log_time("Evaluating ML-ROM output"):
        rmse_df, merged_snapshots = align_and_evaluate_ml_rom(X_pred, pivoted_columns, times, config, scaler)

    with log_time("Plotting ML-ROM results"):
        filtered_snapshots = [(t, df) for (t, df) in merged_snapshots if any(np.isclose(t, config.target_times))]
        plot_comparisons_ml(filtered_snapshots, config.field_variable, config.target_times, config.user_parameter)
