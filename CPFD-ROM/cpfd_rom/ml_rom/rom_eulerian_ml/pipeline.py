import os
import numpy as np
import pandas as pd
from keras.saving import load_model, save_model
from keras import config as keras_config
from cpfd_rom.ml_rom.rom_eulerian_ml.loader import load_and_preprocess_eulerian_data
from cpfd_rom.ml_rom.rom_eulerian_ml.model import train_autoencoder
from cpfd_rom.ml_rom.rom_eulerian_ml.inference import find_nearest_velocity, load_input_for_velocity, run_inference
from cpfd_rom.ml_rom.rom_eulerian_ml.evaluation import compute_rmse
from cpfd_rom.ml_rom.rom_eulerian_ml.visualization import plot_comparisons as plot_comparisons_ml
from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir
import joblib

def align_and_evaluate_ml_rom(X_pred, pivoted_columns, times, config, scaler):
    test_data = config.test_df.dropna(subset=[config.field_variable])
    pivoted_test = test_data.pivot_table(index='time', columns=['x', 'y', 'z'], values=config.field_variable)

    # Ensure pivoted_test.columns is MultiIndex
    if not isinstance(pivoted_test.columns, pd.MultiIndex):
        print("[WARNING] pivoted_test.columns is not MultiIndex  attempting to convert.")
        if all(isinstance(col, tuple) and len(col) == 3 for col in pivoted_test.columns):
            pivoted_test.columns = pd.MultiIndex.from_tuples(pivoted_test.columns, names=["x", "y", "z"])
        else:
            N = len(pivoted_test.columns)
            x_vals = list(range(N))
            y_vals = [0] * N
            z_vals = [0] * N
            pivoted_test.columns = pd.MultiIndex.from_arrays([x_vals, y_vals, z_vals], names=["x", "y", "z"])
            print("[WARNING] Assigned dummy x, y, z coordinates for pivoted_test columns.")

    # Process ROM predictions
    X_pred = np.squeeze(X_pred)
    pred_df_all = pd.DataFrame(X_pred)
    pred_df_all.index = times

    model_columns = list(pivoted_columns)[:X_pred.shape[1]]
    pred_df_all.columns = model_columns

    # Ensure pred_df_all.columns is MultiIndex
    if not isinstance(pred_df_all.columns, pd.MultiIndex):
        print("[WARNING] pred_df_all.columns is not MultiIndex  attempting to convert.")
        if all(isinstance(col, tuple) and len(col) == 3 for col in pred_df_all.columns):
            pred_df_all.columns = pd.MultiIndex.from_tuples(pred_df_all.columns, names=["x", "y", "z"])
        else:
            N = len(pred_df_all.columns)
            x_vals = list(range(N))
            y_vals = [0] * N
            z_vals = [0] * N
            pred_df_all.columns = pd.MultiIndex.from_arrays([x_vals, y_vals, z_vals], names=["x", "y", "z"])
            print("[WARNING] Assigned dummy x, y, z coordinates for pred_df_all columns.")

    # Align common columns and times
    common_cols = sorted(set(pivoted_test.columns).intersection(set(pred_df_all.columns)))
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

    # Rebuild pivoted_columns for downstream functions
    pivoted_columns = pd.MultiIndex.from_tuples(common_cols, names=["x", "y", "z"])

    rmse_df, merged_snapshots = compute_rmse(
        cfd_values, all_rom_values, common_times, config.target_times, pivoted_columns, scaler
    )

    return rmse_df, merged_snapshots




def run_ml_rom_pipeline(config, log_time):
    model = None
    scaler = None
    keras_config.enable_unsafe_deserialization()
    # Setup output directory once at start
    setup_output_dir(config)
    setup_model_paths(config)

    model_path = config.model_path_eulerian
    if getattr(config, 'skip_training', False) and os.path.exists(model_path):
        with log_time("Loading pretrained model"):
            print(f"[INFO] Loading pretrained model from {model_path}")
            model = load_model(model_path)
        scaler_path = model_path.replace(".keras", "_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        with log_time("Loading and preprocessing Eulerian data for ML"):
            print("[INFO] Loading Eulerian training data for ML...")
            X_train, X_test, scaler = load_and_preprocess_eulerian_data()

        with log_time("Training ML-based Eulerian ROM"):
            print("[INFO] Training CNN Autoencoder...")
            model, history = train_autoencoder(X_train, X_test)
            save_model(model, model_path)
            joblib.dump(scaler, model_path.replace(".keras", "_scaler.pkl"))

    with log_time("Inference using nearest parameter"):
        print("[INFO] Running inference...")
        nearest_rev = find_nearest_velocity(config.user_parameter, config.param_mapping)
        print("[DEBUG] Nearest training parameter selected:", nearest_rev)

        X_scaled, X_raw, _, times, pivoted_columns = load_input_for_velocity(nearest_rev, config.field_variable)
        print("[DEBUG] Output lengths:", len(locals()))
        print("[DEBUG] Using function from:", load_input_for_velocity.__code__.co_filename)

        X_pred = run_inference(model, X_scaled)

    with log_time("Evaluating ML-ROM output"):
        rmse_df, merged_snapshots = align_and_evaluate_ml_rom(X_pred, pivoted_columns, times, config, scaler)

    with log_time("Plotting ML-ROM results"):
        # Filter merged_snapshots to target_times only for plotting
        filtered_snapshots = [(t, df) for (t, df) in merged_snapshots if any(np.isclose(t, config.target_times))]
        plot_comparisons_ml(
            filtered_snapshots,
            config.field_variable,
            config.target_times,
            config.user_parameter
        )
