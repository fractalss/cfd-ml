# util/model_utils.py
import os


def setup_model_paths(config):
    """
    Setup rom_output/model_files directory and update model paths in config.
    """
    model_dir = os.path.join(config.base_data_dir, "model_files")
    os.makedirs(model_dir, exist_ok=True)
    config.model_dir = model_dir
    config.model_path_eulerian = os.path.join(model_dir, "cnn_autoencoder_eulerian.keras")
    config.model_path_lagrangian = os.path.join(model_dir, "cnn_autoencoder_lagrangian.keras")
    print(f"[INFO] Model directory set up at: {model_dir}")
