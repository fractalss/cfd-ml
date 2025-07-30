import os
import pandas as pd
import numpy as np
import time
from contextlib import contextmanager
import argparse
import yaml
import os
import sys
import gc
import tensorflow as tf

from cpfd_rom.util import config
from cpfd_rom.util.io_utils import process_directory
from cpfd_rom.util.lagrangian_io import process_directory_lagrangian
from cpfd_rom.ml_rom.rom_eulerian_ml.pipeline import run_ml_rom_pipeline
from cpfd_rom.ml_rom.rom_lagrangian_ml.pipeline import run_lagrangian_ml_pipeline
from cpfd_rom.pca_rbf_rom.rom_eulerian_pca_rbf.pipeline import run_eulerian_pca_rbf_pipeline
from cpfd_rom.pca_rbf_rom.rom_lagrangian_pca_rbf.pipeline import run_lagrangian_pca_rbf_pipeline

@contextmanager
def log_time(task_name):
    start = time.time()
    yield
    end = time.time()
    print(f"[Timing] {task_name} took {end - start:.2f} seconds")

def clear_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        user_config = yaml.safe_load(f)
    for key, value in user_config.items():
        setattr(config, key, value)

    # Prepend base_data_dir to rev_dirs and test_directory if defined
    if hasattr(config, 'base_data_dir'):
        config.rev_dirs = [os.path.join(config.base_data_dir, d) for d in config.rev_dirs]
        config.test_directory = os.path.join(config.base_data_dir, config.test_dir)

def main():
    parser = argparse.ArgumentParser(
        description="""
    Run ROM pipelines with configuration specified in a YAML file.

    Example usage:
      rom-cli --config_yaml rom_inputs.yaml

    The YAML file should define the following fields:

      rom_type:         Type of ROM to use. One of ['ML', 'PCA-RBF']
      type_of_field:    Type of field data. One of ['Eulerian', 'Lagrangian']
      field_variable:   CFD field to model (e.g., 'particle volume fraction')

      base_data_dir:    Path to the folder containing all rev_dirs and test_directory
      rev_dirs:         List of folders used for training, relative to base_data_dir
      vel_mapping:      Dictionary mapping each rev_dir to a velocity value

      test_directory:   Folder used for testing, relative to base_data_dir

      target_times:     List of snapshot times to evaluate (e.g., [10.0, 20.0])
      user_velocity:    The velocity condition to simulate/test (e.g., 11.0)

      vel_mapping:      Mapping of training folders to velocities (e.g., {'Rev1': 10, 'Rev2': 12})
      test_times:       [Optional] Times available in the test directory for interpolation

    """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--config_yaml", type=str, required=True,
                        help="Path to YAML file with all configuration options.")

    args, unknown = parser.parse_known_args()

    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit(0)

    clear_memory()  # Clear memory before execution

    load_config_from_yaml(args.config_yaml)

    if config.type_of_field == "Eulerian":
        config.model_path = getattr(config, 'model_path_eulerian', 'cnn_autoencoder_eulerian.keras')
    else:
        config.model_path = getattr(config, 'model_path_lagrangian', 'cnn_autoencoder_lagrangian.keras')

    if config.type_of_field == "Lagrangian":
        config.test_times, config.test_df = process_directory_lagrangian(config.test_directory)
    else:
        config.test_df = process_directory(config.test_directory)
        config.test_times = sorted(config.test_df['time'].unique())
    if config.rom_type == "ML" and config.type_of_field == "Eulerian":
        run_ml_rom_pipeline(config, log_time)
    elif config.rom_type == "ML" and config.type_of_field == "Lagrangian":
        run_lagrangian_ml_pipeline(config, log_time)
    elif config.rom_type == "PCA-RBF" and config.type_of_field == "Eulerian":
        run_eulerian_pca_rbf_pipeline(config, log_time)
    elif config.rom_type == "PCA-RBF" and config.type_of_field == "Lagrangian":
        run_lagrangian_pca_rbf_pipeline(config, log_time)
    else:
        raise ValueError("Unsupported ROM type or field type in config.")

    clear_memory()  # Clear memory after execution

if __name__ == "__main__":
    main()
