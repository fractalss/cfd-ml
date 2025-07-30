# data_loader.py for rom_lagrangian_ml

from cpfd_rom.util.lagrangian_io import process_directory_lagrangian
import numpy as np


def load_lagrangian_snapshots(directories):
    """
    Loads and combines Lagrangian particle data from multiple directories.

    Args:
        directories (list of str): List of directory paths (e.g., ["Rev1", "Rev2", ...])

    Returns:
        tuple: (all_times (np.ndarray), all_data (np.ndarray))
    """
    all_times = []
    all_data = []

    for directory in directories:
        times, data = process_directory_lagrangian(directory)
        all_times.extend(times)
        all_data.extend(data)

    return np.array(all_times), np.array(all_data)
