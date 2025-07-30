import os
import numpy as np
import gc
from cpfd_rom.util.lagrangian_io import process_directory_lagrangian

def load_lagrangian_data(rev_dirs, param_mapping):
    all_snapshots_flat = []
    parameters = []
    times_all = []

    for rev_path in rev_dirs:
        rev_name = os.path.basename(rev_path)
        param = param_mapping.get(rev_name, None)
        if param is None:
            continue

        times, temp_files = process_directory_lagrangian(rev_path)
        for i, temp_file in enumerate(temp_files):
            snapshot = np.load(temp_file, mmap_mode='r')
            all_snapshots_flat.append(snapshot.flatten())
            parameters.append([param])
            times_all.append(times[i])
            os.remove(temp_file)
            del snapshot
            gc.collect()

    rbf_inputs = np.hstack([
        np.array(times_all).reshape(-1, 1),
        np.array(parameters)
    ])
    X_flat = np.array(all_snapshots_flat)
    del all_snapshots_flat, parameters, times_all
    gc.collect()
    return rbf_inputs, X_flat
