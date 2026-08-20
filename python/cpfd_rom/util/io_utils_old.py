
import os
import glob
import pandas as pd
import numpy as np
from cpfd_rom.util.file_parsing import parse_file_metadata, extract_truncated_time
from tqdm import tqdm


# def process_directory(directory):
#     dir_path = os.path.abspath(directory)
#     files = glob.glob(os.path.join(dir_path, 'cell*.txt'))
#     column_names, data_start_idx = parse_file_metadata(files[0])
#     dfs = []
#     for file in tqdm(files, desc=f"Processing {os.path.basename(directory)}", unit="file"):
#         df = pd.read_csv(file, sep=r'\s+', skiprows=data_start_idx, names=column_names)
#         df['time'] = extract_truncated_time(file)
#         df['source'] = os.path.basename(directory)
#         dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)


def process_directory_npy(directory, field_variable):
    """
    Loads all .npy files and times.csv from a directory,
    returning X matrix and parameter values.
    """
    # Load column names
    columns_path = os.path.join(directory, 'columns.txt')
    if not os.path.exists(columns_path):
        raise FileNotFoundError(f"[ERROR] columns.txt not found in {directory}. Please check the conversion step.")

    with open(columns_path, 'r') as f:
        column_names = [line.strip() for line in f.readlines()]

    if field_variable not in column_names:
        raise ValueError(
            f"[ERROR] Field variable '{field_variable}' not found in columns.txt. Available: {column_names}")

    field_index = column_names.index(field_variable)
    times_df = pd.read_csv(os.path.join(directory, 'times.csv'))
    times_df = times_df.sort_values(by='time').reset_index(drop=True)  # Ensure proper time ordering
    data_list = []
    param_list = []
    for _, row in tqdm(times_df.iterrows(), total=times_df.shape[0],
                       desc=f"Processing {os.path.basename(directory)}", unit="file"):

        npy_file = os.path.join(directory, row['filename'])
        array_data = np.load(npy_file)  # shape: (num_cells, 4) ? x, y, z, field
        if array_data.shape[1] != len(column_names):
            raise ValueError(
                f"[ERROR] Column count mismatch in {npy_file}: expected {len(column_names)} from columns.txt, got {array_data.shape[1]}"
            )

        # Extract selected field_variable values
        field_values = array_data[:, field_index]
        data_list.append(field_values)

        param_list.append([os.path.basename(directory), row['time']])

    X = np.stack(data_list)  # shape: (num_snapshots, num_cells)
    param_values = np.array(param_list)
    return X, param_values
