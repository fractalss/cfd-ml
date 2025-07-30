import os
import glob
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from cpfd_rom.util.file_parsing import parse_file_metadata, extract_truncated_time
from cpfd_rom.util import config

def process_directory_lagrangian(directory):
    dir_path = os.path.abspath(directory)
    files = sorted(glob.glob(os.path.join(dir_path, 'particles*.txt')), key=extract_truncated_time)
    column_names, data_start_idx = parse_file_metadata(files[0])

    times = []
    all_data = []

    for file in tqdm(files, desc=f"Processing {os.path.basename(directory)}", unit="file"):
        df = pd.read_csv(file, skiprows=data_start_idx, sep=r'\s+', header=None)
        df.columns = column_names

        if 'Cloud Id' in df.columns:
            df = df.sort_values(by='Cloud Id').reset_index(drop=True)

        array_data = df[['x', 'y', 'z', config.field_variable]].values.astype(np.float32)
        all_data.append(array_data)

        times.append(extract_truncated_time(file))

        del df, array_data
        gc.collect()

    return np.array(times), all_data  # < return list of arrays, NOT file paths

