
import os
import glob
import pandas as pd
import numpy as np
from cpfd_rom.util.file_parsing import parse_file_metadata, extract_truncated_time
from tqdm import tqdm


def process_directory(directory):
    dir_path = os.path.abspath(directory)
    files = glob.glob(os.path.join(dir_path, 'cell*.txt'))
    column_names, data_start_idx = parse_file_metadata(files[0])
    dfs = []
    for file in tqdm(files, desc=f"Processing {os.path.basename(directory)}", unit="file"):
        df = pd.read_csv(file, sep=r'\s+', skiprows=data_start_idx, names=column_names)
        df['time'] = extract_truncated_time(file)
        df['source'] = os.path.basename(directory)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)