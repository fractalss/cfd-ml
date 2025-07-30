# pca_rbf_rom/file_parsing.py

import os
import re

def parse_file_metadata(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    column_lines = [line for line in lines if line.startswith('#@')]
    column_names = [line.split('"')[1] for line in column_lines]
    data_start_idx = len(column_lines) + 3
    return column_names, data_start_idx

def extract_truncated_time(filename):
    """
    Extracts the time from filenames like 'particles_0.005s.txt' or 'cell_0.005s.txt'.
    Dynamically detects which pattern is used.
    """
    base = os.path.basename(filename)
    match = re.search(r'_(\d+\.\d{3})s\.txt$', base)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"[ERROR] Could not extract time from filename: {filename}")

