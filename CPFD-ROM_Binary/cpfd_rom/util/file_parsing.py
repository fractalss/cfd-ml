# pca_rbf_rom/file_parsing.py

import os
import re

def parse_file_metadata(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_start_idx = None
    for idx, line in enumerate(lines):
        s = line.strip()
        if s and not s.startswith('#'):
            data_start_idx = idx
            break
    if data_start_idx is None:
        raise ValueError(f"[ERROR] No data in {filename}")
    column_names = []
    for line in lines[:data_start_idx]:
        s = line.lstrip()
        if s.startswith('#@'):
            parts = s.split('"')
            if len(parts) >= 2:
                column_names.append(parts[1])
    if not column_names:
        raise ValueError(f"[ERROR] No '#@' column lines in {filename}")
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

