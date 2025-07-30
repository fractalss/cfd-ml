import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from cpfd_rom.util.file_parsing import parse_file_metadata, extract_truncated_time

def convert_ascii_to_npy_allvars(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, 'cell*.txt')), key=extract_truncated_time)

    if not files:
        print(f"[ERROR] No cell*.txt files found in {input_dir}")
        return

    # Get column names and data start index from the first file
    column_names, data_start_idx = parse_file_metadata(files[0])
    print(f"[INFO] Parsed column names: {column_names}")
    # Save column names to output_dir
    columns_file = os.path.join(output_dir, 'columns.txt')
    with open(columns_file, 'w') as f:
        f.write('\n'.join(column_names))
    print(f"[INFO] Saved column names to: {columns_file}")
    times = []

    for file in tqdm(files, desc=f"Converting {input_dir}"):
        df = pd.read_csv(file, skiprows=data_start_idx, sep=r'\s+', header=None)
        df.columns = column_names

        # Explicit sort by (z, y, x) to ensure consistent ordering
        df = df.sort_values(by=['z', 'y', 'x']).reset_index(drop=True)

        # Select all numeric columns (x, y, z + all variables)
        array_data = df.select_dtypes(include=[np.number]).values.astype(np.float32)

        # Save as .npy
        base_name = os.path.basename(file).replace('.txt', '.npy')
        out_file = os.path.join(output_dir, base_name)
        np.save(out_file, array_data)

        # Save time mapping
        time = extract_truncated_time(file)
        times.append({'filename': base_name, 'time': time})

        # Clean up memory
        del df, array_data

    # Save times.csv for fast lookup later
    times_df = pd.DataFrame(times)
    times_df.to_csv(os.path.join(output_dir, 'times.csv'), index=False)

    print(f"[INFO] Conversion complete. Binary .npy files and times.csv saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert cell*.txt ASCII files to .npy binary files (all variables).")
    parser.add_argument("--input_dir", required=True, help="Input directory containing cell*.txt files")
    parser.add_argument("--output_dir", required=True, help="Output directory to store .npy files")
    args = parser.parse_args()

    convert_ascii_to_npy_allvars(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
