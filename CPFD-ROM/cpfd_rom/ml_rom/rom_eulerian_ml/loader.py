import numpy as np
import pandas as pd
import tempfile
import gc
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cpfd_rom.util.io_utils import process_directory
from cpfd_rom.util import config
from cpfd_rom.util.data_utils import crop_to_divisible_by_4

def load_and_preprocess_eulerian_data():
    temp_files = []

    # Step 1: Save each rev_dir to temporary feather (keep only needed columns)
    for path in config.rev_dirs:
        df = process_directory(path)
        df['source'] = os.path.basename(path)
        df = df[['source', 'time', 'x', 'y', 'z', config.field_variable]]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".feather")
        df.reset_index(drop=True).to_feather(tmp_file.name)
        temp_files.append(tmp_file.name)
        del df
        gc.collect()

    print("[DEBUG] Incrementally loading and concatenating data...")
    df_all = None

    for idx, f in enumerate(temp_files):
        df_chunk = pd.read_feather(f)
        if df_all is None:
            df_all = df_chunk
        else:
            df_all = pd.concat([df_all, df_chunk], ignore_index=True)
        del df_chunk
        gc.collect()
        if idx % 10 == 0:
            print(f"[DEBUG] Loaded {idx + 1}/{len(temp_files)} files...")

    # Step 2: Clean up temp files
    for f in temp_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"[WARNING] Could not delete temp file {f}: {e}")
    del temp_files
    gc.collect()

    print("[DEBUG] Creating pivot table...")
    pivoted = df_all.pivot_table(
        index=['source', 'time'],
        columns=['x', 'y', 'z'],
        values=config.field_variable
    )
    del df_all
    gc.collect()

    if pivoted.empty:
        raise ValueError(f"[ERROR] Pivot table is empty for field '{config.field_variable}'")

    # Step 3: Convert to NumPy array
    X = pivoted.values.astype(np.float32)
    X = crop_to_divisible_by_4(X)
    X = X[..., np.newaxis]

    # Step 4: Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

    # Step 5: Train/test split
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

    return X_train, X_test, scaler
