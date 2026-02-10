# cpfd_rom/util/io_utils.py

from __future__ import annotations

import os
import glob
from typing import Tuple

import numpy as np
from tqdm import tqdm

from cpfd_rom.util.file_parsing import (
    get_simulation_time_from_json_fast,
    get_columns_from_json_cached,
)


def _pair_npy_with_json(npy_path: str) -> str:
    """
    Given a .npy path like:
        .../Raw.cell.00001_1.0000e-02.npy
    returns the matching JSON:
        .../Raw.cell.00001_1.0000e-02.json
    """
    base, _ = os.path.splitext(npy_path)
    json_path = base + ".json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] Missing JSON header for {npy_path}: expected {json_path}")
    return json_path


def list_cell_snapshots(directory: str) -> list[str]:
    """
    List Eulerian snapshots (Raw.cell.*.npy) in a directory.
    """
    directory = os.path.abspath(directory)
    pattern = os.path.join(directory, "Raw.cell.*.npy")
    return sorted(glob.glob(pattern))


def build_cell_time_index(directory: str) -> list[tuple[float, str]]:
    """
    Build and return a sorted list of (simulation_time, npy_path) for Raw.cell snapshots.

    Uses fast, cached JSON parsing for simulation time.
    """
    npy_files = list_cell_snapshots(directory)
    if not npy_files:
        raise FileNotFoundError(f"[ERROR] No Raw.cell.*.npy files found in {os.path.abspath(directory)}")

    entries: list[tuple[float, str]] = []
    for npy_path in npy_files:
        json_path = _pair_npy_with_json(npy_path)
        t = get_simulation_time_from_json_fast(json_path)
        entries.append((t, npy_path))

    entries.sort(key=lambda x: x[0])
    return entries


def load_field_from_cell_snapshot(
    npy_path: str,
    field_variable: str,
    column_names: list[str],
    field_index: int,
) -> np.ndarray:
    """
    Load one Raw.cell snapshot and extract a single field as a 1D vector (num_cells,).

    Supports:
      A) structured dtype NPY -> data[field_variable]
      B) plain 2D numeric array -> data[:, field_index]
    """
    data = np.load(npy_path, allow_pickle=False)

    # A) Structured dtype (recommended by BVR: data['Cell center y'])
    if getattr(data, "dtype", None) is not None and data.dtype.names is not None:
        if field_variable not in data.dtype.names:
            raise ValueError(
                f"[ERROR] Structured dtype field '{field_variable}' not found in {npy_path}. "
                f"Available: {list(data.dtype.names)}"
            )
        field_values = np.asarray(data[field_variable]).reshape(-1)
        return field_values

    # B) Plain numeric array with columns in JSON order
    if data.ndim != 2:
        raise ValueError(f"[ERROR] Expected 2D array in {npy_path}, got shape {data.shape}")

    if data.shape[1] != len(column_names):
        raise ValueError(
            f"[ERROR] Column count mismatch in {npy_path}: expected {len(column_names)} "
            f"(from JSON), got {data.shape[1]}"
        )

    return np.asarray(data[:, field_index]).reshape(-1)


def process_directory_npy(directory: str, field_variable: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eulerian-only loader for new BVR output.

    Reads ONLY:
      - Raw.cell.*.npy  (Eulerian snapshots)
      - Raw.cell.*.json (per-snapshot metadata)

    Time handling:
      - Simulation time is read from JSON key: "simulation time"
      - Uses fast regex extraction (no json.load) + caching

    Columns handling:
      - Columns are read once from the FIRST snapshot JSON (json.load once) + caching
      - Field extraction uses either structured dtype field access or column index

    Returns:
      X: (num_snapshots, num_cells)
      param_values: (num_snapshots, 2) object array [case_name, simulation_time]
    """
    directory = os.path.abspath(directory)
    case_name = os.path.basename(os.path.normpath(directory))

    # Build sorted time index (t, npy_path)
    entries = build_cell_time_index(directory)

    # Columns from first snapshot (cached)
    first_json = _pair_npy_with_json(entries[0][1])
    column_names = list(get_columns_from_json_cached(first_json))

    if field_variable not in column_names:
        raise ValueError(
            f"[ERROR] Field variable '{field_variable}' not in JSON columns (Eulerian/cell). "
            f"Available (from {os.path.basename(first_json)}): {column_names}"
        )
    field_index = column_names.index(field_variable)

    data_list: list[np.ndarray] = []
    param_list: list[list[object]] = []

    for t, npy_path in tqdm(entries, desc=f"Processing {case_name} (cell)", unit="snap"):
        field_values = load_field_from_cell_snapshot(
            npy_path=npy_path,
            field_variable=field_variable,
            column_names=column_names,
            field_index=field_index,
        )
        data_list.append(field_values)
        param_list.append([case_name, t])

    X = np.stack(data_list, axis=0)
    param_values = np.array(param_list, dtype=object)
    return X, param_values
