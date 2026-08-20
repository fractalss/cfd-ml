# cpfd_rom/util/file_parsing.py

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, List, Tuple


# ----------------------------------------------------------------------
# FAST JSON PARSING UTILITIES (for new BVR NPY+JSON output)
# ----------------------------------------------------------------------

# Regex to extract simulation time without json.load
# Matches e.g.:  "simulation time": 0.010000
_SIM_TIME_RE = re.compile(
    r'"simulation time"\s*:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'
)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=4096)
def get_simulation_time_from_json_fast(json_path: str) -> float:
    """
    Fast path: extract 'simulation time' from BVR JSON header using regex
    (avoids full json.load). Cached by json_path.
    """
    txt = _read_text(json_path)
    m = _SIM_TIME_RE.search(txt)
    if not m:
        raise KeyError(f"[ERROR] 'simulation time' not found in {json_path}")
    return float(m.group(1))


@lru_cache(maxsize=256)
def get_columns_from_json_cached(json_path: str) -> Tuple[str, ...]:
    """
    Read column names from BVR JSON header.

    Columns are identical across snapshots, so we parse once via json.load
    and cache the result. Returned as a tuple for cache safety.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cols = meta.get("columns", None)
    if not cols:
        raise KeyError(f"[ERROR] 'columns' not found or empty in {json_path}")

    return tuple(c["name"] for c in cols)


@lru_cache(maxsize=256)
def get_units_from_json_cached(json_path: str) -> Dict[str, str]:
    """
    Optional helper: cache units mapping (column name -> unit).

    Useful if units are needed later for plotting or metadata propagation,
    without repeatedly reading JSON files.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cols = meta.get("columns", None)
    if not cols:
        raise KeyError(f"[ERROR] 'columns' not found or empty in {json_path}")

    return {c["name"]: c.get("unit", "") for c in cols}


# ----------------------------------------------------------------------
# LEGACY ASCII FILE PARSING (kept for backward compatibility)
# ----------------------------------------------------------------------

def parse_file_metadata_txt(filename: str) -> Tuple[List[str], int]:
    """
    Legacy parser for Barracuda ASCII output (cell*.txt / particle*.txt).

    Returns:
      column_names : list of column names from '#@' header lines
      data_start_idx : first line index where numeric data begins
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_start_idx = None
    for idx, line in enumerate(lines):
        s = line.strip()
        if s and not s.startswith("#"):
            data_start_idx = idx
            break

    if data_start_idx is None:
        raise ValueError(f"[ERROR] No data found in {filename}")

    column_names: List[str] = []
    for line in lines[:data_start_idx]:
        s = line.lstrip()
        if s.startswith("#@"):
            parts = s.split('"')
            if len(parts) >= 2:
                column_names.append(parts[1])

    if not column_names:
        raise ValueError(f"[ERROR] No '#@' column definitions found in {filename}")

    return column_names, data_start_idx


def extract_truncated_time_from_filename(filename: str) -> float:
    """
    Legacy helper: extract time from filenames like:
      cell_0.005s.txt
      particles_0.010s.txt

    Kept for backward compatibility with older pipelines.
    """
    base = filename.split("/")[-1]
    m = re.search(r"_(\d+\.\d{3})s\.txt$", base)
    if not m:
        raise ValueError(f"[ERROR] Could not extract time from filename: {filename}")
    return float(m.group(1))
