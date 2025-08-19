# Saurav Mitra
# evaluation.py  Writers for ROM outputs
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata

__all__ = ["_write_rom_only"]


def _write_rom_only(
    preds: np.ndarray,
    times: np.ndarray | Sequence[float],
    nodes_df: pd.DataFrame,
    field_name: Optional[str] = None,   # optional; prefer config.field_variable
    out_dir: Path = Path("."),
) -> None:
    """Write ROM-only snapshots to Tecplot-style TXT files.

    Args:
        preds: Array of shape (S, N) with predicted field values in **node_id** order.
        times: Sequence/array of length S with solution times (seconds).
        nodes_df: DataFrame with columns [node_id, x, y, z, i, j, k] in canonical node order.
        field_name: Optional override; if None, uses config.field_variable.
        out_dir: Output directory where files will be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer global config; fall back to provided arg, then a generic name
    field_name_eff = (getattr(config, "field_variable", None) or field_name or "field")

    # Header metadata
    header_cols = ["x", "y", "z", "i", "j", "k", field_name_eff]
    header_md = [format_metadata(i + 1, c) for i, c in enumerate(header_cols)]

    # Canonical key table (one row per node)
    key = (
        nodes_df[["node_id", "x", "y", "z", "i", "j", "k"]]
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    times = np.asarray(times, dtype=float)
    preds = np.asarray(preds, dtype=np.float64)

    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D (S, N); got shape {preds.shape}")
    if len(times) != preds.shape[0]:
        raise ValueError(f"len(times) ({len(times)}) must equal preds.shape[0] ({preds.shape[0]})")
    if preds.shape[1] != key.shape[0]:
        raise ValueError(f"N mismatch: preds has N={preds.shape[1]} vs nodes_df rows {key.shape[0]} (check node_id order)")

    for s, t in enumerate(tqdm(times, desc="Writing ROM snapshots", unit="snap")):
        y = preds[s]
        if y.ndim != 1:
            y = y.reshape(-1)

        df = key.copy()
        df[field_name_eff] = y
        # Sort to i-fastest (k outermost) order for output
        df = df.sort_values(by=["k", "j", "i"], kind="mergesort")

        out_path = out_dir / f"cells_{float(t):09.3f}s.txt"
        with open(out_path, "w") as f:
            f.write('# Zone name = "Cells"\n')
            f.write(f"# Solution time = {float(t):.6f} s\n")
            for line in header_md:
                f.write(line)
            df[["x", "y", "z", "i", "j", "k", field_name_eff]].to_csv(
                f, sep="\t", header=False, index=False, float_format="%.6e"
            )
