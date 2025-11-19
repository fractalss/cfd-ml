# cpfd_rom/ml_rom/rom_lagrangian_ml/evaluation.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata

__all__ = ["write_lagrangian_rom_only"]


def _load_columns_from_dir(columns_dir: Path) -> list[str]:
    """
    Load column names from columns.txt in a Rev*_npy directory.
    """
    columns_dir = Path(columns_dir)
    columns_file = columns_dir / "columns.txt"
    if not columns_file.exists():
        raise FileNotFoundError(f"[Lagrangian] columns.txt not found in {columns_dir}")

    with open(columns_file, "r") as f:
        columns = [line.strip() for line in f if line.strip()]

    if not columns:
        raise ValueError(f"[Lagrangian] columns.txt in {columns_dir} is empty")

    return columns


def write_lagrangian_rom_only(
    preds: np.ndarray,
    times: np.ndarray | Sequence[float],
    columns_dir: Path,
    out_dir: Path = Path("."),
    zone_name: str = "Particles",
) -> None:
    """
    Write Lagrangian ROM-only snapshots to Tecplot-style particles*.txt files.

    Uses columns.txt from `columns_dir` as the single source of truth for
    feature ordering and names, and uses config.field_variable to decide which
    scalar to output along with x, y, z.

    Assumes preds have the same feature ordering as the numeric columns that
    were stored to .npy by the ASCII?NPY converter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = np.asarray(preds, dtype=np.float64)
    times = np.asarray(times, dtype=float)

    if preds.ndim != 3:
        raise ValueError(f"preds must be 3D (S, P, F); got shape {preds.shape}")
    S, P, F = preds.shape

    if len(times) != S:
        raise ValueError(f"len(times) ({len(times)}) must equal preds.shape[0] ({S})")

    # Load authoritative column names from columns.txt
    columns = _load_columns_from_dir(columns_dir)

    if len(columns) != F:
        raise ValueError(
            f"[Lagrangian] columns.txt has {len(columns)} entries but preds have F={F}. "
            "Check that .npy feature ordering matches columns.txt."
        )

    # Resolve which scalar field to output
    field_var = getattr(config, "field_variable", None)
    if field_var is None:
        raise ValueError(
            "[Lagrangian] config.field_variable must be set to choose the ROM scalar."
        )

    # We want x, y, z, and field_var
    required = ["x", "y", "z", field_var]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"[Lagrangian] Missing required columns in columns.txt: {missing}"
        )

    # Index mapping
    idx_x = columns.index("x")
    idx_y = columns.index("y")
    idx_z = columns.index("z")
    idx_f = columns.index(field_var)

    header_cols = ["x", "y", "z", field_var]
    header_md = [format_metadata(i + 1, name) for i, name in enumerate(header_cols)]

    for s, t in enumerate(
        tqdm(times, desc="Writing Lagrangian ROM snapshots", unit="snap")
    ):
        snap = preds[s]  # [P, F]
        if snap.shape != (P, F):
            snap = snap.reshape(P, F)

        # Extract only the 4 columns we want
        data_out = np.stack(
            [snap[:, idx_x], snap[:, idx_y], snap[:, idx_z], snap[:, idx_f]], axis=1
        )
        df_out = pd.DataFrame(data_out, columns=header_cols)

        out_path = out_dir / f"particles_{float(t):09.3f}s.txt"
        with open(out_path, "w") as f:
            f.write(f'# Zone name = "{zone_name}"\n')
            f.write(f"# Solution time = {float(t):.6f} s\n")
            for line in header_md:
                f.write(line)
            df_out.to_csv(
                f,
                sep="\t",
                header=False,
                index=False,
                float_format="%.6e",
            )
