# cpfd_rom/ml_rom/rom_lagrangian_ml/evaluation.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from cpfd_rom.util.output_utils import format_metadata

__all__ = ["write_lagrangian_rom_only"]


def _load_columns_from_dir(columns_dir: Path) -> list[str]:
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
    field_var: str | None = None,
    train_min: np.ndarray | None = None,
    train_max: np.ndarray | None = None,
) -> None:
    """Write Lagrangian ROM snapshots to Tecplot-style particles*.txt files.

    This is aligned with the current Lagrangian ROM logic, where each
    per-point prediction has **6 features** in the ROM feature space:

        [x, y, z, field, CloudID, CloudID_base]

    The original CFD / npy data may have more columns (e.g., 11), but
    the pipeline has already selected and reassembled these 6 columns
    into `preds` in the above order. We therefore:

         trust `preds` as [S, P, 6]
         use columns.txt only to check that the field variable name
          exists in the original schema
         write out reordered columns as: x, y, z, CloudID, CloudID_base, field
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

    if F != 6:
        raise ValueError(
            f"[Lagrangian] ROM writer expects preds with 6 features (x,y,z,field,CloudID,CloudID_base), got F={F}."
        )

    if field_var is None:
        raise ValueError(
            "[Lagrangian] field_var must be provided to write_lagrangian_rom_only."
        )

    columns = _load_columns_from_dir(columns_dir)
    if field_var not in columns:
        raise ValueError(
            f"[Lagrangian] field_variable='{field_var}' not found in columns.txt. "
            f"Available columns: {columns}"
        )

    header_cols = ["x", "y", "z", "CloudID", "CloudID_base", field_var]
    header_md = [format_metadata(i + 1, name) for i, name in enumerate(header_cols)]

    if train_min is not None and train_max is not None:
        preds[..., :4] = np.clip(preds[..., :4], train_min[:4], train_max[:4])

    for s, t in enumerate(
        tqdm(times, desc="Writing Lagrangian ROM snapshots", unit="snap")
    ):
        snap = preds[s]  # [P, 6]
        if snap.shape != (P, F):
            snap = snap.reshape(P, F)

        # Rearrange columns to: x, y, z, CloudID, CloudID_base, field
        reordered = np.stack(
            [
                snap[:, 0],  # x
                snap[:, 1],  # y
                snap[:, 2],  # z
                snap[:, 4],  # CloudID
                snap[:, 5],  # CloudID_base
                snap[:, 3],  # field
            ],
            axis=-1,
        )

        df_out = pd.DataFrame(reordered, columns=header_cols)

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