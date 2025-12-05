# cpfd_rom/ml_rom/rom_lagrangian_ml/evaluation.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from cpfd_rom.util.output_utils import format_metadata

__all__ = ["write_lagrangian_rom_only"]


def _load_columns_from_dir(columns_dir: Path) -> list[str]:
    """Read columns.txt from a Rev*_npy directory and return the column names."""
    columns_dir = Path(columns_dir)
    columns_file = columns_dir / "columns.txt"
    if not columns_file.exists():
        raise FileNotFoundError(f"[Lagrangian] columns.txt not found in {columns_dir}")

    with open(columns_file, "r") as f:
        columns = [line.strip() for line in f if line.strip()]

    if not columns:
        raise ValueError(f"[Lagrangian] columns.txt in {columns_dir} is empty")

    return columns


def _format_particle_row(vals: np.ndarray) -> str:
    """
    Format a single particle row in Tecplot/Barracuda style:

      - Leading space before the first value
      - Each value as ' {val: .6e}' (sign column: space for +, '-' for -)
      - Two spaces between columns
    """
    return " " + "  ".join(f"{v: .6e}" for v in vals)


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
    """Write Lagrangian ROM snapshots to Tecplot-style particles_*.txt files.

    ROM feature order in `preds` is:

        [x, y, z, field, CloudID, CloudID_base]

    We write out columns in this order:

        x, y, z, CloudID, CloudID_base, field

    and snap the header / filename times to a uniform grid to avoid
    16.200001 vs 16.200000 mismatches with the cells writer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = np.asarray(preds, dtype=np.float64)
    times = np.asarray(times, dtype=float)

    # ----------------- shape checks -----------------
    if preds.ndim != 3:
        raise ValueError(f"preds must be 3D (S, P, F); got shape {preds.shape}")
    S, P, F = preds.shape

    if len(times) != S:
        raise ValueError(f"len(times) ({len(times)}) must equal preds.shape[0] ({S})")

    if F != 6:
        raise ValueError(
            f"[Lagrangian] ROM writer expects preds with 6 features "
            f"(x,y,z,field,CloudID,CloudID_base), got F={F}."
        )

    # ----------------- time snapping grid -----------------
    # Build a snapped time grid for headers/filenames to kill float noise.
    t0 = float(times[0])
    if S > 1:
        # estimate uniform dt from the diffs (robust to tiny noise)
        dt_est = float(np.median(np.diff(times)))
    else:
        dt_est = 0.0

    def snap_time(t_raw: float) -> float:
        """Snap raw time to nearest multiple of dt_est, then to 0.001s grid."""
        if S <= 1 or dt_est <= 0.0:
            # single snapshot: just round to 0.001s
            return round(float(t_raw), 3)
        k = round((t_raw - t0) / dt_est)
        t_snap = t0 + k * dt_est
        # hard round to 0.001s so headers become 16.200000, 16.300000, ...
        return round(t_snap, 3)

    # ----------------- columns / header metadata -----------------
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

    # Header column names in the order we will write
    header_cols = ["x", "y", "z", "Cloud Id", "Cloud Id base", field_var]
    header_md = [format_metadata(i + 1, name) for i, name in enumerate(header_cols)]

    # Optional clipping in physical space for first 4 channels (x,y,z,field)
    if train_min is not None and train_max is not None:
        preds[..., :4] = np.clip(preds[..., :4], train_min[:4], train_max[:4])

    # ----------------- write each snapshot -----------------
    for s, t in enumerate(
        tqdm(times, desc="Writing Lagrangian ROM snapshots", unit="snap")
    ):
        snap = preds[s]  # [P, 6]
        if snap.shape != (P, F):
            snap = snap.reshape(P, F)

        # Rearrange to: x, y, z, CloudID, CloudID_base, field
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
        )  # [P, 6]

        # use snapped time for filenames + header to match cells exactly
        t_raw = float(t)
        t_hdr = snap_time(t_raw)  # e.g. 16.2, 16.3, ...
        # filename uses 3 decimals, header uses 6 (with trailing zeros)
        out_path = out_dir / f"particles_{t_hdr:09.3f}s.txt"
        with open(out_path, "w") as f:
            f.write(f'# Zone name = "{zone_name}"\n')
            f.write(f"# Solution time = {t_hdr:.6f} s\n")
            for line in header_md:
                f.write(line)

            # Write each particle row with Tecplot-style formatting
            for p in range(P):
                vals = reordered[p, :]
                line = _format_particle_row(vals)
                f.write(line + "\n")
