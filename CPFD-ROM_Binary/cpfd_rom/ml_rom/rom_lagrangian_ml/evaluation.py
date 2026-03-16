# evaluation.py (Option 1: strict 6-column format with "Cloud Id base" = dummy 0)
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata


def _format_particle_row(vals: np.ndarray) -> str:
    """
    Format a single particle row in Tecplot/Barracuda style.

      - Leading space before first value
      - Each value as ' {val: .6e}'
      - Two spaces between columns
    """
    return "\t".join(f"{v: .6e}" for v in vals)


def _as_list_of_arrays(
    preds: Union[np.ndarray, Sequence[np.ndarray]]
) -> list[np.ndarray]:
    """
    Normalize preds to list-of-arrays, each [N_i, 5].

    The writer takes 5-column inputs:
        [x, y, z, Cloud Id, field]

    and will STRICTLY output 6 columns by inserting:
        Cloud Id base = 0

    Accepts:
      - np.ndarray [S, N, 5]
      - list/tuple of np.ndarray each [N_i, 5]
    """
    if isinstance(preds, np.ndarray):
        if preds.ndim != 3 or preds.shape[2] != 5:
            raise ValueError(f"preds ndarray must be [S,N,5], got {preds.shape}")
        return [preds[s] for s in range(preds.shape[0])]

    out: list[np.ndarray] = []
    for i, a in enumerate(preds):
        a = np.asarray(a)
        if a.ndim != 2 or a.shape[1] != 5:
            raise ValueError(f"preds[{i}] must be [N_i,5], got {a.shape}")
        out.append(a)
    return out


def write_lagrangian_rom_only(
    preds: Union[np.ndarray, Sequence[np.ndarray]],
    times: Sequence[float],
    field_variable: Optional[str] = None,
    out_dir: Path | str = ".",
    zone_name: str = "Particles",
) -> None:
    """
    Write Lagrangian ROM snapshots to Tecplot-style TXT files.

    Input rows (preds) are expected as:
        [x, y, z, Cloud Id, field]    (5 columns)

    Output rows written to file are:
        [x, y, z, Cloud Id, Cloud Id base, field]   (6 columns)
    where:
        Cloud Id base = 0 (dummy constant)

    Supports:
      - preds as np.ndarray [S, N, 5]
      - preds as list[np.ndarray], each [N_i, 5]

    Time snapping:
      - snap to nearest multiple of dt_est
      - then round to 0.001 s for filenames
      - header uses 6 decimals
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_name = field_variable or getattr(config, "field_variable", None)
    if field_name is None:
        raise ValueError(
            "[Lagrangian/Raw] field_variable must be provided to write_lagrangian_rom_only."
        )

    times = np.asarray(times, dtype=np.float64)
    preds_list = _as_list_of_arrays(preds)

    S = len(preds_list)
    if len(times) != S:
        raise ValueError(
            f"Number of times ({len(times)}) must match number of snapshots ({S})"
        )

    # ----------------- time snapping grid -----------------
    t0 = float(times[0])
    if S > 1:
        dt_est = float(np.median(np.diff(times)))
    else:
        dt_est = 0.0

    def snap_time(t_raw: float) -> float:
        if S <= 1 or dt_est <= 0.0:
            return round(float(t_raw), 3)
        k = round((t_raw - t0) / dt_est)
        t_snap = t0 + k * dt_est
        return round(t_snap, 3)

    # ----------------- header metadata -----------------
    header_cols = ["x", "y", "z", "Cloud Id", "Cloud Id base", field_name]
    header_md = [format_metadata(i + 1, col) for i, col in enumerate(header_cols)]

    # ----------------- write snapshots -----------------
    for s, t in enumerate(tqdm(times, desc="Writing ROM snapshots", unit="snap")):
        snapshot = np.asarray(preds_list[s], dtype=np.float64)

        if snapshot.ndim != 2 or snapshot.shape[1] != 5:
            raise ValueError(
                f"[Lagrangian/Raw] Snapshot {s} must be [N_i,5], got {snapshot.shape}"
            )

        # Build output with 6 columns by inserting Cloud Id base = 0 before field
        # snapshot columns: [x, y, z, Cloud Id, field]
        # output columns  : [x, y, z, Cloud Id, Cloud Id base, field]
        N = snapshot.shape[0]
        out_snapshot = np.empty((N, 6), dtype=np.float64)
        out_snapshot[:, 0:4] = snapshot[:, 0:4]   # x, y, z, Cloud Id
        out_snapshot[:, 4] = 0.0                  # Cloud Id base (dummy)
        out_snapshot[:, 5] = snapshot[:, 4]       # field

        t_hdr = snap_time(float(t))
        out_path = out_dir / f"particles_{t_hdr:09.3f}s.txt"

        with open(out_path, "w") as f:
            f.write(f'# Zone name = "{zone_name}"\n')
            f.write(f"# Solution time = {t_hdr:.6f} s\n")
            for line in header_md:
                f.write(line)

            for p in range(out_snapshot.shape[0]):
                f.write(_format_particle_row(out_snapshot[p, :]) + "\n")