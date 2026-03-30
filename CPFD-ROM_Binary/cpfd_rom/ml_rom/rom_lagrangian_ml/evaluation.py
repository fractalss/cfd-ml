# evaluation.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata


TimeMode = Literal["raw", "canonical"]


def _format_particle_row(vals: np.ndarray) -> str:
    """
    Format a single particle row in Tecplot/Barracuda style.

    Output style:
      - scientific notation
      - tab-separated columns
    """
    return "\t".join(f"{v: .6e}" for v in vals)


def _as_list_of_arrays(
    preds: Union[np.ndarray, Sequence[np.ndarray]]
) -> list[np.ndarray]:
    """
    Normalize preds to a list of arrays, each shaped [N_i, 5].

    Expected input columns per row:
        [x, y, z, Cloud Id, field]

    The writer outputs 6 columns by inserting:
        Cloud Id base = 0
    """
    if isinstance(preds, np.ndarray):
        if preds.ndim != 3 or preds.shape[2] != 5:
            raise ValueError(f"preds ndarray must be [S, N, 5], got {preds.shape}")
        return [preds[s] for s in range(preds.shape[0])]

    out: list[np.ndarray] = []
    for i, arr in enumerate(preds):
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(f"preds[{i}] must be [N_i, 5], got {arr.shape}")
        out.append(arr)
    return out


def _resolve_field_name(field_variable: Optional[str]) -> str:
    field_name = field_variable or getattr(config, "field_variable", None)
    if field_name is None:
        raise ValueError(
            "[Lagrangian/Raw] field_variable must be provided to write_lagrangian_rom_only."
        )
    return field_name


def _estimate_dt(times: np.ndarray) -> Optional[float]:
    """
    Estimate a representative dt from times using median diff.
    Returns None if estimation is not meaningful.
    """
    if times.size <= 1:
        return None

    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs)]

    if diffs.size == 0:
        return None

    # Guard against repeated or degenerate times
    positive_diffs = diffs[np.abs(diffs) > 0.0]
    if positive_diffs.size == 0:
        return None

    dt_est = float(np.median(positive_diffs))
    if not np.isfinite(dt_est) or dt_est <= 0.0:
        return None

    return dt_est


def _canonicalize_time(
    t_raw: float,
    *,
    t0: float,
    dt: float,
    decimals: int = 3,
) -> float:
    """
    Snap raw time to nearest point on a regular grid defined by (t0, dt).
    """
    k = round((t_raw - t0) / dt)
    t_snap = t0 + k * dt
    return round(float(t_snap), decimals)


def _format_output_time(
    t_raw: float,
    *,
    time_mode: TimeMode,
    decimals: int,
    t0: Optional[float] = None,
    dt: Optional[float] = None,
) -> float:
    """
    Convert raw time to output time according to the requested mode.

    Modes
    -----
    raw:
        Preserve the incoming time, only round for file/header formatting.
    canonical:
        Snap to a regular grid using (t0, dt), then round for formatting.
    """
    if time_mode == "raw":
        return round(float(t_raw), decimals)

    if time_mode == "canonical":
        if t0 is None or dt is None or dt <= 0.0:
            raise ValueError(
                "time_mode='canonical' requires valid t0 and dt."
            )
        return _canonicalize_time(t_raw, t0=t0, dt=dt, decimals=decimals)

    raise ValueError(f"Unsupported time_mode: {time_mode}")


def write_lagrangian_rom_only(
    preds: Union[np.ndarray, Sequence[np.ndarray]],
    times: Sequence[float],
    field_variable: Optional[str] = None,
    out_dir: Path | str = ".",
    zone_name: str = "Particles",
    *,
    time_mode: TimeMode = "raw",
    dt_canonical: Optional[float] = None,
    time_decimals_filename: int = 3,
    time_decimals_header: int = 6,
    filename_width: int = 9,
    log_time_mapping: bool = False,
) -> None:
    """
    Write Lagrangian ROM snapshots to Tecplot-style TXT files.

    Input row format
    ----------------
    Each input row must be:
        [x, y, z, Cloud Id, field]

    Output row format
    -----------------
    Each output row written to file becomes:
        [x, y, z, Cloud Id, Cloud Id base, field]

    where:
        Cloud Id base = 0.0

    Supported input forms
    ---------------------
    - preds as np.ndarray [S, N, 5]
    - preds as list[np.ndarray], each [N_i, 5]

    Time handling
    -------------
    time_mode="raw":
        Preserve input times. This is the recommended default and avoids
        Eulerian/Lagrangian mismatch when the Eulerian side preserves raw times.

    time_mode="canonical":
        Snap times to a regular grid. Uses dt_canonical if provided, otherwise
        estimates dt from median diff of input times.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_name = _resolve_field_name(field_variable)

    times_arr = np.asarray(times, dtype=np.float64)
    preds_list = _as_list_of_arrays(preds)

    n_snapshots = len(preds_list)
    if len(times_arr) != n_snapshots:
        raise ValueError(
            f"Number of times ({len(times_arr)}) must match number of snapshots ({n_snapshots})."
        )

    if n_snapshots == 0:
        return

    # Time-grid parameters for canonical mode
    t0 = float(times_arr[0])
    dt = dt_canonical if dt_canonical is not None else _estimate_dt(times_arr)

    if time_mode == "canonical" and (dt is None or dt <= 0.0):
        raise ValueError(
            "Could not determine canonical dt. Provide dt_canonical explicitly "
            "or use time_mode='raw'."
        )

    # Header metadata
    header_cols = ["x", "y", "z", "Cloud Id", "Cloud Id base", field_name]
    header_md = [format_metadata(i + 1, col) for i, col in enumerate(header_cols)]

    for s, t_raw in enumerate(tqdm(times_arr, desc="Writing ROM snapshots", unit="snap")):
        snapshot = np.asarray(preds_list[s], dtype=np.float64)

        if snapshot.ndim != 2 or snapshot.shape[1] != 5:
            raise ValueError(
                f"[Lagrangian/Raw] Snapshot {s} must be [N_i, 5], got {snapshot.shape}"
            )

        # Input:  [x, y, z, Cloud Id, field]
        # Output: [x, y, z, Cloud Id, Cloud Id base, field]
        n_particles = snapshot.shape[0]
        out_snapshot = np.empty((n_particles, 6), dtype=np.float64)
        out_snapshot[:, 0:4] = snapshot[:, 0:4]
        out_snapshot[:, 4] = 0.0
        out_snapshot[:, 5] = snapshot[:, 4]

        t_out = _format_output_time(
            float(t_raw),
            time_mode=time_mode,
            decimals=time_decimals_filename,
            t0=t0,
            dt=dt,
        )

        if log_time_mapping:
            print(
                f"[Lagrangian Writer] snap={s} raw_time={float(t_raw)!r} "
                f"output_time={t_out!r} time_mode={time_mode!r}"
            )

        out_path = out_dir / f"particles_{t_out:0{filename_width}.{time_decimals_filename}f}s.txt"

        with open(out_path, "w") as f:
            f.write(f'# Zone name = "{zone_name}"\n')
            f.write(f"# Solution time = {t_out:.{time_decimals_header}f} s\n")
            for line in header_md:
                f.write(line)

            for p in range(out_snapshot.shape[0]):
                f.write(_format_particle_row(out_snapshot[p, :]) + "\n")