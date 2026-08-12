from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from cpfd_rom.util import config
from cpfd_rom.util.logging_config import DETAIL_LEVEL
from cpfd_rom.util.output_utils import format_metadata


logger = logging.getLogger(__name__)

TimeMode = Literal["raw", "canonical"]


def _format_particle_row(vals: np.ndarray) -> str:
    """Format one particle row using scientific notation and tab separators."""
    return "\t".join(f"{v: .6e}" for v in vals)


def _as_list_of_arrays(
    preds: Union[np.ndarray, Sequence[np.ndarray]],
) -> list[np.ndarray]:
    """Normalize predictions to a list of arrays, each shaped ``[N_i, 5]``."""
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
            "[Lagrangian/Raw] field_variable must be provided to "
            "write_lagrangian_rom_only."
        )
    return field_name


def _estimate_dt(times: np.ndarray) -> Optional[float]:
    """Estimate a representative positive time step using the median difference."""
    if times.size <= 1:
        return None

    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None

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
    """Snap a raw time to the nearest point on the grid defined by ``(t0, dt)``."""
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
    """Convert a raw time to the requested output-time representation."""
    if time_mode == "raw":
        return round(float(t_raw), decimals)

    if time_mode == "canonical":
        if t0 is None or dt is None or dt <= 0.0:
            raise ValueError("time_mode='canonical' requires valid t0 and dt.")
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
    """Write Lagrangian ROM snapshots to Tecplot-style text files.

    Input rows are ``[x, y, z, Cloud Id, field]``. Output rows insert
    ``Cloud Id base = 0`` before the field column.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_name = _resolve_field_name(field_variable)
    times_arr = np.asarray(times, dtype=np.float64)
    preds_list = _as_list_of_arrays(preds)

    n_snapshots = len(preds_list)
    if len(times_arr) != n_snapshots:
        raise ValueError(
            f"Number of times ({len(times_arr)}) must match number of snapshots "
            f"({n_snapshots})."
        )

    if n_snapshots == 0:
        return

    t0 = float(times_arr[0])
    dt = dt_canonical if dt_canonical is not None else _estimate_dt(times_arr)
    if time_mode == "canonical" and (dt is None or dt <= 0.0):
        raise ValueError(
            "Could not determine canonical dt. Provide dt_canonical explicitly "
            "or use time_mode='raw'."
        )

    header_cols = ["x", "y", "z", "Cloud Id", "Cloud Id base", field_name]
    header_md = [format_metadata(i + 1, col) for i, col in enumerate(header_cols)]

    show_progress = logger.isEnabledFor(DETAIL_LEVEL)
    snapshots = tqdm(
        enumerate(times_arr),
        total=n_snapshots,
        desc="Writing ROM snapshots",
        unit="snap",
        disable=not show_progress,
    )
    for s, t_raw in snapshots:
        snapshot = np.asarray(preds_list[s], dtype=np.float64)
        if snapshot.ndim != 2 or snapshot.shape[1] != 5:
            raise ValueError(
                f"[Lagrangian/Raw] Snapshot {s} must be [N_i, 5], "
                f"got {snapshot.shape}"
            )

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

        if log_time_mapping and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[Lagrangian Writer] snap=%d raw_time=%r output_time=%r "
                "time_mode=%r",
                s,
                float(t_raw),
                t_out,
                time_mode,
            )

        out_path = out_dir / (
            f"particles_{t_out:0{filename_width}.{time_decimals_filename}f}s.txt"
        )
        with out_path.open("w", encoding="utf-8") as output_file:
            output_file.write(f'# Zone name = "{zone_name}"\n')
            output_file.write(
                f"# Solution time = {t_out:.{time_decimals_header}f} s\n"
            )
            output_file.writelines(header_md)
            for row in out_snapshot:
                output_file.write(_format_particle_row(row) + "\n")
