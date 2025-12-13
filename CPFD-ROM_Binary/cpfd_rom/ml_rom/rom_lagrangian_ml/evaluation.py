from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from cpfd_rom.util import config
from cpfd_rom.util.output_utils import format_metadata

def write_lagrangian_rom_only(
    preds: np.ndarray,
    times: Sequence[float],
    field_variable: Optional[str] = None,
    out_dir: Path | str = "."
) -> None:
    """
    Write Lagrangian ROM snapshots to Tecplot-style TXT files.

    Args:
        preds: Array of shape (S, P, 6) with each row as [x, y, z, Cloud Id, Cloud Id base, field].
        times: Sequence of length S with solution times (seconds).
        field_variable: Field variable name (e.g., 'Particle volume fraction').
        out_dir: Output directory where files will be written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_name = field_variable or getattr(config, "field_variable", None)
    if field_name is None:
        raise ValueError("[Lagrangian] field_variable must be provided to write_lagrangian_rom_only.")

    preds = np.asarray(preds, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    if preds.ndim != 3:
        raise ValueError(f"preds must be 3D (S, P, 6); got shape {preds.shape}")
    if preds.shape[2] != 6:
        raise ValueError(f"[Lagrangian] Each particle row must have 6 features (x, y, z, Cloud Id, Cloud Id base, field); got {preds.shape[2]}")
    if len(times) != preds.shape[0]:
        raise ValueError(f"Number of times ({len(times)}) must match number of snapshots ({preds.shape[0]})")

    header_cols = ["x", "y", "z", "Cloud Id", "Cloud Id base", field_name]
    header_md = [format_metadata(i + 1, col) for i, col in enumerate(header_cols)]

    for s, t in enumerate(tqdm(times, desc="Writing ROM snapshots", unit="snap")):
        snapshot = preds[s]  # shape (P, 6)
        df = pd.DataFrame(snapshot, columns=header_cols)

        out_path = out_dir / f"particles_{t:09.3f}s.txt"
        with open(out_path, "w") as f:
            f.write('# Zone name = "Particles"\n')
            f.write(f"# Solution time = {float(t):.6f} s\n")
            for line in header_md:
                f.write(line)
            df.to_csv(f, sep="\t", header=False, index=False, float_format="%.6e")
