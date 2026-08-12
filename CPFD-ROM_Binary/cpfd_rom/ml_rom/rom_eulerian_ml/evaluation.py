"""Writers for Eulerian ROM output snapshots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from cpfd_rom.util import config
from cpfd_rom.util.logging_config import detail, progress_enabled
from cpfd_rom.util.output_utils import format_metadata


logger = logging.getLogger(__name__)

__all__ = ["_write_rom_only"]


def _write_rom_only(
    preds: np.ndarray,
    times: np.ndarray | Sequence[float],
    nodes_df: pd.DataFrame,
    field_name: Optional[str] = None,
    out_dir: Path = Path("."),
) -> None:
    """Write ROM-only snapshots to Tecplot-style text files.

    Parameters
    ----------
    preds:
        Array of shape ``(S, N)`` containing predicted field values in
        ``node_id`` order.
    times:
        Sequence of length ``S`` containing solution times in seconds.
    nodes_df:
        DataFrame containing ``node_id, x, y, z, i, j, k`` in canonical node
        order.
    field_name:
        Optional field-name override. The configured ``field_variable`` takes
        precedence when it is available.
    out_dir:
        Directory in which the snapshot files will be written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    field_name_eff = (
        getattr(config, "field_variable", None) or field_name or "field"
    )

    required_columns = {"node_id", "x", "y", "z", "i", "j", "k"}
    missing_columns = required_columns.difference(nodes_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"nodes_df is missing required columns: {missing}")

    header_cols = ["x", "y", "z", "i", "j", "k", field_name_eff]
    header_md = [
        format_metadata(index + 1, column)
        for index, column in enumerate(header_cols)
    ]

    key = (
        nodes_df[["node_id", "x", "y", "z", "i", "j", "k"]]
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    times_array = np.asarray(times, dtype=float)
    preds_array = np.asarray(preds, dtype=np.float64)

    if times_array.ndim != 1:
        raise ValueError(
            f"times must be 1D; got shape {times_array.shape}"
        )
    if preds_array.ndim != 2:
        raise ValueError(
            f"preds must be 2D (S, N); got shape {preds_array.shape}"
        )
    if len(times_array) != preds_array.shape[0]:
        raise ValueError(
            f"len(times) ({len(times_array)}) must equal preds.shape[0] "
            f"({preds_array.shape[0]})"
        )
    if preds_array.shape[1] != key.shape[0]:
        raise ValueError(
            f"N mismatch: preds has N={preds_array.shape[1]} versus "
            f"nodes_df rows {key.shape[0]} (check node_id order)"
        )

    logger.info("Writing Eulerian ROM output snapshots")
    detail(
        logger,
        "Writing %d snapshots for field %s to %s",
        len(times_array),
        field_name_eff,
        out_dir,
    )

    for snapshot_index, solution_time in enumerate(
        tqdm(
            times_array,
            desc="Writing ROM snapshots",
            unit="snap",
            disable=not progress_enabled(),
        )
    ):
        values = preds_array[snapshot_index]
        if values.ndim != 1:
            values = values.reshape(-1)

        output_df = key.copy()
        output_df.loc[:, field_name_eff] = values
        output_df = output_df.sort_values(
            by=["k", "j", "i"],
            kind="mergesort",
        )

        output_path = out_dir / f"cells_{float(solution_time):09.3f}s.txt"
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            stream.write('# Zone name = "Cells"\n')
            stream.write(
                f"# Solution time = {float(solution_time):.6f} s\n"
            )
            for line in header_md:
                stream.write(line)
            output_df[
                ["x", "y", "z", "i", "j", "k", field_name_eff]
            ].to_csv(
                stream,
                sep="\t",
                header=False,
                index=False,
                float_format="%.6e",
            )

        logger.debug("Wrote ROM snapshot %s", output_path)

    logger.info("Eulerian ROM output saved to %s", out_dir)
