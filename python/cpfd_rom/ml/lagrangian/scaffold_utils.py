"""Utilities for selecting Lagrangian scaffold graphs."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch_geometric.data import Data

from cpfd_rom.util.logging_config import detail

from .data_loader import load_lagrangian_snapshots_as_graphs


logger = logging.getLogger(__name__)


def get_scaffold_graphs_for_param(
    user_param: float,
    param_mapping: Mapping[str, float],
    rev_dirs: Sequence[str],
    base_data_dir: str,
    field_variable: str = "Particle volume fraction",
    radius: float = 0.01,
    sample_ratio: float = 1.0,
    feature_stats: Mapping[str, Any] | None = None,
) -> list[Data]:
    """Load scaffold graphs from the available revision nearest ``user_param``.

    Only revisions present in both ``rev_dirs`` and ``param_mapping`` are
    considered. The selected revision is passed to the standard Lagrangian
    graph loader without changing its sampling or normalization behavior.
    """
    if not param_mapping:
        raise ValueError("param_mapping is required.")
    if not rev_dirs:
        raise ValueError("rev_dirs must contain at least one revision directory.")

    candidate_revs = [rev for rev in rev_dirs if rev in param_mapping]
    if not candidate_revs:
        raise ValueError(
            "None of the requested rev_dirs are present in param_mapping."
        )

    rev_params = np.asarray(
        [param_mapping[rev] for rev in candidate_revs], dtype=np.float64
    )
    if not np.all(np.isfinite(rev_params)):
        raise ValueError("param_mapping values must be finite numbers.")
    if not np.isfinite(user_param):
        raise ValueError("user_param must be a finite number.")

    closest_index = int(np.argmin(np.abs(rev_params - float(user_param))))
    closest_rev = candidate_revs[closest_index]
    closest_param = float(rev_params[closest_index])

    detail(
        logger,
        "[Scaffold] Closest revision to param=%.4f is '%s' with param=%.4f",
        user_param,
        closest_rev,
        closest_param,
    )

    return load_lagrangian_snapshots_as_graphs(
        rev_dirs=[closest_rev],
        base_data_dir=base_data_dir,
        param_mapping=param_mapping,
        field_variable=field_variable,
        radius=radius,
        sample_ratio=sample_ratio,
        feature_stats=feature_stats,
    )


__all__ = ["get_scaffold_graphs_for_param"]
