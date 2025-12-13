# cpfd_rom/ml_rom/rom_lagrangian_ml/scaffold_utils.py

import os
import numpy as np
from typing import List, Dict
from torch_geometric.data import Data

from .data_loader import load_lagrangian_snapshots_as_graphs

def get_scaffold_graphs_for_param(
    user_param: float,
    param_mapping: Dict[str, float],
    rev_dirs: List[str],
    base_data_dir: str,
    field_variable: str = "Particle volume fraction",
    radius: float = 0.01,
    sample_ratio: float = 1.0,
    feature_stats: Dict = None,
) -> List[Data]:
    """
    Selects the closest Rev directory based on user_param and loads its graphs as templates.

    Args:
        user_param: The user's physical parameter (float).
        param_mapping: Dict mapping rev_dir names (e.g. 'Rev1') to physical parameters.
        rev_dirs: List of rev_dir names to search.
        base_data_dir: Path to base folder containing rev_dir subfolders.
        field_variable: Name of field variable (optional).
        radius: Radius used to build edge connections in graphs.
        sample_ratio: Fraction of snapshots to sample (e.g. 1.0 = all).
        feature_stats: Optional normalization stats (used if graphs need to be normalized).

    Returns:
        List[Data]: Graphs from the closest rev_dir to the user_param.
    """
    if not param_mapping:
        raise ValueError("param_mapping is required.")

    # Match closest Rev
    rev_list = list(param_mapping.keys())
    rev_params = np.array([param_mapping[r] for r in rev_list])
    idx_closest = np.argmin(np.abs(rev_params - user_param))
    rev_closest = rev_list[idx_closest]
    print(f"[Scaffold] Closest Rev to param={user_param:.4f} is '{rev_closest}' with param={rev_params[idx_closest]:.4f}")

    # Load graphs only from this Rev
    graphs = load_lagrangian_snapshots_as_graphs(
        rev_dirs=[rev_closest],
        base_data_dir=base_data_dir,
        param_mapping=param_mapping,
        field_variable=field_variable,
        radius=radius,
        sample_ratio=sample_ratio,
        feature_stats=feature_stats,
    )

    return graphs
