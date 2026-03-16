# cpfd_rom/ml_rom/rom_lagrangian_ml/data_loader.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from cpfd_rom.util.file_parsing import (
    get_simulation_time_from_json_fast,
    get_columns_from_json_cached,
)


# ---------------------------------------------------------------------
# Required JSON column names for raw-particle Lagrangian ROM
# ---------------------------------------------------------------------

RAW_REQUIRED_BASE = [
    "Cloud ID",
    "X position",
    "Y position",
    "Z position",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _list_raw_particle_jsons(dir_path: Path) -> list[Path]:
    """
    Return sorted list of Raw.particle.*.json files in a Rev directory.
    """
    files = sorted(dir_path.glob("Raw.particle.*.json"))
    return [f for f in files if f.is_file()]


def _matching_npy_from_json(json_path: Path) -> Path:
    """
    For:
        Raw.particle.00200_2.0000e+01.json
    return:
        Raw.particle.00200_2.0000e+01.npy
    """
    npy_path = json_path.with_suffix(".npy")
    if not npy_path.exists():
        raise FileNotFoundError(
            f"[Lagrangian/Raw] Matching NPY file not found for {json_path}"
        )
    return npy_path


def _get_json_columns(json_path: Path) -> list[str]:
    """
    Read column names from JSON in listed order.
    """
    return list(get_columns_from_json_cached(str(json_path)))


def _build_column_index(json_path: Path, field_variable: str) -> dict[str, int]:
    """
    Read JSON column names and return required column indices.
    """
    cols = _get_json_columns(json_path)
    required = RAW_REQUIRED_BASE + [field_variable]

    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"[Lagrangian/Raw] Missing required columns in {json_path.name}: {missing}\n"
            f"Available columns: {cols}"
        )

    return {name: cols.index(name) for name in required}


def _structured_to_dense_2d(arr: np.ndarray, columns: list[str]) -> np.ndarray:
    """
    Convert loaded NPY into dense [N, C] matrix following JSON column order.

    Supports:
      1) regular ndarray of shape [N, C]
      2) structured ndarray of shape [N] with named fields

    For structured arrays, JSON columns are the source of truth for column order.
    """
    # Plain dense matrix case
    if arr.ndim == 2:
        return arr

    # Structured/record array case
    if arr.ndim == 1 and arr.dtype.names is not None:
        field_names = arr.dtype.names
        if len(field_names) != len(columns):
            raise ValueError(
                f"[Lagrangian/Raw] Structured NPY field count ({len(field_names)}) does not "
                f"match JSON column count ({len(columns)}). "
                f"NPY fields: {field_names}, JSON columns: {columns}"
            )

        # Fast path: if exact order/labels match, use names directly
        if list(field_names) == list(columns):
            dense_cols = [np.asarray(arr[name]) for name in columns]
            return np.column_stack(dense_cols)

        # Fallback: try exact-name lookup for every JSON column
        missing = [c for c in columns if c not in field_names]
        if missing:
            raise ValueError(
                f"[Lagrangian/Raw] Structured NPY missing JSON-defined columns: {missing}\n"
                f"Structured fields: {field_names}\n"
                f"JSON columns: {columns}"
            )

        dense_cols = [np.asarray(arr[name]) for name in columns]
        return np.column_stack(dense_cols)

    raise ValueError(
        f"[Lagrangian/Raw] Unsupported NPY format in structured loader: "
        f"shape={arr.shape}, dtype={arr.dtype}"
    )


def _normalize_features(
    pos_raw: torch.Tensor,
    field_raw: torch.Tensor,
    feature_stats: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize [x,y,z] with mean/std and field with min/max.
    If feature_stats is None, return raw tensors.
    """
    if feature_stats is None:
        return pos_raw, field_raw

    eps = 1e-8
    pos_mean = feature_stats["pos_mean"]
    pos_std = feature_stats["pos_std"]
    fmin = float(feature_stats["field_min"])
    fmax = float(feature_stats["field_max"])

    pos_feat = (pos_raw - pos_mean) / (pos_std + eps)
    field_feat = (field_raw - fmin) / (fmax - fmin + eps)
    return pos_feat, field_feat


def _param_to_1d_array(param_val) -> np.ndarray:
    """
    Support scalar or vector param mapping.
    """
    return np.asarray(param_val, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------

def compute_feature_stats(graphs: list[Data]) -> dict[str, torch.Tensor | float]:
    """
    Compute normalization stats from UNNORMALIZED graphs.

    Assumptions:
      - g.pos is always physical xyz
      - g.x[:, 3:4] is physical field if stats are computed before normalization
    """
    if not graphs:
        raise ValueError("[Lagrangian/Raw] Cannot compute feature stats on empty graph list.")

    all_pos = torch.cat([g.pos for g in graphs], dim=0)          # [*, 3]
    all_field = torch.cat([g.x[:, 3:4] for g in graphs], dim=0)  # [*, 1]

    pos_std = all_pos.std(dim=0)
    pos_std = torch.where(pos_std > 0, pos_std, torch.ones_like(pos_std))

    return {
        "pos_mean": all_pos.mean(dim=0),
        "pos_std": pos_std,
        "field_min": all_field.min().item(),
        "field_max": all_field.max().item(),
    }


# ---------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------

def load_lagrangian_snapshots_as_graphs(
    rev_dirs,
    base_data_dir,
    param_mapping,
    field_variable="Particle volume fraction",
    radius=0.1,
    sample_ratio=1.0,
    feature_stats=None,
):
    """
    Load raw-particle Lagrangian snapshots as PyG graphs.

    Source of truth:
      - simulation time: JSON
      - column schema:   JSON
      - numeric values:  matching NPY

    Each graph contains:
      - data.x        : [N,4] = [x,y,z,field] (normalized if feature_stats given)
      - data.y        : [N,4] same as x for AE reconstruction
      - data.pos      : [N,3] physical xyz
      - data.edge_index
      - data.params   : [1,P_aug] = [physical params..., t_norm]
      - data.time     : [1]
      - data.cloud_id : [N]

    Notes:
      - No Cloud ID base in this raw format
      - Graph edges are always built from PHYSICAL xyz
      - Supports both plain dense NPY and structured NPY
    """
    all_graphs: list[Data] = []
    all_times: list[float] = []

    rev_paths = [Path(base_data_dir) / rev for rev in rev_dirs]

    # -------------------------------
    # First pass: gather all times
    # -------------------------------
    print("[DataLoader] Scanning raw-particle JSON files for time range...")

    rev_to_jsons: dict[str, list[Path]] = {}
    for rev_path in rev_paths:
        if not rev_path.exists():
            raise FileNotFoundError(f"[Lagrangian/Raw] Rev directory not found: {rev_path}")

        json_files = _list_raw_particle_jsons(rev_path)
        if not json_files:
            raise FileNotFoundError(
                f"[Lagrangian/Raw] No Raw.particle.*.json files found in {rev_path}"
            )

        rev_to_jsons[str(rev_path)] = json_files
        for json_path in json_files:
            tval = float(get_simulation_time_from_json_fast(str(json_path)))
            all_times.append(tval)

    if not all_times:
        raise RuntimeError("[Lagrangian/Raw] No simulation times found in raw-particle JSON files.")

    t_min = float(min(all_times))
    t_max = float(max(all_times))
    print(f"[DataLoader] Global time range: t_min={t_min:.6f}, t_max={t_max:.6f}")

    if t_max == t_min:
        raise ValueError("[Lagrangian/Raw] Zero time range  all snapshot times are identical.")

    # -------------------------------
    # Second pass: load selected files
    # -------------------------------
    print("[DataLoader] Loading raw-particle snapshots and constructing graphs...")

    for rev_path in rev_paths:
        rev_name = rev_path.name
        if rev_name not in param_mapping:
            raise KeyError(f"[Lagrangian/Raw] Missing param mapping for rev '{rev_name}'")

        param_val = _param_to_1d_array(param_mapping[rev_name])

        json_files = sorted(
            rev_to_jsons[str(rev_path)],
            key=lambda p: get_simulation_time_from_json_fast(str(p))
        )

        n_total = len(json_files)
        n_sample = max(1, int(sample_ratio * n_total))
        stride = max(1, n_total // n_sample)
        sampled_jsons = json_files[::stride]

        for json_path in sampled_jsons:
            npy_path = _matching_npy_from_json(json_path)
            tval = float(get_simulation_time_from_json_fast(str(json_path)))
            idx = _build_column_index(json_path, field_variable)

            print(f"[DataLoader] Loading: {npy_path}")
            arr = np.load(npy_path, allow_pickle=False)

            # Convert structured/record arrays to dense 2D in JSON column order
            json_columns = _get_json_columns(json_path)
            arr2d = _structured_to_dense_2d(arr, json_columns)

            required_col_count = max(idx.values()) + 1
            if arr2d.shape[1] < required_col_count:
                raise ValueError(
                    f"[Lagrangian/Raw] Array in {npy_path} has {arr2d.shape[1]} columns, "
                    f"but required index goes up to {required_col_count - 1}"
                )

            # ---------------------------
            # Raw physical quantities
            # ---------------------------
            pos_raw = torch.tensor(
                arr2d[:, [idx["X position"], idx["Y position"], idx["Z position"]]],
                dtype=torch.float32,
            )  # [N,3]

            field_raw = torch.tensor(
                arr2d[:, idx[field_variable]].reshape(-1, 1),
                dtype=torch.float32,
            )  # [N,1]

            cloud_id = torch.tensor(
                arr2d[:, idx["Cloud ID"]],
                dtype=torch.long,
            )  # [N]

            # ---------------------------
            # Features for learning
            # ---------------------------
            pos_feat, field_feat = _normalize_features(pos_raw, field_raw, feature_stats)
            x_feat = torch.cat([pos_feat, field_feat], dim=1)  # [N,4]

            # Build graph from PHYSICAL coordinates
            edge_index = radius_graph(pos_raw, r=radius, loop=False)

            # params_aug = [param_phys..., t_norm]
            t_norm = (tval - t_min) / (t_max - t_min)
            params_aug = np.concatenate(
                [param_val, np.array([t_norm], dtype=np.float32)],
                axis=0,
            )

            data = Data(
                x=x_feat,
                pos=pos_raw,
                edge_index=edge_index,
            )
            data.y = x_feat.clone()
            data.time = torch.tensor([tval], dtype=torch.float32)
            data.params = torch.tensor(params_aug.reshape(1, -1), dtype=torch.float32)
            data.cloud_id = cloud_id
            data.snapshot_name = json_path.name
            data.rev_dir = rev_name

            all_graphs.append(data)

    print(f"[DataLoader] Total graphs loaded: {len(all_graphs)}")
    return all_graphs


# ---------------------------------------------------------------------
# Scaffold extraction
# ---------------------------------------------------------------------

def extract_scaffold_graphs(
    rev_dirs,
    base_data_dir,
    param_mapping,
    param_train_array,   # kept for API compatibility
    user_param_array,
    field_variable="Particle volume fraction",
    radius=0.1,
    sample_ratio=1.0,
    feature_stats=None,
):
    """
    Extract scaffold graphs from the Rev closest to user_param_array
    in PHYSICAL parameter space.

    Uses Rev-level parameters from param_mapping, not snapshot indices.
    """
    user = np.asarray(user_param_array, dtype=np.float32).reshape(1, -1)

    rev_list = list(rev_dirs)
    rev_names = [Path(r).name for r in rev_list]

    rev_params = []
    for rn in rev_names:
        if rn not in param_mapping:
            raise KeyError(f"[Lagrangian/Raw] Missing param mapping for rev '{rn}'")
        rev_params.append(_param_to_1d_array(param_mapping[rn]))

    rev_params = np.stack(rev_params, axis=0)  # [R, P]
    if rev_params.shape[1] != user.shape[1]:
        raise ValueError(
            f"[Lagrangian/Raw] user_param dim {user.shape[1]} "
            f"does not match rev param dim {rev_params.shape[1]}"
        )

    dists = np.linalg.norm(rev_params - user, axis=1)
    best_r = int(np.argmin(dists))
    nearest_rev = rev_list[best_r]

    print(
        f"[Scaffold] Using nearest Rev: {nearest_rev} "
        f"for user_param={user.flatten()}, dist={dists[best_r]:.3e}"
    )

    graphs = load_lagrangian_snapshots_as_graphs(
        rev_dirs=[nearest_rev],
        base_data_dir=base_data_dir,
        param_mapping=param_mapping,
        field_variable=field_variable,
        radius=radius,
        sample_ratio=sample_ratio,
        feature_stats=feature_stats,
    )

    if not graphs:
        raise ValueError(f"[Lagrangian/Raw] No scaffold graphs found for Rev dir: {nearest_rev}")

    return graphs


__all__ = [
    "load_lagrangian_snapshots_as_graphs",
    "compute_feature_stats",
    "extract_scaffold_graphs",
]