# cpfd_rom/ml_rom/rom_lagrangian_ml/data_loader.py

from __future__ import annotations
import os
import re
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from cpfd_rom.util.file_parsing import get_columns_from_json_cached


# ---------------------------------------------------------------------
# Required JSON column names for raw-particle Lagrangian ROM
# ---------------------------------------------------------------------

RAW_REQUIRED_BASE = [
    "Cloud ID",
    "X position",
    "Y position",
    "Z position",
]

# Matches:
#   Raw.particle.00200_2.0000e+01.json
#   Raw.particle.00200_2.0000e+01.npy
RAW_PARTICLE_TIME_RE = re.compile(
    r"^Raw\.particle\.(?P<step>\d+)_(?P<time>[-+0-9.eE]+)\.(json|npy)$"
)
# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _sample_sorted_sequence(seq, sample_ratio: float):
    if sample_ratio >= 1.0:
        return list(seq)

    n_total = len(seq)
    n_keep = max(1, int(round(n_total * sample_ratio)))
    idx = np.linspace(0, n_total - 1, n_keep, dtype=int)
    return [seq[i] for i in idx]
def load_lagrangian_snapshot_times(
    rev_dir: str,
    base_data_dir: str,
    sample_ratio: float = 1.0,
) -> np.ndarray:
    rev_path = os.path.join(base_data_dir, rev_dir)
    files = [
        os.path.join(rev_path, f)
        for f in os.listdir(rev_path)
        if f.startswith("Raw.particle.") and f.endswith(".json")
    ]
    files = sorted(files, key=lambda f: _time_from_raw_particle_filename(Path(f)))
    if len(files) == 0:
        raise RuntimeError(f"No Raw.particle JSON files found in {rev_path}")

    if sample_ratio < 1.0:
        files = _sample_sorted_sequence(files, sample_ratio)

    times = np.array(
        [_time_from_raw_particle_filename(Path(f)) for f in files],
        dtype=np.float64,
    )
    return times
def _list_raw_particle_jsons(dir_path: Path) -> list[Path]:
    """
    Return sorted list of Raw.particle.*.json files in a Rev directory.
    Sorting is by time parsed from filename.
    """
    files = [f for f in dir_path.glob("Raw.particle.*.json") if f.is_file()]
    return sorted(files, key=_time_from_raw_particle_filename)


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


def _time_from_raw_particle_filename(path: Path) -> float:
    """
    Parse simulation time directly from filename.

    Example:
      Raw.particle.00595_5.9502e+01.json -> 59.502
    """
    m = RAW_PARTICLE_TIME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"[Lagrangian/Raw] Could not parse simulation time from filename: {path.name}"
        )
    return float(m.group("time"))


def _get_json_columns(json_path: Path) -> list[str]:
    """
    Read column names from JSON in listed order.
    """
    return list(get_columns_from_json_cached(str(json_path)))


def _build_column_index_from_columns(columns: list[str], field_variable: str) -> dict[str, int]:
    """
    Build required column index map from a pre-read JSON column list.
    """
    required = RAW_REQUIRED_BASE + [field_variable]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"[Lagrangian/Raw] Missing required columns: {missing}\n"
            f"Available columns: {columns}"
        )
    return {name: columns.index(name) for name in required}


def _param_to_1d_array(param_val) -> np.ndarray:
    """
    Support scalar or vector param mapping.
    """
    return np.asarray(param_val, dtype=np.float32).reshape(-1)


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


def _as_contiguous(a: np.ndarray) -> np.ndarray:
    """
    Ensure array is contiguous before torch.from_numpy.
    """
    if a.flags["C_CONTIGUOUS"]:
        return a
    return np.ascontiguousarray(a)


def _extract_required_columns(
    arr: np.ndarray,
    columns: list[str],
    idx: dict[str, int],
    field_variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract only the required columns:
      - xyz
      - field
      - cloud id

    Supports:
      1) dense ndarray of shape [N, C]
      2) structured ndarray of shape [N] with named fields

    Returns:
      pos_np   : [N, 3] float32
      field_np : [N, 1] float32
      cloud_np : [N]    integer-like
    """
    # ------------------------------------------------------------
    # Dense 2D case
    # ------------------------------------------------------------
    if arr.ndim == 2:
        required_col_count = max(idx.values()) + 1
        if arr.shape[1] < required_col_count:
            raise ValueError(
                f"[Lagrangian/Raw] Dense array has {arr.shape[1]} columns, "
                f"but required index goes up to {required_col_count - 1}"
            )

        pos_np = arr[:, [idx["X position"], idx["Y position"], idx["Z position"]]].astype(
            np.float32, copy=False
        )
        field_np = arr[:, idx[field_variable]].reshape(-1, 1).astype(np.float32, copy=False)
        cloud_np = arr[:, idx["Cloud ID"]]
        return pos_np, field_np, cloud_np

    # ------------------------------------------------------------
    # Structured / record array case
    # ------------------------------------------------------------
    if arr.ndim == 1 and arr.dtype.names is not None:
        field_names = arr.dtype.names

        # Fast path: exact field-name lookup
        required_names = [
            "X position",
            "Y position",
            "Z position",
            "Cloud ID",
            field_variable,
        ]

        if all(name in field_names for name in required_names):
            pos_np = np.stack(
                [
                    np.asarray(arr["X position"], dtype=np.float32),
                    np.asarray(arr["Y position"], dtype=np.float32),
                    np.asarray(arr["Z position"], dtype=np.float32),
                ],
                axis=1,
            )
            field_np = np.asarray(arr[field_variable], dtype=np.float32).reshape(-1, 1)
            cloud_np = np.asarray(arr["Cloud ID"])
            return pos_np, field_np, cloud_np

        # Fallback: JSON-defined order must align with structured fields count
        if len(field_names) != len(columns):
            raise ValueError(
                f"[Lagrangian/Raw] Structured NPY field count ({len(field_names)}) does not "
                f"match JSON column count ({len(columns)}).\n"
                f"NPY fields: {field_names}\n"
                f"JSON columns: {columns}"
            )

        # Extract only required columns, not the whole dense matrix
        x_name = columns[idx["X position"]]
        y_name = columns[idx["Y position"]]
        z_name = columns[idx["Z position"]]
        cloud_name = columns[idx["Cloud ID"]]
        field_name = columns[idx[field_variable]]

        missing = [n for n in [x_name, y_name, z_name, cloud_name, field_name] if n not in field_names]
        if missing:
            raise ValueError(
                f"[Lagrangian/Raw] Structured NPY missing required fields: {missing}\n"
                f"Structured fields: {field_names}"
            )

        pos_np = np.stack(
            [
                np.asarray(arr[x_name], dtype=np.float32),
                np.asarray(arr[y_name], dtype=np.float32),
                np.asarray(arr[z_name], dtype=np.float32),
            ],
            axis=1,
        )
        field_np = np.asarray(arr[field_name], dtype=np.float32).reshape(-1, 1)
        cloud_np = np.asarray(arr[cloud_name])
        return pos_np, field_np, cloud_np

    raise ValueError(
        f"[Lagrangian/Raw] Unsupported NPY format: shape={arr.shape}, dtype={arr.dtype}"
    )


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
    radius=0.001,
    sample_ratio=1.0,
    feature_stats=None,
    verbose_timing: bool = False,
):
    """
    Load raw-particle Lagrangian snapshots as PyG graphs.

    Optimizations in this version:
      - Read simulation time from filename, not JSON
      - Read columns once per Rev from first JSON
      - Avoid converting full structured arrays to dense if only a few fields are needed
      - Use np.load(..., mmap_mode="r")
      - Use torch.from_numpy when possible

    Each graph contains:
      - data.x        : [N,4] = [x,y,z,field] (normalized if feature_stats given)
      - data.y        : [N,4] same as x for AE reconstruction
      - data.pos      : [N,3] physical xyz
      - data.edge_index
      - data.params   : [1,P_aug] = [physical params..., t_norm]
      - data.time     : [1]
      - data.cloud_id : [N]
    """
    all_graphs: list[Data] = []
    all_times: list[float] = []

    rev_paths = [Path(base_data_dir) / rev for rev in rev_dirs]

    # ------------------------------------------------------------
    # First pass: gather per-rev metadata + global time range
    # ------------------------------------------------------------
    print("[DataLoader] Scanning raw-particle files for time range and schema...")

    rev_meta: dict[str, dict] = {}

    for rev_path in rev_paths:
        if not rev_path.exists():
            raise FileNotFoundError(f"[Lagrangian/Raw] Rev directory not found: {rev_path}")

        json_files = _list_raw_particle_jsons(rev_path)
        if not json_files:
            raise FileNotFoundError(
                f"[Lagrangian/Raw] No Raw.particle.*.json files found in {rev_path}"
            )

        # Read schema only once per Rev
        first_json = json_files[0]
        columns = _get_json_columns(first_json)
        idx = _build_column_index_from_columns(columns, field_variable)

        snapshot_meta = []
        for json_path in json_files:
            tval = _time_from_raw_particle_filename(json_path)
            all_times.append(tval)
            snapshot_meta.append(
                {
                    "json_path": json_path,
                    "npy_path": _matching_npy_from_json(json_path),
                    "time": tval,
                }
            )

        rev_meta[str(rev_path)] = {
            "columns": columns,
            "idx": idx,
            "snapshots": snapshot_meta,
        }

    if not all_times:
        raise RuntimeError("[Lagrangian/Raw] No simulation times found in raw-particle files.")

    t_min = float(min(all_times))
    t_max = float(max(all_times))
    print(f"[DataLoader] Global time range: t_min={t_min:.6f}, t_max={t_max:.6f}")

    if t_max == t_min:
        raise ValueError("[Lagrangian/Raw] Zero time range  all snapshot times are identical.")

    # ------------------------------------------------------------
    # Second pass: load selected files
    # ------------------------------------------------------------
    print("[DataLoader] Loading raw-particle snapshots and constructing graphs...")

    for rev_path in rev_paths:
        rev_name = rev_path.name
        if rev_name not in param_mapping:
            raise KeyError(f"[Lagrangian/Raw] Missing param mapping for rev '{rev_name}'")

        param_val = _param_to_1d_array(param_mapping[rev_name])

        meta = rev_meta[str(rev_path)]
        columns = meta["columns"]
        idx = meta["idx"]
        snapshots = meta["snapshots"]

        # Already sorted by time because _list_raw_particle_jsons sorts by parsed time
        n_total = len(snapshots)
        sampled_snapshots = _sample_sorted_sequence(snapshots, sample_ratio)

        for snap in tqdm(
                sampled_snapshots,
                desc=f"[DataLoader] {rev_name}",
                unit="snapshot",
                leave=True,
        ):
            json_path = snap["json_path"]
            npy_path = snap["npy_path"]
            tval = snap["time"]

            if verbose_timing:
                t0 = time.time()

            arr = np.load(npy_path, allow_pickle=False, mmap_mode="r")

            if verbose_timing:
                t1 = time.time()

            pos_np, field_np, cloud_np = _extract_required_columns(
                arr=arr,
                columns=columns,
                idx=idx,
                field_variable=field_variable,
            )

            if verbose_timing:
                t2 = time.time()

            pos_raw = torch.from_numpy(_as_contiguous(pos_np)).to(torch.float32)
            field_raw = torch.from_numpy(_as_contiguous(field_np)).to(torch.float32)
            cloud_id = torch.from_numpy(_as_contiguous(cloud_np)).to(torch.long)

            if verbose_timing:
                t3 = time.time()

            pos_feat, field_feat = _normalize_features(pos_raw, field_raw, feature_stats)
            x_feat = torch.cat([pos_feat, field_feat], dim=1)  # [N,4]

            # Build graph from PHYSICAL coordinates
            edge_index = radius_graph(
                pos_raw,
                r=radius,
                loop=False,
                max_num_neighbors=16,
            )

            if verbose_timing:
                t4 = time.time()

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
            data.time = torch.tensor([tval], dtype=torch.float64)
            data.params = torch.tensor(params_aug.reshape(1, -1), dtype=torch.float32)
            data.cloud_id = cloud_id
            data.snapshot_name = json_path.name
            data.rev_dir = rev_name

            all_graphs.append(data)

            if verbose_timing:
                print(
                    f"[Timing] {json_path.name}: "
                    f"load={t1 - t0:.3f}s, "
                    f"extract={t2 - t1:.3f}s, "
                    f"tensor={t3 - t2:.3f}s, "
                    f"graph={t4 - t3:.3f}s"
                )

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
    verbose_timing: bool = False,
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
        verbose_timing=verbose_timing,
    )

    if not graphs:
        raise ValueError(f"[Lagrangian/Raw] No scaffold graphs found for Rev dir: {nearest_rev}")

    return graphs


__all__ = [
    "load_lagrangian_snapshots_as_graphs",
    "load_lagrangian_snapshot_times",
    "compute_feature_stats",
    "extract_scaffold_graphs",
]