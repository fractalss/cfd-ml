# cpfd_rom/ml_rom/rom_lagrangian_ml/data_loader.py

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from tqdm import tqdm

from cpfd_rom.util.file_parsing import get_columns_from_json_cached
from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)

# The shared DETAIL level sits between DEBUG (10) and INFO (20).
DETAIL_LEVEL = logging.INFO - 5


RAW_REQUIRED_BASE = [
    "Cloud ID",
    "X position",
    "Y position",
    "Z position",
]

RAW_PARTICLE_TIME_RE = re.compile(
    r"^Raw\.particle\.(?P<step>\d+)_(?P<time>[-+0-9.eE]+)\.(json|npy)$"
)


def _sample_sorted_sequence(seq, sample_ratio: float):
    if sample_ratio >= 1.0:
        return list(seq)
    n_total = len(seq)
    n_keep = max(1, int(round(n_total * sample_ratio)))
    idx = np.linspace(0, n_total - 1, n_keep, dtype=int)
    return [seq[i] for i in idx]


def _time_from_raw_particle_filename(path: Path) -> float:
    m = RAW_PARTICLE_TIME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"[Lagrangian/Raw] Could not parse simulation time from filename: {path.name}"
        )
    return float(m.group("time"))


def _list_raw_particle_jsons(dir_path: Path) -> list[Path]:
    files = [f for f in dir_path.glob("Raw.particle.*.json") if f.is_file()]
    files = sorted(files, key=_time_from_raw_particle_filename)

    if len(files) >= 2:
        times = [_time_from_raw_particle_filename(f) for f in files]
        if not np.all(np.diff(times) >= 0):
            raise ValueError(f"[Lagrangian/Raw] Snapshot times are not sorted for {dir_path}")

    return files


def _matching_npy_from_json(json_path: Path) -> Path:
    npy_path = json_path.with_suffix(".npy")
    if not npy_path.exists():
        raise FileNotFoundError(
            f"[Lagrangian/Raw] Matching NPY file not found for {json_path}"
        )
    return npy_path


def _get_json_columns(json_path: Path) -> list[str]:
    return list(get_columns_from_json_cached(str(json_path)))


def _build_column_index_from_columns(columns: list[str], field_variable: str) -> dict[str, int]:
    required = RAW_REQUIRED_BASE + [field_variable]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"[Lagrangian/Raw] Missing required columns: {missing}\n"
            f"Available columns: {columns}"
        )
    return {name: columns.index(name) for name in required}


def _param_to_1d_array(param_val) -> np.ndarray:
    return np.asarray(param_val, dtype=np.float32).reshape(-1)


def _as_contiguous(a: np.ndarray) -> np.ndarray:
    return a if a.flags["C_CONTIGUOUS"] else np.ascontiguousarray(a)


def _normalize_features(
    pos_raw: torch.Tensor,
    field_raw: torch.Tensor,
    feature_stats: Optional[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_stats is None:
        return pos_raw, field_raw

    eps = 1e-8
    pos_mean = feature_stats["pos_mean"].to(pos_raw.device, dtype=pos_raw.dtype)
    pos_std = feature_stats["pos_std"].to(pos_raw.device, dtype=pos_raw.dtype)
    fmin = torch.tensor(float(feature_stats["field_min"]), device=field_raw.device, dtype=field_raw.dtype)
    fmax = torch.tensor(float(feature_stats["field_max"]), device=field_raw.device, dtype=field_raw.dtype)

    pos_feat = (pos_raw - pos_mean) / (pos_std + eps)
    field_feat = (field_raw - fmin) / (fmax - fmin + eps)
    return pos_feat, field_feat


def _extract_required_columns(
    arr: np.ndarray,
    columns: list[str],
    idx: dict[str, int],
    field_variable: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arr.ndim == 2:
        required_col_count = max(idx.values()) + 1
        if arr.shape[1] < required_col_count:
            raise ValueError(
                f"[Lagrangian/Raw] Dense array has {arr.shape[1]} columns, "
                f"but required index goes up to {required_col_count - 1}"
            )

        pos_np = arr[:, [idx["X position"], idx["Y position"], idx["Z position"]]].astype(
            np.float32,
            copy=False,
        )
        field_np = arr[:, idx[field_variable]].reshape(-1, 1).astype(np.float32, copy=False)
        cloud_np = arr[:, idx["Cloud ID"]]
        return pos_np, field_np, cloud_np

    if arr.ndim == 1 and arr.dtype.names is not None:
        field_names = arr.dtype.names

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

        if len(field_names) != len(columns):
            raise ValueError(
                f"[Lagrangian/Raw] Structured NPY field count ({len(field_names)}) does not "
                f"match JSON column count ({len(columns)}).\n"
                f"NPY fields: {field_names}\n"
                f"JSON columns: {columns}"
            )

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


def _build_rev_snapshot_meta(
    rev_path: Path,
    field_variable: str,
) -> dict:
    if not rev_path.exists():
        raise FileNotFoundError(f"[Lagrangian/Raw] Rev directory not found: {rev_path}")

    json_files = _list_raw_particle_jsons(rev_path)
    if not json_files:
        raise FileNotFoundError(
            f"[Lagrangian/Raw] No Raw.particle.*.json files found in {rev_path}"
        )

    first_json = json_files[0]
    columns = _get_json_columns(first_json)
    idx = _build_column_index_from_columns(columns, field_variable)

    snapshot_meta = []
    for json_path in json_files:
        snapshot_meta.append(
            {
                "json_path": json_path,
                "npy_path": _matching_npy_from_json(json_path),
                "time": _time_from_raw_particle_filename(json_path),
            }
        )

    snapshot_meta = sorted(snapshot_meta, key=lambda s: float(s["time"]))
    times = np.array([float(s["time"]) for s in snapshot_meta], dtype=np.float64)

    if len(times) >= 2 and not np.all(np.diff(times) >= 0):
        raise ValueError(f"[Lagrangian/Raw] Snapshot metadata not time-sorted for {rev_path}")

    return {
        "rev_path": rev_path,
        "rev_name": rev_path.name,
        "columns": columns,
        "idx": idx,
        "snapshots": snapshot_meta,
    }


def _build_data_object(
    *,
    pos_raw: torch.Tensor,
    field_raw: torch.Tensor,
    cloud_id: torch.Tensor,
    edge_index: torch.Tensor,
    params: np.ndarray,
    tval: float,
    snapshot_name: str,
    rev_name: str,
    feature_stats: Optional[dict],
) -> Data:
    pos_feat, field_feat = _normalize_features(pos_raw, field_raw, feature_stats)
    x_feat = torch.cat([pos_feat, field_feat], dim=1)

    data = Data(
        x=x_feat,
        y=x_feat.clone(),
        pos=pos_raw,
        edge_index=edge_index,
    )
    data.time = torch.tensor([tval], dtype=torch.float32)
    data.params = torch.tensor(params.reshape(1, -1), dtype=torch.float32)
    data.cloud_id = cloud_id
    data.snapshot_name = snapshot_name
    data.rev_dir = rev_name
    return data


def compute_feature_stats(graphs: list[Data]) -> dict[str, torch.Tensor | float]:
    if not graphs:
        raise ValueError("[Lagrangian/Raw] Cannot compute feature stats on empty graph list.")

    all_pos = torch.cat([g.pos for g in graphs], dim=0)

    field_tensors = []
    for g in graphs:
        if hasattr(g, "field_raw"):
            field_tensors.append(g.field_raw)
        else:
            field_tensors.append(g.x[:, 3:4])

    all_field = torch.cat(field_tensors, dim=0)

    pos_std = all_pos.std(dim=0)
    pos_std = torch.where(pos_std > 0, pos_std, torch.ones_like(pos_std))

    return {
        "pos_mean": all_pos.mean(dim=0),
        "pos_std": pos_std,
        "field_min": all_field.min().item(),
        "field_max": all_field.max().item(),
    }


def load_lagrangian_snapshot_times(
    rev_dir: str,
    base_data_dir: str,
    sample_ratio: float = 1.0,
    field_variable: str = "Particle volume fraction",
) -> np.ndarray:
    rev_path = Path(base_data_dir) / rev_dir
    meta = _build_rev_snapshot_meta(rev_path, field_variable=field_variable)
    snapshots = _sample_sorted_sequence(meta["snapshots"], sample_ratio)
    times = np.array([float(s["time"]) for s in snapshots], dtype=np.float64)

    if len(times) >= 2 and not np.all(np.diff(times) >= 0):
        raise ValueError(
            f"[Lagrangian/Raw] load_lagrangian_snapshot_times produced unsorted times for {rev_dir}"
        )

    return times


def load_lagrangian_snapshots_as_graphs(
    rev_dirs,
    base_data_dir,
    param_mapping,
    field_variable="Particle volume fraction",
    radius=0.001,
    sample_ratio=1.0,
    feature_stats=None,
    max_num_neighbors: int = 16,
    verbose_timing: bool = False,
):
    all_graphs: list[Data] = []
    all_times: list[float] = []

    rev_paths = [Path(base_data_dir) / rev for rev in rev_dirs]

    logger.info("[DataLoader] Scanning raw-particle files for time range and schema...")
    rev_meta: dict[str, dict] = {}

    for rev_path in rev_paths:
        meta = _build_rev_snapshot_meta(rev_path, field_variable=field_variable)
        rev_meta[str(rev_path)] = meta
        all_times.extend(float(s["time"]) for s in meta["snapshots"])

    if not all_times:
        raise RuntimeError("[Lagrangian/Raw] No simulation times found in raw-particle files.")

    t_min = float(min(all_times))
    t_max = float(max(all_times))
    detail(logger, "[DataLoader] Global time range: t_min=%.6f, t_max=%.6f", t_min, t_max)

    if t_max == t_min:
        raise ValueError("[Lagrangian/Raw] Zero time range  all snapshot times are identical.")

    logger.info("[DataLoader] Loading raw-particle snapshots and constructing graphs...")

    for rev_path in rev_paths:
        meta = rev_meta[str(rev_path)]
        rev_name = meta["rev_name"]

        if rev_name not in param_mapping:
            raise KeyError(f"[Lagrangian/Raw] Missing param mapping for rev '{rev_name}'")

        param_val = _param_to_1d_array(param_mapping[rev_name])
        snapshots = _sample_sorted_sequence(meta["snapshots"], sample_ratio)

        sampled_times = np.array([float(s["time"]) for s in snapshots], dtype=np.float64)
        if len(sampled_times) >= 2 and not np.all(np.diff(sampled_times) >= 0):
            raise ValueError(f"[Lagrangian/Raw] Sampled snapshots not sorted for rev '{rev_name}'")

        for snap in tqdm(
            snapshots,
            desc=f"[DataLoader] {rev_name}",
            unit="snapshot",
            leave=True,
            disable=not logger.isEnabledFor(DETAIL_LEVEL),
        ):
            json_path = snap["json_path"]
            npy_path = snap["npy_path"]
            tval = float(snap["time"])

            if verbose_timing:
                t0 = time.time()

            arr = np.load(npy_path, allow_pickle=False, mmap_mode="r")

            if verbose_timing:
                t1 = time.time()

            pos_np, field_np, cloud_np = _extract_required_columns(
                arr=arr,
                columns=meta["columns"],
                idx=meta["idx"],
                field_variable=field_variable,
            )

            if verbose_timing:
                t2 = time.time()

            pos_raw = torch.from_numpy(_as_contiguous(pos_np)).to(torch.float32)
            field_raw = torch.from_numpy(_as_contiguous(field_np)).to(torch.float32)
            cloud_id = torch.from_numpy(_as_contiguous(cloud_np)).to(torch.long)

            if verbose_timing:
                t3 = time.time()

            edge_index = radius_graph(
                pos_raw,
                r=radius,
                loop=False,
                max_num_neighbors=max_num_neighbors,
            )

            if verbose_timing:
                t4 = time.time()

            data = _build_data_object(
                pos_raw=pos_raw,
                field_raw=field_raw,
                cloud_id=cloud_id,
                edge_index=edge_index,
                params=param_val,
                tval=tval,
                snapshot_name=json_path.name,
                rev_name=rev_name,
                feature_stats=feature_stats,
            )
            data.field_raw = field_raw

            all_graphs.append(data)

            if verbose_timing:
                logger.debug(
                    "[Timing] %s: load=%.3fs, extract=%.3fs, tensor=%.3fs, graph=%.3fs",
                    json_path.name,
                    t1 - t0,
                    t2 - t1,
                    t3 - t2,
                    t4 - t3,
                )

    logger.info("[DataLoader] Total graphs loaded: %d", len(all_graphs))
    return all_graphs


def sort_graphs_by_time(graphs: list[Data]) -> list[Data]:
    if not graphs:
        return []

    sorted_graphs = sorted(graphs, key=lambda g: float(g.time.item()))
    times = np.array([float(g.time.item()) for g in sorted_graphs], dtype=np.float64)

    if len(times) >= 2 and not np.all(np.diff(times) >= 0):
        raise ValueError("[Lagrangian/Raw] sort_graphs_by_time failed to produce sorted graphs.")

    return sorted_graphs


def get_initial_template_graph(graphs: list[Data]) -> Data:
    if not graphs:
        raise ValueError("[Lagrangian/Raw] get_initial_template_graph received empty graph list.")

    sorted_graphs = sort_graphs_by_time(graphs)
    g0 = sorted_graphs[0]

    if not hasattr(g0, "snapshot_name"):
        raise AttributeError("[Lagrangian/Raw] Initial template graph is missing snapshot_name.")
    if not hasattr(g0, "time"):
        raise AttributeError("[Lagrangian/Raw] Initial template graph is missing time.")

    detail(
        logger,
        "[TemplateInit] Using initial template snapshot '%s' at time %.6f",
        g0.snapshot_name,
        float(g0.time.item()),
    )
    return g0


def extract_scaffold_graphs(
    rev_dirs,
    base_data_dir,
    param_mapping,
    param_train_array,
    user_param_array,
    field_variable="Particle volume fraction",
    radius=0.1,
    sample_ratio=1.0,
    feature_stats=None,
    max_num_neighbors: int = 16,
    verbose_timing: bool = False,
):
    del param_train_array

    user = np.asarray(user_param_array, dtype=np.float32).reshape(1, -1)

    rev_list = list(rev_dirs)
    rev_names = [Path(r).name for r in rev_list]

    rev_params = []
    for rn in rev_names:
        if rn not in param_mapping:
            raise KeyError(f"[Lagrangian/Raw] Missing param mapping for rev '{rn}'")
        rev_params.append(_param_to_1d_array(param_mapping[rn]))

    rev_params = np.stack(rev_params, axis=0)
    if rev_params.shape[1] != user.shape[1]:
        raise ValueError(
            f"[Lagrangian/Raw] user_param dim {user.shape[1]} "
            f"does not match rev param dim {rev_params.shape[1]}"
        )

    dists = np.linalg.norm(rev_params - user, axis=1)
    best_r = int(np.argmin(dists))
    nearest_rev = rev_list[best_r]
    nearest_rev_name = Path(nearest_rev).name

    detail(
        logger,
        "[TemplateSelect] Using nearest Rev: %s for user_param=%s, dist=%.3e",
        nearest_rev,
        user.flatten(),
        dists[best_r],
    )

    graphs = load_lagrangian_snapshots_as_graphs(
        rev_dirs=[nearest_rev],
        base_data_dir=base_data_dir,
        param_mapping=param_mapping,
        field_variable=field_variable,
        radius=radius,
        sample_ratio=sample_ratio,
        feature_stats=feature_stats,
        max_num_neighbors=max_num_neighbors,
        verbose_timing=verbose_timing,
    )

    if not graphs:
        raise ValueError(
            f"[Lagrangian/Raw] No template graphs found for nearest Rev dir: {nearest_rev}"
        )

    graphs = sort_graphs_by_time(graphs)
    g0 = get_initial_template_graph(graphs)

    detail(
        logger,
        "[TemplateSelect] Confirmed nearest Rev '%s' has %d time-sorted graphs. "
        "Initial template = '%s'.",
        nearest_rev_name,
        len(graphs),
        g0.snapshot_name,
    )

    return graphs


__all__ = [
    "load_lagrangian_snapshots_as_graphs",
    "load_lagrangian_snapshot_times",
    "compute_feature_stats",
    "sort_graphs_by_time",
    "get_initial_template_graph",
    "extract_scaffold_graphs",
]
