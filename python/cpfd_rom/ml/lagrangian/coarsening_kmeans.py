"""K-means-based spatial coarsening utilities for Lagrangian ROM data."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)


@dataclass
class KMeansCoarseningConfig:
    """Configuration for centroid fitting and centroid-graph construction."""

    n_clusters: int = 2000
    snapshot_fracs: tuple[float, ...] = (0.0, 0.01, 0.05, 0.5, 1.0)
    max_points: int = 200_000
    xyz_indices: tuple[int, int, int] = (0, 1, 2)
    random_state: int = 42
    batch_size: int = 4096
    max_iter: int = 200
    knn_k: int = 12


def _pick_snapshot_indices_from_fracs(
    times: np.ndarray,
    snapshot_fracs: Iterable[float],
) -> list[int]:
    """Map fractions of a snapshot sequence to sorted, unique indices."""
    times = np.asarray(times, dtype=float).reshape(-1)
    n_snapshots = times.shape[0]
    if n_snapshots == 0:
        raise ValueError("No snapshots in times array.")

    indices: set[int] = set()
    for fraction in snapshot_fracs:
        fraction_value = float(fraction)
        if not np.isfinite(fraction_value):
            raise ValueError("snapshot_fracs must contain only finite values.")
        fraction_clamped = float(np.clip(fraction_value, 0.0, 1.0))
        index = int(round(fraction_clamped * (n_snapshots - 1)))
        indices.add(int(np.clip(index, 0, n_snapshots - 1)))

    return sorted(indices)


def collect_multi_snapshot_coords(
    train_times: np.ndarray,
    train_data: np.ndarray,
    cfg: KMeansCoarseningConfig,
) -> tuple[np.ndarray, list[int]]:
    """Pool and optionally subsample coordinates from selected snapshots."""
    train_times = np.asarray(train_times, dtype=float).reshape(-1)
    train_data = np.asarray(train_data)

    if train_data.ndim != 3:
        raise ValueError(
            f"train_data must be [S, N, F], got shape={train_data.shape}"
        )

    n_snapshots, _, n_features = train_data.shape
    if train_times.shape[0] != n_snapshots:
        raise ValueError(
            "train_times and train_data must contain the same number of "
            f"snapshots, got {train_times.shape[0]} and {n_snapshots}."
        )
    if len(cfg.xyz_indices) != 3 or len(set(cfg.xyz_indices)) != 3:
        raise ValueError("xyz_indices must contain three distinct indices.")
    if min(cfg.xyz_indices) < 0 or max(cfg.xyz_indices) >= n_features:
        raise ValueError(
            f"train_data has F={n_features} features, but "
            f"xyz_indices={cfg.xyz_indices} is invalid."
        )
    if cfg.max_points <= 0:
        raise ValueError("max_points must be greater than zero.")

    snapshot_indices = _pick_snapshot_indices_from_fracs(
        train_times, cfg.snapshot_fracs
    )
    if not snapshot_indices:
        raise ValueError("snapshot_fracs must select at least one snapshot.")

    # np.take preserves the requested x/y/z column order and avoids NumPy's
    # advanced-index axis reordering for train_data[idx, :, tuple_indices].
    coords_all = np.concatenate(
        [
            np.take(train_data[index], cfg.xyz_indices, axis=1).reshape(-1, 3)
            for index in snapshot_indices
        ],
        axis=0,
    )
    if coords_all.shape[0] == 0:
        raise ValueError("Selected snapshots contain no particle coordinates.")
    if not np.all(np.isfinite(coords_all)):
        raise ValueError("Selected particle coordinates contain NaN or infinity.")

    total_points = coords_all.shape[0]
    if total_points > cfg.max_points:
        rng = np.random.default_rng(cfg.random_state)
        selected = rng.choice(total_points, size=cfg.max_points, replace=False)
        coords_sampled = coords_all[selected]
    else:
        coords_sampled = coords_all

    detail(
        logger,
        "[KMeans] Selected snapshots=%s; pooled_points=%d; sampled_points=%d",
        snapshot_indices,
        total_points,
        coords_sampled.shape[0],
    )
    return coords_sampled.astype(np.float32, copy=False), snapshot_indices


def fit_kmeans_centroids(
    coords_sampled: np.ndarray,
    cfg: KMeansCoarseningConfig,
) -> MiniBatchKMeans:
    """Fit MiniBatchKMeans to pooled particle coordinates."""
    coords_sampled = np.asarray(coords_sampled, dtype=np.float32)
    if coords_sampled.ndim != 2 or coords_sampled.shape[1] != 3:
        raise ValueError(
            f"coords_sampled must be [M, 3], got shape={coords_sampled.shape}"
        )
    if not np.all(np.isfinite(coords_sampled)):
        raise ValueError("coords_sampled contains NaN or infinity.")
    if cfg.n_clusters <= 0:
        raise ValueError("n_clusters must be greater than zero.")
    if cfg.n_clusters > coords_sampled.shape[0]:
        raise ValueError(
            f"n_clusters ({cfg.n_clusters}) cannot exceed the number of "
            f"sampled points ({coords_sampled.shape[0]})."
        )
    if cfg.batch_size <= 0 or cfg.max_iter <= 0:
        raise ValueError("batch_size and max_iter must be greater than zero.")

    logger.info("Fitting Lagrangian K-means centroids")
    detail(
        logger,
        "[KMeans] clusters=%d, points=%d, batch_size=%d, max_iter=%d",
        cfg.n_clusters,
        coords_sampled.shape[0],
        cfg.batch_size,
        cfg.max_iter,
    )

    kmeans = MiniBatchKMeans(
        n_clusters=cfg.n_clusters,
        batch_size=cfg.batch_size,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        verbose=False,
    )
    kmeans.fit(coords_sampled)

    detail(logger, "[KMeans] Fit complete; inertia=%.6g", kmeans.inertia_)
    return kmeans


def build_centroid_knn_edges(
    centroids: np.ndarray,
    k: int = 12,
) -> np.ndarray:
    """Build a deduplicated, directed representation of a symmetric k-NN graph."""
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"centroids must be [K, 3], got shape={centroids.shape}")
    if not np.all(np.isfinite(centroids)):
        raise ValueError("centroids contains NaN or infinity.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    n_centroids = centroids.shape[0]
    if n_centroids < 2:
        detail(logger, "[KMeans] Fewer than two centroids; returning no edges")
        return np.empty((0, 2), dtype=np.int64)

    effective_k = min(int(k), n_centroids - 1)
    logger.info("Building the Lagrangian centroid k-NN graph")
    detail(
        logger,
        "[KMeans] centroids=%d, requested_k=%d, effective_k=%d",
        n_centroids,
        k,
        effective_k,
    )

    nearest_neighbors = NearestNeighbors(
        n_neighbors=effective_k + 1,
        algorithm="auto",
    )
    nearest_neighbors.fit(centroids)
    neighbors = nearest_neighbors.kneighbors(centroids, return_distance=False)

    edge_list: list[tuple[int, int]] = []
    for source in range(n_centroids):
        for target_value in neighbors[source]:
            target = int(target_value)
            if source != target:
                edge_list.append((source, target))
                edge_list.append((target, source))

    edges = np.asarray(edge_list, dtype=np.int64).reshape(-1, 2)
    if edges.size:
        edges = np.unique(edges, axis=0)

    detail(logger, "[KMeans] Built %d directed edges", edges.shape[0])
    return edges


def assign_parcels_to_centroids(
    snapshot_xyz: np.ndarray,
    kmeans: MiniBatchKMeans,
) -> np.ndarray:
    """Assign every parcel in one snapshot to its nearest fitted centroid."""
    snapshot_xyz = np.asarray(snapshot_xyz, dtype=np.float32)
    if snapshot_xyz.ndim != 2 or snapshot_xyz.shape[1] != 3:
        raise ValueError(
            f"snapshot_xyz must be [N, 3], got shape={snapshot_xyz.shape}"
        )
    if not np.all(np.isfinite(snapshot_xyz)):
        raise ValueError("snapshot_xyz contains NaN or infinity.")

    return kmeans.predict(snapshot_xyz).astype(np.int64, copy=False)


__all__ = [
    "KMeansCoarseningConfig",
    "assign_parcels_to_centroids",
    "build_centroid_knn_edges",
    "collect_multi_snapshot_coords",
    "fit_kmeans_centroids",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    snapshot_count, particle_count, feature_count = 10, 100_000, 6
    demo_times = np.linspace(0.0, 1.0, snapshot_count)
    demo_data = np.random.default_rng(42).random(
        (snapshot_count, particle_count, feature_count), dtype=np.float32
    )
    demo_cfg = KMeansCoarseningConfig()

    demo_coords, demo_indices = collect_multi_snapshot_coords(
        demo_times, demo_data, demo_cfg
    )
    demo_kmeans = fit_kmeans_centroids(demo_coords, demo_cfg)
    demo_edges = build_centroid_knn_edges(
        demo_kmeans.cluster_centers_, k=demo_cfg.knn_k
    )

    logger.info(
        "Demo complete: snapshots=%s, sampled_coords=%s, centroids=%s, edges=%s",
        demo_indices,
        demo_coords.shape,
        demo_kmeans.cluster_centers_.shape,
        demo_edges.shape,
    )
