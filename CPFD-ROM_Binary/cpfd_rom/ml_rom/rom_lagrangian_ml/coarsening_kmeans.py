# cpfd_rom/ml_rom/rom_lagrangian_ml/coarsening_kmeans.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


@dataclass
class KMeansCoarseningConfig:
    n_clusters: int = 2000
    snapshot_fracs: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.5, 1.0)
    max_points: int = 200_000
    xyz_indices: Tuple[int, int, int] = (0, 1, 2)  # indices in feature dim for x,y,z
    random_state: int = 42
    batch_size: int = 4096
    max_iter: int = 200
    knn_k: int = 12   # neighbors per centroid for graph edges


def _pick_snapshot_indices_from_fracs(
    times: np.ndarray,
    snapshot_fracs: Iterable[float],
) -> List[int]:
    """
    Map fractions in [0,1] of the time span to nearest snapshot indices.

    times: [S] (not necessarily sorted, but usually is)
    Returns a sorted list of unique indices.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    S = times.shape[0]
    if S == 0:
        raise ValueError("No snapshots in times array.")

    # assume times are already ordered by snapshot index
    idxs = set()
    for f in snapshot_fracs:
        f_clamped = float(np.clip(f, 0.0, 1.0))
        i = int(round(f_clamped * (S - 1)))
        i = int(np.clip(i, 0, S - 1))
        idxs.add(i)
    return sorted(idxs)


def collect_multi_snapshot_coords(
    train_times: np.ndarray,      # [S]
    train_data: np.ndarray,       # [S, N, F] (must contain x,y,z in xyz_indices)
    cfg: KMeansCoarseningConfig,
) -> Tuple[np.ndarray, List[int]]:
    """
    Pool coordinates from multiple snapshots (fraction-based selection)
    for K-means training.

    Parameters
    ----------
    train_times : [S]
        Snapshot times (for all revs if you already stacked them).
    train_data : [S, N, F]
        Lagrangian data. We only use train_data[..., xyz_indices] for coords.
    cfg : KMeansCoarseningConfig
        Config with snapshot fractions, max_points, xyz_indices, etc.

    Returns
    -------
    coords_sampled : [M, 3]
        Subsampled coordinates from selected snapshots.
    snapshot_indices : list[int]
        The snapshot indices that were used (fractions ? indices).
    """
    train_times = np.asarray(train_times, dtype=float).reshape(-1)
    train_data = np.asarray(train_data, dtype=float)

    S, N, F = train_data.shape
    if F <= max(cfg.xyz_indices):
        raise ValueError(
            f"train_data has F={F} features, but xyz_indices={cfg.xyz_indices} "
            "expects x,y,z within the last dim."
        )

    # 1) pick snapshot indices by fractions
    snap_indices = _pick_snapshot_indices_from_fracs(train_times, cfg.snapshot_fracs)
    if not snap_indices:
        raise RuntimeError("No snapshot indices selected by fractions.")

    coords_list = []

    for idx in snap_indices:
        # coords: [N, 3] for snapshot idx
        coords_s = train_data[idx, :, cfg.xyz_indices]  # (x,y,z)
        coords_list.append(coords_s.reshape(-1, 3))

    coords_all = np.concatenate(coords_list, axis=0)  # [S_sel * N, 3]

    # 2) subsample if too many points
    M_total = coords_all.shape[0]
    if M_total > cfg.max_points:
        rng = np.random.default_rng(cfg.random_state)
        sel = rng.choice(M_total, size=cfg.max_points, replace=False)
        coords_sampled = coords_all[sel]
    else:
        coords_sampled = coords_all

    return coords_sampled.astype(np.float32), snap_indices


def fit_kmeans_centroids(
    coords_sampled: np.ndarray,  # [M, 3]
    cfg: KMeansCoarseningConfig,
) -> MiniBatchKMeans:
    """
    Fit MiniBatchKMeans on pooled coordinates to obtain static centroids.

    Returns
    -------
    kmeans : MiniBatchKMeans
        Fitted model with attributes:
          - cluster_centers_ : [n_clusters, 3]
    """
    coords_sampled = np.asarray(coords_sampled, dtype=np.float32)
    if coords_sampled.ndim != 2 or coords_sampled.shape[1] != 3:
        raise ValueError(
            f"coords_sampled must be [M, 3], got shape={coords_sampled.shape}"
        )

    print(
        f"[KMEANS] Fitting MiniBatchKMeans with "
        f"{cfg.n_clusters} clusters on {coords_sampled.shape[0]} points..."
    )

    kmeans = MiniBatchKMeans(
        n_clusters=cfg.n_clusters,
        batch_size=cfg.batch_size,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        verbose=False,
    )
    kmeans.fit(coords_sampled)

    print("[KMEANS] Done. inertia =", kmeans.inertia_)
    return kmeans


def build_centroid_knn_edges(
    centroids: np.ndarray,   # [K, 3]
    k: int = 12,
) -> np.ndarray:
    """
    Build a static k-NN graph between centroids.

    Parameters
    ----------
    centroids : [K, 3]
        Cluster centers from k-means.
    k : int
        Number of neighbors for each centroid.

    Returns
    -------
    edges : [E, 2]
        Integer array of undirected edges (i, j). We include both directions
        (i->j and j->i) and then deduplicate.
    """
    centroids = np.asarray(centroids, dtype=np.float32)
    K = centroids.shape[0]
    if K == 0:
        raise ValueError("No centroids provided.")
    if k >= K:
        k = K - 1

    print(f"[KMEANS] Building k-NN graph over {K} centroids with k={k} ...")

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(centroids)
    # distances: [K, k+1]; indices: [K, k+1] (self is included at index 0)
    distances, neighbors = nn.kneighbors(centroids)

    edge_list = []
    for i in range(K):
        for j_idx in range(1, k + 1):
            j = int(neighbors[i, j_idx])
            if i == j:
                continue
            edge_list.append((i, j))
            edge_list.append((j, i))  # make it undirected/symmetric

    edges = np.array(edge_list, dtype=np.int64)
    # deduplicate edges
    if edges.size > 0:
        edges = np.unique(edges, axis=0)

    print(f"[KMEANS] Built {edges.shape[0]} directed edges.")
    return edges


def assign_parcels_to_centroids(
    snapshot_xyz: np.ndarray,  # [N, 3]
    kmeans: MiniBatchKMeans,
) -> np.ndarray:
    """
    Assign parcels of a single snapshot to nearest centroids.

    Parameters
    ----------
    snapshot_xyz : [N, 3]
        Parcel coordinates for one snapshot.
    kmeans : fitted MiniBatchKMeans
        From fit_kmeans_centroids.

    Returns
    -------
    labels : [N]
        Cluster index 0..K-1 for each parcel.
    """
    snapshot_xyz = np.asarray(snapshot_xyz, dtype=np.float32)
    if snapshot_xyz.ndim != 2 or snapshot_xyz.shape[1] != 3:
        raise ValueError(
            f"snapshot_xyz must be [N, 3], got shape={snapshot_xyz.shape}"
        )
    labels = kmeans.predict(snapshot_xyz)  # [N]
    return labels.astype(np.int64)


# Optional: small example of how to tie it together (not used in pipeline directly)
if __name__ == "__main__":
    # Example usage sketch (you'll replace these with real loads):
    S, N = 10, 100_000
    F = 6  # say [x,y,z,field,CloudId,CloudId_base]
    fake_times = np.linspace(0.0, 1.0, S)
    fake_data = np.random.rand(S, N, F).astype(np.float32)

    cfg = KMeansCoarseningConfig(
        n_clusters=2000,
        snapshot_fracs=(0.0, 0.01, 0.05, 0.5, 1.0),
        max_points=200_000,
    )

    coords_sampled, snap_idxs = collect_multi_snapshot_coords(
        fake_times, fake_data, cfg
    )
    kmeans = fit_kmeans_centroids(coords_sampled, cfg)
    edges = build_centroid_knn_edges(kmeans.cluster_centers_, k=cfg.knn_k)

    print("[DEMO] Sampled coords shape:", coords_sampled.shape)
    print("[DEMO] Centroids shape:", kmeans.cluster_centers_.shape)
    print("[DEMO] Edges shape:", edges.shape)
