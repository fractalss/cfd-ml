# cpfd_rom/ml_rom/rom_lagrangian_ml/baseline_all4.py
#
# Lagrangian baseline for all 4 dynamic channels [x, y, z, field]
# using a direct polynomial / ridge regression, reusing the Eulerian
# baseline infrastructure. No bed-height logic is used here.
#
# We:
#   1) Extract the first 4 dynamic channels from train_data: dyn_phys: [S_tr, N, 4]
#   2) Flatten over points and channels: Y_train: [S_tr, N*4]
#   3) Fit a per-time polynomial / ridge model using build_train_val_baseline.
#   4) At inference, predict the flattened [N*4] vector and reshape to [S_inf, N, 4].
#   5) Scale with the provided StandardScaler (fit on [*, 4]).
#
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

from cpfd_rom.ml_rom.rom_eulerian_ml.steps.baseline import (
    build_train_val_baseline,
    make_baseline_model,
)


# =============================================================================
# CONSISTENCY CHECKS (Cloud IDs, time grid)
# =============================================================================


def _group_indices_by_params(params_all: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Group snapshot indices by unique parameter vectors.

    Assumes each 'rev dir' corresponds to a unique parameter vector P,
    and within that rev the parameters are constant across snapshots.
    """
    params_all = np.asarray(params_all, dtype=float)
    if params_all.ndim == 1:
        params_all = params_all.reshape(-1, 1)

    # np.unique with axis=0 groups rows that are bitwise identical
    unique_P, inverse = np.unique(params_all, axis=0, return_inverse=True)
    groups: Dict[int, np.ndarray] = {}
    for g in range(unique_P.shape[0]):
        groups[g] = np.where(inverse == g)[0]
    return groups


def _check_time_grid_consistency(
    train_times: np.ndarray,
    params_all: np.ndarray,
    time_atol: float = 1e-9,
) -> None:
    """
    Ensure that the time stamps per snapshot are the same in each rev dir.

    We infer 'rev dirs' by grouping snapshots with identical parameter vectors.
    For each group g (one rev), we take its time grid and compare (sorted)
    against a reference time grid. If any group differs, we raise an error.
    """
    train_times = np.asarray(train_times, dtype=float).reshape(-1)
    groups = _group_indices_by_params(params_all)

    ref_time_grid: Optional[np.ndarray] = None
    for g, idx in groups.items():
        if idx.size == 0:
            continue

        t_group = np.sort(train_times[idx])
        if ref_time_grid is None:
            ref_time_grid = t_group
        else:
            if t_group.shape != ref_time_grid.shape:
                raise ValueError(
                    "[Lagrangian] Time grid mismatch across revs: "
                    f"group {g} has {t_group.size} times, "
                    f"reference has {ref_time_grid.size}."
                )
            if np.max(np.abs(t_group - ref_time_grid)) > time_atol:
                raise ValueError(
                    "[Lagrangian] Time grid mismatch across revs: "
                    f"group {g} times differ from reference by more than {time_atol}."
                )


def _check_cloud_ids_consistency(
    train_data: np.ndarray,
    params_all: np.ndarray,
    cloud_id_idx: int = 4,
) -> None:
    """
    Ensure that Cloud IDs are consistent across snapshots within each rev dir.

    Logic:
      - Group snapshots by parameter vector (one group ~ one rev dir).
      - For each group:
          * Take Cloud IDs from the first snapshot (sorted).
          * For every other snapshot in the group, check that the sorted
            Cloud IDs match exactly.
    """
    train_data = np.asarray(train_data, dtype=float)
    S_tr, N, F = train_data.shape

    if cloud_id_idx < 0 or cloud_id_idx >= F:
        # No Cloud ID column available; nothing to check.
        return

    groups = _group_indices_by_params(params_all)

    for g, idx in groups.items():
        if idx.size <= 1:
            # Only one snapshot for this param group; trivially consistent.
            continue

        # Reference snapshot for this rev
        ref_cloud = train_data[idx[0], :, cloud_id_idx]
        ref_sorted = np.sort(ref_cloud)

        for s in idx[1:]:
            cloud_s = train_data[s, :, cloud_id_idx]
            cloud_s_sorted = np.sort(cloud_s)
            if ref_sorted.shape != cloud_s_sorted.shape or not np.array_equal(
                ref_sorted, cloud_s_sorted
            ):
                raise ValueError(
                    "[Lagrangian] Inconsistent Cloud IDs across snapshots for one "
                    f"rev (parameter group {g}). "
                    "Cloud ID multiset differs between snapshots."
                )


# =============================================================================
# PUBLIC  BUILD BASELINE MODEL (DIRECT POLY / RIDGE)
# =============================================================================


def build_lagrangian_baseline_model_all4(
    train_times: np.ndarray,        # [S_tr]
    params_all: np.ndarray,         # [S_tr, P]
    train_data: np.ndarray,         # [S_tr, N, F] (dyn + IDs in physical units)
    poly_deg: int = 1,              # linear by default
    ridge_alpha: float = 1e-6,
    atol: float | str = 1e-8,
    *,
    # Consistency-check knobs
    check_time_grid: bool = True,
    check_cloud_ids: bool = True,
    cloud_id_idx: int = 4,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a BaselineModel for *all 4 dynamic channels* [x, y, z, field].

    We flatten over points and channels:

        dyn_phys : [S_tr, N, 4]  (first 4 channels of train_data)
        Y_train  : [S_tr, N*4]

    and fit a per-time polynomial / ridge model over the parameters using the
    cpfd_rom Eulerian baseline infrastructure.

    During construction, we also optionally enforce:

      1) Time grid consistency:
         For each distinct parameter vector (rev dir), the sorted list of
         snapshot times must match across all revs.

      2) Cloud ID consistency:
         For each rev dir, the multiset of Cloud IDs (column cloud_id_idx) must
         be identical across snapshots.

    Parameters
    ----------
    train_times : [S_tr]
        Snapshot times (global list across all revs).
    params_all : [S_tr, P]
        Operating parameters for each snapshot (e.g., velocity, mass flow).
        We assume that within one rev dir, the parameters are constant.
    train_data : [S_tr, N, F]
        Lagrangian features in physical units.
        We assume:
            - train_data[..., 0:4] = [x, y, z, field]
            - train_data[..., cloud_id_idx] = Cloud ID (if available).
    poly_deg : int
        Polynomial degree in the baseline (1 => linear).
    ridge_alpha : float
        Ridge regularization strength.
    atol : float or str
        Absolute tolerance for small singular values (see Eulerian baseline).
    check_time_grid : bool
        If True, validate that the time grid per snapshot is the same in each
        rev dir (grouped by identical parameter vectors).
    check_cloud_ids : bool
        If True, validate that Cloud IDs are consistent across snapshots within
        each rev dir.
    cloud_id_idx : int
        Index of the Cloud ID channel in train_data (default 4).

    Returns
    -------
    baseline_model : BaselineModel
        Object with .predict_samples(times, P) -> [S_inf, N*4] in physical units.
    meta : dict
        Metadata, including:
            - "observable": "lagrangian_all4_linear"
            - "n_points": N
            - "n_dyn_channels": 4
            - "flat_dim": N*4
            - plus fields from build_train_val_baseline(...)
    """
    # ---- Coerce inputs ----
    train_times = np.asarray(train_times, dtype=float).reshape(-1)
    params_all = np.asarray(params_all, dtype=float)
    train_data = np.asarray(train_data, dtype=float)

    S_tr, N, F = train_data.shape
    if F < 4:
        raise ValueError(
            f"[Lagrangian] Expected at least 4 channels in train_data for "
            f"[x,y,z,field], got F={F}"
        )

    if params_all.shape[0] != S_tr:
        raise ValueError(
            f"[Lagrangian] params_all length {params_all.shape[0]} does not "
            f"match number of snapshots S_tr={S_tr}."
        )

    # ---- Coerce atol to float (allow string from YAML) ----
    if isinstance(atol, str):
        try:
            atol = float(atol)
        except ValueError:
            raise ValueError(
                f"[Lagrangian] atol must be convertible to float, got {atol!r}"
            )
    elif not isinstance(atol, (float, int)):
        raise TypeError(
            f"[Lagrangian] atol must be float or int, got {type(atol)}"
        )
    atol = float(atol)

    # ---- Optional consistency checks ----
    if check_time_grid:
        _check_time_grid_consistency(
            train_times=train_times,
            params_all=params_all,
            time_atol=1e-9,
        )

    if check_cloud_ids:
        _check_cloud_ids_consistency(
            train_data=train_data,
            params_all=params_all,
            cloud_id_idx=cloud_id_idx,
        )

    # ---- Extract dynamic channels: [x, y, z, field] ----
    dyn_phys = train_data[..., :4]  # [S_tr, N, 4]

    # ---- Flatten over points and channels: [S_tr, N*4] ----
    Y_train = dyn_phys.reshape(S_tr, -1)  # [S_tr, N*4]
    flat_dim = Y_train.shape[1]

    # ---- Fit polynomial / ridge baseline model on flattened output ----
    W_by_time, meta_base = build_train_val_baseline(
        times_train=train_times,
        P_train=params_all,
        Y_train=Y_train,
        times_val=None,
        P_val=None,
        Y_val=None,
        poly_deg=poly_deg,
        ridge_alpha=ridge_alpha,
        atol=atol,
    )

    baseline_model = make_baseline_model(
        W_by_time=W_by_time,
        poly_deg=poly_deg,
        ridge_alpha=ridge_alpha,
        atol=atol,
    )

    # ---- Assemble metadata ----
    meta = dict(meta_base)
    meta.update(
        {
            "observable": "lagrangian_all4_linear",
            "n_points": int(N),
            "n_dyn_channels": 4,
            "flat_dim": int(flat_dim),
            "poly_deg": int(poly_deg),
            "ridge_alpha": float(ridge_alpha),
            # consistency-check settings
            "check_time_grid": bool(check_time_grid),
            "check_cloud_ids": bool(check_cloud_ids),
            "cloud_id_idx": int(cloud_id_idx),
        }
    )

    return baseline_model, meta


# =============================================================================
# PUBLIC  BASELINE IN SCALED SPACE
# =============================================================================


def build_lagrangian_baseline_dyn_scaled_all4(
    H_model,                  # actually: baseline_model from above
    infer_times: np.ndarray,  # [S_inf]
    params_infer: np.ndarray, # [S_inf, P]
    scaler,
    ref_layout: Dict[str, Any],
    out_path: str | Path,
    *,
    n_points: Optional[int] = None,
) -> np.ndarray:
    """
    Build baseline snapshots in *scaled* space for all 4 dynamic channels.

        baseline_dyn_scaled : [S_inf, N, 4]

    Notes
    -----
    - `H_model` is the generic BaselineModel returned by
      build_lagrangian_baseline_model_all4 (name kept for compatibility).
    - `ref_layout` must at least contain:
          ref_layout["N"]              -> N (number of particles)
          ref_layout["n_dyn_channels"] -> 4
      If you don't have a rich layout, you can pass:
          ref_layout = {"N": N, "n_dyn_channels": 4}
    """
    baseline_model = H_model  # rename for clarity

    infer_times = np.asarray(infer_times, dtype=float).reshape(-1)
    params_infer = np.asarray(params_infer, dtype=float)
    S_inf = infer_times.shape[0]

    # ---- Retrieve layout info (N and number of dynamic channels) ----
    if not isinstance(ref_layout, dict):
        raise TypeError(
            f"[LAG-BL-ALL4] Expected ref_layout to be a dict with 'N', "
            f"got {type(ref_layout)}"
        )

    if "N" not in ref_layout:
        raise KeyError(
            "[LAG-BL-ALL4] ref_layout must contain key 'N' (number of particles)"
        )

    N = int(ref_layout["N"])
    n_dyn_channels = int(ref_layout.get("n_dyn_channels", 4))

    if n_dyn_channels != 4:
        raise ValueError(
            f"[LAG-BL-ALL4] Expected n_dyn_channels=4, got {n_dyn_channels}"
        )

    # If user passed an explicit n_points, sanity-check against N
    if n_points is not None and int(n_points) != N:
        raise ValueError(
            f"[LAG-BL-ALL4] n_points={n_points} does not match ref_layout['N']={N}"
        )

    # ---- 1) Predict flattened baseline in *physical* units ----
    #      flat_pred_phys : [S_inf, N*4]
    flat_pred_phys = baseline_model.predict_samples(
        times=infer_times,
        P=params_infer,
    )
    flat_pred_phys = np.asarray(flat_pred_phys, dtype=float)

    if flat_pred_phys.ndim != 2:
        raise ValueError(
            f"[LAG-BL-ALL4] Expected flat_pred_phys to have shape [S_inf, N*4], "
            f"got shape {flat_pred_phys.shape}"
        )

    S_check, flat_dim = flat_pred_phys.shape
    if S_check != S_inf:
        raise ValueError(
            f"[LAG-BL-ALL4] Predicted S_inf={S_check} does not match "
            f"infer_times length {S_inf}"
        )

    expected_dim = N * n_dyn_channels
    if flat_dim != expected_dim:
        raise ValueError(
            f"[LAG-BL-ALL4] Expected flattened dim N*4={expected_dim}, "
            f"got {flat_dim}. Check training/inference consistency."
        )

    # ---- 2) Reshape to [S_inf * N, 4] to use the scaler ----
    dyn_phys = flat_pred_phys.reshape(S_inf * N, n_dyn_channels)  # [S_inf * N, 4]

    # ---- 3) Scale using the provided StandardScaler ----
    dyn_scaled_flat = scaler.transform(dyn_phys)  # [S_inf * N, 4]

    # ---- 4) Reshape back to [S_inf, N, 4] ----
    baseline_dyn_scaled = dyn_scaled_flat.reshape(S_inf, N, n_dyn_channels)

    # ---- 5) Save and log ----
    out_path = Path(out_path)
    np.save(out_path, baseline_dyn_scaled)
    print(f"[LAG-BL-ALL4] Saved baseline_dyn_scaled to {out_path}")

    return baseline_dyn_scaled
