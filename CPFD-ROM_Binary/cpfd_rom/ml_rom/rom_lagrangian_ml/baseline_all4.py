# cpfd_rom/ml_rom/rom_lagrangian_ml/baseline_all4.py  (for example)
from pathlib import Path

import numpy as np
from cpfd_rom.ml_rom.rom_eulerian_ml.steps.baseline import (
    build_train_val_baseline,
    make_baseline_model,
)

def build_lagrangian_baseline_model_all4(
    train_times: np.ndarray,        # [S_tr]
    params_all: np.ndarray,         # [S_tr, P]
    train_data: np.ndarray,         # [S_tr, N, 6] (dyn + IDs in physical units)
    poly_deg: int = 1,              # <-- linear
    ridge_alpha: float = 1e-6,
    atol: float = 1e-8,
):
    """
    Build a BaselineModel for *all 4 dynamic channels* [x,y,z,field].

    We flatten over points and channels:
        dyn_phys: [S_tr, N, 4]
        Y_train:  [S_tr, N*4]

    and fit a per-time linear/ridge model over the scalar parameter.
    """
    train_times = np.asarray(train_times, dtype=float).reshape(-1)
    params_all  = np.asarray(params_all,  dtype=float)
    dyn_phys    = train_data[..., :4]              # [S_tr, N, 4]
    S_tr, N, F  = dyn_phys.shape
    assert F == 4, f"Expected 4 dynamic channels, got {F}"

    Y_train = dyn_phys.reshape(S_tr, N * F)        # [S_tr, N*4]

    # No explicit VAL here; pass None for val sets
    W_by_time, meta = build_train_val_baseline(
        times_train=train_times,
        P_train=params_all,
        Y_train=Y_train,         # (S_tr, N*4)
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
    return baseline_model, meta


def build_lagrangian_baseline_dyn_scaled_all4(
    baseline_model,
    infer_times: np.ndarray,         # [S_inf]
    params_infer: np.ndarray,        # [S_inf, P]
    scaler,                          # fitted on 4 dynamic features
    n_points: int,                   # N
    out_path: str | Path,
):
    """
    Use the all-4-channel baseline model to construct a [S_inf, N, 4] baseline
    in *scaled* space, suitable for lagrangian_baseline_infer_npy.

    baseline_model.predict_samples returns [S_inf, N*4] in physical units.
    """
    infer_times  = np.asarray(infer_times,  dtype=float).reshape(-1)
    params_infer = np.asarray(params_infer, dtype=float)

    # 1) Predict baseline in physical space, flattened: [S_inf, N*4]
    baseline_flat_phys = baseline_model.predict_samples(
        times=infer_times,
        P=params_infer,
    )  # [S_inf, N*4]

    S_inf = baseline_flat_phys.shape[0]
    N4    = baseline_flat_phys.shape[1]
    assert N4 % 4 == 0, f"Expected N*4 columns, got {N4}"
    N_inf = N4 // 4
    assert N_inf == n_points, f"Expected n_points={n_points}, got {N_inf}"

    baseline_dyn_phys = baseline_flat_phys.reshape(S_inf, N_inf, 4)  # [S_inf, N, 4]

    # 2) Scale with the same StandardScaler used for training
    flat = baseline_dyn_phys.reshape(-1, 4)          # [S_inf * N, 4]
    flat_scaled = scaler.transform(flat)             # [S_inf * N, 4]
    baseline_dyn_scaled = flat_scaled.reshape(S_inf, N_inf, 4)

    out_path = Path(out_path)
    np.save(out_path, baseline_dyn_scaled)
    print(f"[LAG-BL-ALL4] Saved baseline_dyn_scaled to {out_path}")
    return baseline_dyn_scaled
