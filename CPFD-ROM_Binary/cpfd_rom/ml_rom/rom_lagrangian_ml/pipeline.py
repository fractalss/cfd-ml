# cpfd_rom/ml_rom/rom_lagrangian_ml/pipeline.py

import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For geometry projection loop

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import load_lagrangian_snapshots
from cpfd_rom.ml_rom.rom_lagrangian_ml.mlp_encoder import (
    PointNetResidualDecoder,
    # PointNetGATResidualDecoder,  # keep commented if you want
)

from cpfd_rom.ml_rom.rom_lagrangian_ml.datasets import (
    BaselineResidualDataset,
    build_params_from_rev_dirs,
)

from cpfd_rom.ml_rom.rom_lagrangian_ml.training import (
    train_pointnet_residual_torch,
)
from cpfd_rom.ml_rom.rom_lagrangian_ml.geometry_projection import GeometryProjector
from cpfd_rom.ml_rom.rom_lagrangian_ml.evaluation import write_lagrangian_rom_only
from cpfd_rom.ml_rom.rom_lagrangian_ml.baseline_all4 import (
    build_lagrangian_baseline_model_all4,
    build_lagrangian_baseline_dyn_scaled_all4,
)

from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.util.model_utils import setup_model_paths


# ---------------------------------------------------------------------------
# Helper: read columns.txt and map to the 6 features used by the ROM
# ---------------------------------------------------------------------------

def _load_columns_from_dir(columns_dir: Path):
    """Load column names from columns.txt in a Rev*_npy directory."""
    columns_dir = Path(columns_dir)
    columns_file = columns_dir / "columns.txt"
    if not columns_file.exists():
        raise FileNotFoundError(f"[Lagrangian] columns.txt not found in {columns_dir}")

    with open(columns_file, "r") as f:
        cols = [line.strip() for line in f if line.strip()]

    if not cols:
        raise ValueError(f"[Lagrangian] columns.txt in {columns_dir} is empty")
    return cols


def _select_rom_features(data: np.ndarray, rev_dirs, field_var: str) -> np.ndarray:
    """Select the 6 ROM features from raw data using columns.txt.

    Raw data may have more than 6 columns (e.g., 11). We use columns.txt
    in the first Rev*_npy directory to find the indices for:

        ["x", "y", "z", field_var, "Cloud Id", "Cloud Id base"]

    and slice the data accordingly, returning an array with shape
    [S, N, 6] in that exact order.
    """
    if data.ndim != 3:
        raise ValueError(f"[Lagrangian] Expected 3D data [S, N, C], got {data.shape}")

    columns_dir = Path(rev_dirs[0])
    cols = _load_columns_from_dir(columns_dir)

    required = ["x", "y", "z", field_var, "Cloud Id", "Cloud Id base"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"[Lagrangian] Missing required columns in columns.txt: {missing}"
        )

    idx = [cols.index(c) for c in required]
    # Slice in the required order: [x, y, z, field, CloudID, CloudID_base]
    data_sel = data[..., idx]
    return data_sel


def run_lagrangian_ml_pipeline(config, log_time):
    """Baseline + residual decoder pipeline for Lagrangian ROM.

    Workflow:
      1) Load snapshots & params (once).
      2) Scale CFD dynamic features [x,y,z,field].
      3) Build linear/ridge baseline model and evaluate it on training set
         using *physical* parameters only (no time appended).
      4) Augment parameters for the residual decoder by appending a
         normalized time coordinate t_norm in [0, 1].
      5) Compute residual_dyn_scaled = CFD_dyn_scaled - baseline_dyn_scaled.
      6) Train PointNetResidualDecoder:
             (baseline_dyn_scaled, params_phys+time) -> residual_dyn_scaled.
      7) Inference at config.user_parameter:
             baseline_dyn_scaled(user_param_phys, t)
           + residual_pred_scaled(user_param_phys+time, t)
           -> inverse-scale, clip, write ROM for 201 snapshots.
    """

    # ---------------- Device ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA with {torch.cuda.device_count()} GPU(s).")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA not available, using CPU.")

    # ---------------- Paths & config ----------------
    setup_output_dir(config)
    setup_model_paths(config)

    # robust handling of model/scaler/script paths
    base_model_path = getattr(
        config,
        "model_path_lagrangian",
        os.path.join(config.model_dir, "pointnet_lagrangian.pt"),
    )
    base_stem, base_ext = os.path.splitext(base_model_path)
    if base_ext == "":
        base_ext = ".pt"
        base_model_path = base_stem + base_ext

    model_path = base_model_path
    scaler_path = base_stem + "_scaler.pkl"
    script_path = base_stem + "_script.pt"
    baseline_path = os.path.join(config.model_dir, "baseline_dyn_scaled.npy")

    epochs = getattr(config, "epochs", 50)
    batch_size = getattr(config, "batch_size", 2)
    patience = getattr(config, "patience", 10)
    lr = getattr(config, "learning_rate", 3e-4)

    field_var = getattr(config, "field_variable", None)
    if field_var is None:
        raise ValueError(
            "[Lagrangian] config.field_variable must be set (e.g., 'pvf', 'speed')."
        )

    if not hasattr(config, "rev_dirs") or not config.rev_dirs:
        raise ValueError("[Lagrangian] config.rev_dirs must be provided.")

    # ==============================================================
    # 1. LOAD SNAPSHOTS & PARAMS ONCE
    # ==============================================================

    with log_time("Loading Lagrangian npy snapshots (once)"):
        print("[INFO] Loading Lagrangian snapshots from npy...")
        train_times, train_data_raw = load_lagrangian_snapshots(config.rev_dirs)
        train_times = train_times.astype(np.float64)  # [S]

        # select 6 ROM features: [x,y,z,field,CloudId,CloudId_base]
        train_data = _select_rom_features(train_data_raw, config.rev_dirs, field_var)
        n_snaps, n_points, n_features_total = train_data.shape

        if n_features_total != 6:
            print(
                f"[WARN] Expected 6 ROM features, got {n_features_total}. "
                "Continuing but downstream assumes 6."
            )

        # build per-snapshot *physical* params (no time yet)
        param_mapping = getattr(config, "param_mapping", None)
        if param_mapping is None:
            raise RuntimeError(
                "[Lagrangian] Baseline+residual ROM requires config.param_mapping."
            )

        # params_all_phys: [S, P_phys] (e.g. superficial velocity, etc.)
        params_all_phys = build_params_from_rev_dirs(config.rev_dirs, param_mapping)
        if params_all_phys.shape[0] != n_snaps:
            raise ValueError(
                f"params_all has {params_all_phys.shape[0]} entries but train_data has {n_snaps} snapshots"
            )

        # ---- Build normalized time coordinate t_norm in [0, 1] for each snapshot ----
        t_min = float(train_times.min())
        t_max = float(train_times.max())
        if t_max > t_min:
            t_norm = (train_times - t_min) / (t_max - t_min)
        else:
            # degenerate case: all times equal
            t_norm = np.zeros_like(train_times)
        t_norm = t_norm.astype(np.float32).reshape(-1, 1)  # [S, 1]

        # ---- Augmented params for residual model: [params_phys, t_norm] ----
        params_all_resid = np.concatenate([params_all_phys, t_norm], axis=1)  # [S, P_phys+1]

        param_dim_phys = params_all_phys.shape[1]
        param_dim_resid = params_all_resid.shape[1]

        # Store for reference if needed elsewhere
        config.param_dim_phys = param_dim_phys
        config.param_dim = param_dim_resid  # param_dim used by residual decoder

    # ==============================================================
    # 2. SCALE CFD DYNAMIC FEATURES [x,y,z,field]
    # ==============================================================

    n_features_dyn = 4  # first 4: x,y,z,field

    with log_time("Scaling Lagrangian dynamic features"):
        flat = train_data.reshape(-1, n_features_total)  # [S*N, 6]
        flat_dyn = flat[:, :n_features_dyn]              # [S*N, 4]
        flat_ids = flat[:, n_features_dyn:]              # [S*N, 2]

        scaler = None
        scaler_loaded = False

        if os.path.exists(scaler_path):
            try:
                loaded = joblib.load(scaler_path)
                if isinstance(loaded, StandardScaler):
                    scaler = loaded
                    scaler_loaded = True
                    print(f"[INFO] Loaded existing scaler from {scaler_path}")
                    if (
                        hasattr(scaler, "n_features_in_")
                        and scaler.n_features_in_ != n_features_dyn
                    ):
                        raise ValueError(
                            f"Loaded scaler expects {scaler.n_features_in_} features, "
                            f"but dynamic feature count is {n_features_dyn}."
                        )
                else:
                    print(
                        f"[WARN] Object loaded from {scaler_path} is not a StandardScaler; "
                        "refitting scaler."
                    )
            except Exception as e:
                print(
                    f"[WARN] Failed to load existing scaler from {scaler_path}: {e}\n"
                    "       Refitting scaler on current data."
                )
                scaler_loaded = False

        if not scaler_loaded:
            scaler = StandardScaler()
            flat_dyn_scaled = scaler.fit_transform(flat_dyn)
            try:
                joblib.dump(scaler, scaler_path)
                print(f"[INFO] Saved new scaler to {scaler_path}")
            except Exception as e:
                print(f"[WARN] Failed to save scaler to {scaler_path}: {e}")
        else:
            flat_dyn_scaled = scaler.transform(flat_dyn)

        # CFD dynamic scaled and full scaled (if ever needed)
        flat_scaled = np.concatenate([flat_dyn_scaled, flat_ids], axis=1)
        train_scaled_full = flat_scaled.reshape(train_data.shape)  # [S, N, 6]

        # keep dynamic only as separate tensor
        train_dyn_scaled = flat_dyn_scaled.reshape(n_snaps, n_points, n_features_dyn)

        # bounds in physical space (for clipping)
        train_min = flat_dyn.min(axis=0)
        train_max = flat_dyn.max(axis=0)

    # ==============================================================
    # 3. BASELINE: ALWAYS BUILD baseline_dyn_scaled FOR THIS RUN
    # ==============================================================

    print(f"[INFO] Lagrangian baseline (scaled) will be written to: {baseline_path}")

    # If an old baseline file exists from a previous run with different
    # rev_dirs / snapshot count, remove it to avoid confusion.
    if os.path.exists(baseline_path):
        print(f"[INFO] Removing stale baseline file at {baseline_path}")
        try:
            os.remove(baseline_path)
        except Exception as e:
            print(f"[WARN] Failed to remove old baseline file: {e}")

    poly_deg    = getattr(config, "lagrangian_baseline_poly_deg", 1)
    ridge_alpha = getattr(config, "lagrangian_baseline_ridge_alpha", 1e-6)
    atol        = getattr(config, "lagrangian_baseline_atol", 1e-8)

    with log_time("Fitting baseline model on full training set"):
        # NOTE: baseline uses *physical* params only (no time column).
        baseline_model, meta = build_lagrangian_baseline_model_all4(
            train_times=train_times,
            params_all=params_all_phys,
            train_data=train_data,  # physical [S, N, 6]
            poly_deg=poly_deg,
            ridge_alpha=ridge_alpha,
            atol=atol,
        )

        # Minimal layout info for baseline scaling
        ref_layout = {
            "N": n_points,
            "n_dyn_channels": n_features_dyn,
        }

        # optional: save baseline model for debugging/inspection
        baseline_model_path = os.path.join(config.model_dir, "baseline_model.pkl")
        try:
            joblib.dump(baseline_model, baseline_model_path)
            print(f"[INFO] Saved baseline model to {baseline_model_path}")
        except Exception as e:
            print(f"[WARN] Failed to save baseline model to {baseline_model_path}: {e}")

    with log_time("Evaluating baseline (scaled) on training set"):
        build_lagrangian_baseline_dyn_scaled_all4(
            baseline_model,
            infer_times=train_times,
            params_infer=params_all_phys,
            scaler=scaler,
            ref_layout=ref_layout,
            n_points=train_data.shape[1],
            out_path=baseline_path,
        )

    baseline_dyn_scaled = np.load(baseline_path)

    if baseline_dyn_scaled.shape != (n_snaps, n_points, n_features_dyn):
        raise ValueError(
            f"[Lagrangian] Newly built baseline_dyn_scaled shape {baseline_dyn_scaled.shape} "
            f"does not match expected {(n_snaps, n_points, n_features_dyn)}"
        )

    # ==============================================================
    # 4. RESIDUAL IN SCALED SPACE: CFD_dyn_scaled - baseline_dyn_scaled
    # ==============================================================

    resid_dyn_scaled = train_dyn_scaled - baseline_dyn_scaled  # [S, N, 4]

    # ---- Sanity check: algebraic reconstruction ----
    recon_train_dyn_scaled = baseline_dyn_scaled + resid_dyn_scaled   # [S, N, 4]
    err_recon = train_dyn_scaled - recon_train_dyn_scaled             # [S, N, 4]

    recon_rmse = np.sqrt(np.mean(err_recon**2))
    recon_max = np.max(np.abs(err_recon))

    print(
        f"[CHECK] Algebraic recon (train_dyn_scaled baseline + resid): "
        f"RMSE={recon_rmse:.3e}, max|err|={recon_max:.3e}"
    )

    # Debug: baseline-only error (scaled space)
    resid_flat = resid_dyn_scaled.reshape(-1, n_features_dyn)
    baseline_mse = np.mean(resid_flat ** 2)
    baseline_rmse = np.sqrt(baseline_mse)
    print(
        f"[DEBUG] Baseline-only MSE (scaled) = {baseline_mse:.4e}, "
        f"RMSE = {baseline_rmse:.4e}"
    )

    # ==============================================================
    # 5. TRAIN OR LOAD RESIDUAL DECODER (baseline + params_phys+time -> residual)
    # ==============================================================

    skip_training = getattr(config, "skip_training", False)
    latent_dim = getattr(config, "latent_dim", 256)

    model = PointNetResidualDecoder(
        in_dim=4,                 # baseline dynamic features
        param_dim=param_dim_resid,  # global params for decoder: [params_phys, t_norm]
        latent_dim=latent_dim,
        out_dim=4,
    ).to(device)

    if skip_training and os.path.exists(model_path):
        with log_time("Loading pretrained residual decoder"):
            print(f"[INFO] Loading residual decoder from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
    else:
        # build dataset & loaders, using augmented params_all_resid
        full_dataset = BaselineResidualDataset(
            baseline_dyn_scaled=baseline_dyn_scaled,
            params_all=params_all_resid,         # NOTE: includes time
            resid_dyn_scaled=resid_dyn_scaled,
        )

        indices = np.arange(len(full_dataset))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True
        )

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        from cpfd_rom.ml_rom.rom_lagrangian_ml.training import (
            train_pointnet_residual_torch,
        )

        with log_time("Training residual decoder (baseline+params+time -> residual)"):
            print("[INFO] Training PointNet residual decoder...")
            model = train_pointnet_residual_torch(
                model,
                train_loader,
                val_loader,
                device,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )

        # save model + scaler
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Saved residual decoder to {model_path}")
        print(f"[INFO] Saved scaler to {scaler_path}")

        # optional TorchScript export
        try:
            model.eval()
            example_baseline = torch.randn(1, n_points, 4, device=device)
            example_params = torch.randn(1, param_dim_resid, device=device)
            scripted = torch.jit.trace(model, (example_baseline, example_params))
            scripted.save(script_path)
            print(f"[INFO] Saved TorchScript residual decoder to {script_path}")
        except Exception as e:
            print(f"[WARN] TorchScript export failed: {e}")

    model.eval()

    # ==============================================================
    # 5b. DIAGNOSTIC: residual model training-set error (UNWEIGHTED)
    # ==============================================================

    with torch.no_grad():
        B_all = torch.from_numpy(baseline_dyn_scaled).float().to(device)    # [S, N, 4]
        P_all = torch.from_numpy(params_all_resid).float().to(device)       # [S, P_resid]
        R_true_all = torch.from_numpy(resid_dyn_scaled).float().to(device)  # [S, N, 4]

        S_total = B_all.shape[0]
        batch_diag = getattr(config, "diagnostic_batch_size", 32)

        mse_sum = 0.0
        count = 0

        for s0 in range(0, S_total, batch_diag):
            B_b = B_all[s0 : s0 + batch_diag]   # [B, N, 4]
            P_b = P_all[s0 : s0 + batch_diag]   # [B, P_resid]
            R_b = R_true_all[s0 : s0 + batch_diag]

            R_pred_b = model(B_b, P_b)          # [B, N, 4]

            mse_b = F.mse_loss(R_pred_b, R_b, reduction="mean").item()
            mse_sum += mse_b * B_b.shape[0]
            count += B_b.shape[0]

        if count > 0:
            mse_mean = mse_sum / count
            rmse_mean = float(np.sqrt(mse_mean))
            print(
                f"[CHECK] Residual model train MSE (unweighted, scaled) = {mse_mean:.4e}, "
                f"RMSE = {rmse_mean:.4e}"
            )
        else:
            print("[CHECK] Residual model diagnostic: no batches?")

    # ==============================================================
    # 6. INFERENCE AT UNSEEN PARAMETER (config.user_parameter)
    # ==============================================================

    with log_time("Running Lagrangian inference at unseen user_parameter"):
        # ---------- 1) Unique time grid (201 times) ----------
        n_revs = len(config.rev_dirs)
        snaps_per_rev = n_snaps // n_revs   # e.g. 1005 // 5 = 201

        # All rev_dirs share the same time instants; first 201 belong to Rev1
        times_unique = train_times[:snaps_per_rev].copy()      # [201]

        # ---------- 2) Build param arrays for user_parameter ----------
        infer_param = getattr(config, "user_parameter", None)
        if infer_param is None:
            raise ValueError(
                "[Lagrangian] config.user_parameter must be set for unseen-parameter inference."
            )

        # user_parameter is in *physical* param space only (no time)
        infer_param_vec_phys = np.atleast_1d(infer_param).astype(np.float32)   # [P_phys]
        P_phys = infer_param_vec_phys.shape[0]
        if P_phys != param_dim_phys:
            raise ValueError(
                f"[Lagrangian] user_parameter dim {P_phys} does not match param_dim_phys {param_dim_phys}"
            )

        # 201 copies of the same physical param vector => [201, P_phys]
        params_infer_phys = np.repeat(infer_param_vec_phys[None, :], snaps_per_rev, axis=0)

        # Normalized time for the inference time grid, using the same scaling
        # as training (t_min, t_max).
        if t_max > t_min:
            t_norm_inf = (times_unique - t_min) / (t_max - t_min)
        else:
            t_norm_inf = np.zeros_like(times_unique)
        t_norm_inf = t_norm_inf.astype(np.float32).reshape(-1, 1)   # [201, 1]

        # Augmented params for residual inference: [params_phys, t_norm_inf]
        params_infer_resid = np.concatenate(
            [params_infer_phys.astype(np.float32), t_norm_inf],
            axis=1,
        )  # [201, P_phys+1]

        # ---------- 3) Evaluate baseline at (times_unique, user_parameter_phys) ----------
        baseline_user_path = os.path.join(
            config.model_dir, "baseline_dyn_scaled_user_param.npy"
        )
        build_lagrangian_baseline_dyn_scaled_all4(
            baseline_model,
            infer_times=times_unique,
            params_infer=params_infer_phys,   # baseline uses physical params only
            scaler=scaler,
            ref_layout=ref_layout,
            n_points=train_data.shape[1],
            out_path=baseline_user_path,
        )

        # Load baseline in SCALED space for the unseen param
        baseline_dyn_scaled_inf = np.load(baseline_user_path)            # [201, N, 4]

        S_inf, N_inf, _ = baseline_dyn_scaled_inf.shape
        infer_batch_size = getattr(config, "infer_batch_size", batch_size)

        preds_resid_scaled_list = []

        with torch.no_grad():
            B_all_inf = torch.from_numpy(baseline_dyn_scaled_inf).float().to(device)  # [201, N, 4]
            p_all_inf = torch.from_numpy(params_infer_resid).float().to(device)       # [201, P_resid]

            for b_start in range(0, S_inf, infer_batch_size):
                B_b = B_all_inf[b_start : b_start + infer_batch_size]  # [B, N, 4]
                p_b = p_all_inf[b_start : b_start + infer_batch_size]  # [B, P_resid]
                R_pred_b = model(B_b, p_b)                             # [B, N, 4]
                preds_resid_scaled_list.append(R_pred_b.cpu().numpy())

        preds_resid_scaled = np.concatenate(preds_resid_scaled_list, axis=0)      # [201, N, 4]

    # ---------- 6) Combine baseline + residual (scaled) ----------
    dyn_scaled_full = baseline_dyn_scaled_inf + preds_resid_scaled                # [201, N, 4]

    # ==============================================================
    # 7. INVERSE SCALING, CLIPPING, APPEND IDs, GEOMETRY, WRITE
    # ==============================================================

    flat_pred = dyn_scaled_full.reshape(-1, 4)
    flat_pred_unscaled = scaler.inverse_transform(flat_pred)
    flat_pred_clipped = np.clip(flat_pred_unscaled, train_min, train_max)
    pred_dyn = flat_pred_clipped.reshape(dyn_scaled_full.shape)                   # [201, N, 4]

    # Reuse IDs from first Rev only (layout is identical across rev_dirs)
    ids = train_data[:snaps_per_rev, :, 4:]                                       # [201, N, 2]
    pred_array = np.concatenate([pred_dyn, ids], axis=-1)                         # [201, N, 6]

    with log_time("Writing Lagrangian ROM output"):
        rom_out_dir = getattr(config, "rom_output_dir_lagrangian", None)
        if rom_out_dir is None:
            base_rom_dir = getattr(config, "rom_output_dir", "rom_output")
            rom_out_dir = os.path.join(base_rom_dir, "Lagrangian_ROM")

        print(f"[INFO] Writing Lagrangian ROM snapshots to: {rom_out_dir}")

        if getattr(config, "enforce_geometry", False):
            stl_path = getattr(config, "geometry_stl_path", None)
            if stl_path is None:
                raise ValueError(
                    "[Lagrangian] enforce_geometry=True but geometry_stl_path is not set."
                )

            print(f"[INFO] Loading STL geometry: {stl_path}")
            projector = GeometryProjector(stl_path)
            print("[INFO] STL loaded. Beginning geometry projection...")

            S_inf, N_inf, _ = pred_array.shape
            print("[INFO] Projecting predicted particle coordinates into geometry volume...")
            for s in tqdm(range(S_inf), desc="Geometry projection (snapshots)", unit="snap"):
                xyz = pred_array[s, :, 0:3]
                xyz_proj = projector.project_points_inside(xyz)
                pred_array[s, :, 0:3] = xyz_proj

        columns_dir = Path(config.rev_dirs[0])
        write_lagrangian_rom_only(
            preds=pred_array,
            times=times_unique,          # 201 time instants
            columns_dir=columns_dir,
            out_dir=Path(rom_out_dir),
            field_var=field_var,
            train_min=train_min,
            train_max=train_max,
        )
