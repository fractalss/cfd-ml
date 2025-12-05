# cpfd_rom/ml_rom/rom_lagrangian_ml/pipeline.py

import os
import joblib
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from tqdm import tqdm  # For geometry projection loop

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import load_lagrangian_snapshots
from cpfd_rom.ml_rom.rom_lagrangian_ml.mlp_encoder import PointNetAutoencoder
from cpfd_rom.ml_rom.rom_lagrangian_ml.datasets import (
    SnapshotDataset,
    SnapshotParamDataset,
    build_params_from_rev_dirs,
)
from cpfd_rom.ml_rom.rom_lagrangian_ml.training import train_pointnet_torch
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_lagrangian_ml_pipeline(config, log_time):
    """End-to-end pipeline for Lagrangian PointNet ROM.

    Raw npy snapshots may contain more than 6 per-point features
    (e.g., 11). Using columns.txt in the Rev*_npy directories, we
    extract exactly 6 per-point features in this order:

        [x, y, z, field_variable, CloudID, CloudID_base]

    Dynamic (to be predicted by the ROM):
        x, y, z, field_variable   4 features

    Static (constant across snapshots, not predicted, only carried through):
        CloudID, CloudID_base     2 features

    The neural network now operates **only on the 4 dynamic features**:
        in_dim  = 4  (x, y, z, field_variable)
        out_dim = 4  (reconstruct x, y, z, field_variable)

    CloudID / CloudID_base are kept outside the network and reattached
    downstream before writing Tecplot-style particles*.txt.

    The StandardScaler is fit on the 4 dynamic features only, and is
    shared between the baseline and the ML model.
    """

    # ---------------- GPU / device setup (PyTorch) ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA with {torch.cuda.device_count()} GPU(s).")
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA not available, using CPU.")

    # ---------------- Setup output paths ----------------
    setup_output_dir(config)
    setup_model_paths(config)

    model_path = getattr(config, "model_path_lagrangian", "pointnet_lagrangian.pt")
    scaler_path = model_path.replace(".pt", "_scaler.pkl")

    epochs = getattr(config, "epochs", 1)
    batch_size = getattr(config, "batch_size", 2)
    patience = getattr(config, "patience", 10)
    lr = getattr(config, "learning_rate", 1e-3)

    # Field variable to use as the scalar in the ROM
    field_var = getattr(config, "field_variable", None)
    if field_var is None:
        raise ValueError(
            "[Lagrangian] config.field_variable must be set (e.g., 'pvf', 'speed')."
        )

    if not hasattr(config, "rev_dirs") or not config.rev_dirs:
        raise ValueError("[Lagrangian] config.rev_dirs must be provided.")

    # ==============================================================
    # 1. LOAD SNAPSHOTS ONCE (times + data) AND DERIVED QUANTITIES
    # ==============================================================

    with log_time("Loading Lagrangian npy snapshots (once)"):
        print("[INFO] Loading Lagrangian snapshots from npy...")
        train_times, train_data_raw = load_lagrangian_snapshots(config.rev_dirs)
        train_times = train_times.astype(np.float64)

        # Select the 6 ROM features using columns.txt
        train_data = _select_rom_features(train_data_raw, config.rev_dirs, field_var)
        n_snaps, n_points, n_features_total = train_data.shape
        if n_features_total != 6:
            print(
                f"[WARN] Expected 6 ROM features, got {n_features_total}. "
                "Continuing but downstream code assumes 6."
            )

        # Dynamic vs static feature counts
        n_features_dyn = 4  # x, y, z, field
        config.n_features_total = n_features_total  # should be 6
        config.n_dynamic_features = n_features_dyn  # 4

        # Build param array once (if mapping is provided)
        param_mapping = getattr(config, "param_mapping", None)
        params_all = None
        if param_mapping is not None:
            params_all = build_params_from_rev_dirs(config.rev_dirs, param_mapping)
            if params_all.shape[0] != n_snaps:
                raise ValueError(
                    f"params_all has {params_all.shape[0]} entries but "
                    f"train_data has {n_snaps} snapshots"
                )
            param_dim = params_all.shape[1]
        else:
            param_dim = 0

        config.param_dim = param_dim

    # ==============================================================
    # 2. SETUP / LOAD SCALER AND SCALE DATA ONCE
    # ==============================================================

    with log_time("Scaling Lagrangian dynamic features"):
        flat = train_data.reshape(-1, n_features_total)  # [S*N, 6]
        flat_dyn = flat[:, :n_features_dyn]              # [S*N, 4]
        flat_ids = flat[:, n_features_dyn:]              # [S*N, 2]

        scaler_loaded = False
        scaler = None

        if os.path.exists(scaler_path):
            try:
                loaded = joblib.load(scaler_path)
                # Basic sanity check
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
                        "will refit a new scaler."
                    )
            except Exception as e:
                print(
                    f"[WARN] Failed to load existing scaler from {scaler_path}: {e}\n"
                    "       The file may be corrupted or from an incompatible format. "
                    "       A new scaler will be fit and saved."
                )
                scaler_loaded = False

        if not scaler_loaded:
            # Fit a fresh scaler on current data
            scaler = StandardScaler()
            flat_dyn_scaled = scaler.fit_transform(flat_dyn)
            # Overwrite the bad/old scaler file with the new one
            try:
                joblib.dump(scaler, scaler_path)
                print(f"[INFO] Saved new scaler to {scaler_path}")
            except Exception as e:
                print(f"[WARN] Failed to save scaler to {scaler_path}: {e}")
        else:
            # Use the loaded scaler
            flat_dyn_scaled = scaler.transform(flat_dyn)

        flat_scaled = np.concatenate([flat_dyn_scaled, flat_ids], axis=1)
        train_scaled = flat_scaled.reshape(train_data.shape)

        # Precompute bounds for clipping in physical space later
        train_min = flat_dyn.min(axis=0)
        train_max = flat_dyn.max(axis=0)

    # ==============================================================
    # 3. BASELINE: ALWAYS BUILD ONCE (IF MISSING), USING LOADED SNAPSHOTS
    # ==============================================================

    baseline_path = os.path.join(config.model_dir, "baseline_dyn_scaled.npy")
    print(f"[INFO] Lagrangian baseline will be created/used at: {baseline_path}")

    if not os.path.exists(baseline_path):
        print(
            f"[INFO] Baseline file not found at {baseline_path}; "
            "building all-4-channel linear baseline..."
        )

        if params_all is None:
            raise RuntimeError(
                "Baseline requires config.param_mapping to be set and params_all to be available."
            )

        poly_deg = getattr(config, "lagrangian_baseline_poly_deg", 1)
        ridge_alpha = getattr(config, "lagrangian_baseline_ridge_alpha", 1e-6)
        atol = getattr(config, "lagrangian_baseline_atol", 1e-8)

        with log_time("Fitting baseline model on full training set"):
            baseline_model, meta = build_lagrangian_baseline_model_all4(
                train_times=train_times,
                params_all=params_all,
                train_data=train_data,
                poly_deg=poly_deg,
                ridge_alpha=ridge_alpha,
                atol=atol,
            )

        with log_time("Evaluating baseline and writing baseline_dyn_scaled.npy"):
            build_lagrangian_baseline_dyn_scaled_all4(
                baseline_model=baseline_model,
                infer_times=train_times,
                params_infer=params_all,
                scaler=scaler,
                n_points=train_data.shape[1],
                out_path=baseline_path,
            )
    else:
        print(f"[INFO] Baseline already exists at {baseline_path}; not rebuilding.")

    # ==============================================================
    # 4. TRAIN OR LOAD POINTNET MODEL (USING SAME SCALED DATA)
    # ==============================================================

    trained_here = False
    model = None

    if getattr(config, "skip_training", False) and os.path.exists(model_path):
        # --------- Load pretrained model + scaler ---------
        with log_time("Loading pretrained PyTorch PointNet model"):
            print(f"[INFO] Loading pretrained model from {model_path}")
            latent_dim = getattr(config, "latent_dim", 256)
            param_dim = getattr(config, "param_dim", 0)

            model = PointNetAutoencoder(
                in_dim=n_features_dyn,
                out_dim=4,
                latent_dim=latent_dim,
                param_dim=param_dim,
            ).to(device)

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

        # If scaler was not already loaded above, load it now
        if not scaler_loaded:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            scaler = joblib.load(scaler_path)
            scaler_loaded = True
            print(f"[INFO] Loaded scaler from {scaler_path} for pretrained model.")

        # Consistency check
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != n_features_dyn:
            raise ValueError(
                f"Scaler n_features_in_={scaler.n_features_in_} does not match "
                f"expected dynamic feature count={n_features_dyn}"
            )

    else:
        # --------- Build model and train on train_scaled / params_all ---------
        with log_time("Building PyTorch PointNet autoencoder"):
            print("[INFO] Building PyTorch PointNet autoencoder...")
            latent_dim = getattr(config, "latent_dim", 256)
            param_dim = getattr(config, "param_dim", 0)

            model = PointNetAutoencoder(
                in_dim=n_features_dyn,
                out_dim=4,
                latent_dim=latent_dim,
                param_dim=param_dim,
            ).to(device)

        with log_time("Creating DataLoaders and training model"):
            if params_all is not None:
                X_train, X_val, p_train, p_val = train_test_split(
                    train_scaled,
                    params_all,
                    test_size=0.2,
                    random_state=42,
                    shuffle=True,
                )
                train_dataset = SnapshotParamDataset(X_train, p_train)
                val_dataset = SnapshotParamDataset(X_val, p_val)
            else:
                X_train, X_val = train_test_split(
                    train_scaled,
                    test_size=0.2,
                    random_state=42,
                    shuffle=True,
                )
                train_dataset = SnapshotDataset(X_train)
                val_dataset = SnapshotDataset(X_val)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
            )

            print("[INFO] Training PyTorch PointNet autoencoder...")
            model = train_pointnet_torch(
                model,
                train_loader,
                val_loader,
                device,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )

        # Save model + scaler
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Saved model to {model_path}")
        print(f"[INFO] Saved scaler to {scaler_path}")

        # TorchScript export
        try:
            model.eval()
            example_input = torch.randn(1, n_points, n_features_dyn, device=device)
            if param_dim > 0:
                example_param = torch.randn(1, param_dim, device=device)
                scripted = torch.jit.trace(model, (example_input, example_param))
            else:
                scripted = torch.jit.trace(model, example_input)

            script_path = model_path.replace(".pt", "_script.pt")
            scripted.save(script_path)
            print(f"[INFO] Saved TorchScript model to {script_path}")
        except Exception as e:
            print(f"[WARN] TorchScript export failed: {e}")

        trained_here = True

    # Ensure model is in eval mode for inference
    model.eval()

    # ==============================================================
    # 5. INFERENCE ON THE SAME SNAPSHOTS (USING SHARED DATA)
    # ==============================================================

    infer_dirs = config.rev_dirs
    print("[INFO] Using rev_dirs for Lagrangian inference/ROM output.")

    infer_times = train_times
    infer_data = train_data
    infer_scaled = train_scaled
    s_inf, n_points_inf, n_features_inf_total = infer_scaled.shape

    # Build parameter array for inference (reuses params_all)
    params_infer = params_all

    with log_time("Running Lagrangian inference on rev_dirs"):
        infer_batch_size = getattr(
            config, "infer_batch_size", getattr(config, "batch_size", 2)
        )
        preds_scaled_list = []

        with torch.no_grad():
            x_all = torch.from_numpy(infer_scaled).float().to(device)  # [S, N, 6]
            if params_infer is not None:
                p_all = torch.from_numpy(params_infer).float().to(device)  # [S, P]
            else:
                p_all = None

            for b_start in range(0, s_inf, infer_batch_size):
                x_b_full = x_all[b_start : b_start + infer_batch_size]   # [B, N, 6]
                x_b_dyn = x_b_full[..., :n_features_dyn]                 # [B, N, 4]

                if p_all is not None:
                    p_b = p_all[b_start : b_start + infer_batch_size]    # [B, P]
                    recon_b, _ = model(x_b_dyn, p_b)                     # [B, N, 4]
                else:
                    recon_b, _ = model(x_b_dyn)                          # [B, N, 4]

                preds_scaled_list.append(recon_b.cpu().numpy())

        preds_scaled = np.concatenate(preds_scaled_list, axis=0)  # [S, N, 4]

    # ==============================================================
    # 6. BASELINE + RESIDUAL COMPOSITION IN SCALED SPACE
    # ==============================================================

    use_baseline_infer = getattr(config, "lagrangian_use_baseline", False)
    dyn_scaled_full = preds_scaled  # default: pure ML prediction in scaled space

    if use_baseline_infer:
        print(f"[INFO] Using baseline from {baseline_path} for residual ROM.")
        baseline_dyn_scaled = np.load(baseline_path)

        if baseline_dyn_scaled.shape != preds_scaled.shape:
            raise ValueError(
                f"Baseline shape {baseline_dyn_scaled.shape} does not match "
                f"preds_scaled shape {preds_scaled.shape}."
            )

        # Model output is residual in scaled space, add the baseline back
        dyn_scaled_full = baseline_dyn_scaled + preds_scaled

    # ==============================================================
    # 7. INVERSE SCALING, CLIPPING, APPEND STATIC IDS
    # ==============================================================

    flat_pred = dyn_scaled_full.reshape(-1, n_features_dyn)         # [S*N, 4]
    flat_pred_unscaled = scaler.inverse_transform(flat_pred)        # [S*N, 4]

    # Clip using precomputed training bounds in physical space
    flat_pred_clipped = np.clip(flat_pred_unscaled, train_min, train_max)
    pred_dyn = flat_pred_clipped.reshape(preds_scaled.shape)        # [S, N, 4]

    # Append static IDs (CloudID, CloudID_base)
    ids = infer_data[..., 4:]                                       # [S, N, 2]
    pred_array = np.concatenate([pred_dyn, ids], axis=-1)           # [S, N, 6]

    # ==============================================================
    # 8. OPTIONAL GEOMETRY PROJECTION + WRITE ROM OUTPUT
    # ==============================================================

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

            s_inf, n_points_inf, _ = pred_array.shape

            print("[INFO] Projecting predicted particle coordinates into geometry volume...")
            for s in tqdm(range(s_inf), desc="Geometry projection (snapshots)", unit="snap"):
                xyz = pred_array[s, :, 0:3]
                xyz_proj = projector.project_points_inside(xyz)
                pred_array[s, :, 0:3] = xyz_proj

        columns_dir = Path(config.rev_dirs[0])
        write_lagrangian_rom_only(
            preds=pred_array,
            times=infer_times,
            columns_dir=columns_dir,
            out_dir=Path(rom_out_dir),
            field_var=field_var,
            train_min=train_min,
            train_max=train_max,
        )
