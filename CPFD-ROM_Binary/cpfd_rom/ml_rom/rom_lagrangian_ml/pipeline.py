# cpfd_rom/ml_rom/rom_lagrangian_ml/pipeline.py

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from pathlib import Path

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import load_lagrangian_snapshots
from cpfd_rom.ml_rom.rom_lagrangian_ml.mlp_encoder import PointNetAutoencoder
from cpfd_rom.ml_rom.rom_lagrangian_ml.datasets import (
    SnapshotDataset,
    SnapshotParamDataset,
    build_params_from_rev_dirs,
)

from cpfd_rom.ml_rom.rom_lagrangian_ml.training import train_pointnet_torch
# from cpfd_rom.ml_rom.rom_lagrangian_ml.inference import predict_pointnet_torch  # optional

from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.ml_rom.rom_lagrangian_ml.geometry_projection import GeometryProjector
from cpfd_rom.ml_rom.rom_lagrangian_ml.evaluation import write_lagrangian_rom_only


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

    The StandardScaler is fit on the 4 dynamic features only.
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
    scaler = None
    model = None

    epochs = getattr(config, "epochs", 1)
    batch_size = getattr(config, "batch_size", 2)
    patience = getattr(config, "patience", 10)
    lr = getattr(config, "learning_rate", 1e-3)
    trained_here = False

    # Field variable to use as the scalar in the ROM
    field_var = getattr(config, "field_variable", None)
    if field_var is None:
        raise ValueError(
            "[Lagrangian] config.field_variable must be set (e.g., 'pvf', 'speed')."
        )

    # ---------------- TRAIN OR LOAD MODEL ----------------
    if getattr(config, "skip_training", False) and os.path.exists(model_path):
        # --------- Load pretrained model + scaler ---------
        with log_time("Loading pretrained PyTorch PointNet model"):
            print(f"[INFO] Loading pretrained model from {model_path}")
            param_dim = getattr(config, "param_dim", 0)
            latent_dim = getattr(config, "latent_dim", 256)

            # For Lagrangian ROM now: in_dim=4 ([x,y,z,field]), out_dim=4
            model = PointNetAutoencoder(
                in_dim=4,
                out_dim=4,
                latent_dim=latent_dim,
                param_dim=param_dim,
            ).to(device)

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

        scaler_path = model_path.replace(".pt", "_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Self-consistency check between config and scaler (on dynamic features only)
        if hasattr(scaler, "n_features_in_"):
            dyn_expected = getattr(config, "n_dynamic_features", 4)
            if scaler.n_features_in_ != dyn_expected:
                raise ValueError(
                    f"Scaler n_features_in_={scaler.n_features_in_} does not match "
                    f"expected dynamic feature count={dyn_expected}"
                )

    else:
        # --------- Load + scale training data from npy snapshots ---------
        with log_time("Loading and preprocessing Lagrangian npy data"):
            print("[INFO] Loading Lagrangian training data from npy...")
            train_times, train_data_raw = load_lagrangian_snapshots(config.rev_dirs)
            # train_data_raw: [num_snaps, n_points, n_features_raw]

            # Select the 6 ROM features using columns.txt
            train_data = _select_rom_features(train_data_raw, config.rev_dirs, field_var)
            n_snaps, n_points, n_features_total = train_data.shape  # n_features_total should now be 6

            # We will pass only the first 4 dynamic features into the network
            n_features_dyn = 4

            # Lock counts into config for downstream use
            config.n_features_total = n_features_total   # 6
            config.n_dynamic_features = n_features_dyn   # 4

            # Standardize only the first 4 (dynamic) features: x, y, z, field
            scaler = StandardScaler()
            flat = train_data.reshape(-1, n_features_total)  # [num_snaps * n_points, 6]
            flat_dyn = flat[:, :n_features_dyn]
            flat_ids = flat[:, n_features_dyn:]

            flat_dyn_scaled = scaler.fit_transform(flat_dyn)
            flat_scaled = np.concatenate([flat_dyn_scaled, flat_ids], axis=1)
            train_scaled = flat_scaled.reshape(train_data.shape)

            # --------- Build param array if mapping is provided ---------
            param_mapping = getattr(config, "param_mapping", None)
            params_all = None
            if param_mapping is not None:
                print("[INFO] Building parameter array from rev_dirs + param_mapping...")
                params_all = build_params_from_rev_dirs(config.rev_dirs, param_mapping)
                if params_all.shape[0] != train_scaled.shape[0]:
                    raise ValueError(
                        f"params_all has {params_all.shape[0]} entries but "
                        f"train_scaled has {train_scaled.shape[0]} snapshots"
                    )
                param_dim = params_all.shape[1]
                config.param_dim = param_dim
            else:
                param_dim = 0
                config.param_dim = param_dim

        # --------- Build model ---------
        with log_time("Building PyTorch PointNet autoencoder"):
            print("[INFO] Building PyTorch PointNet autoencoder...")
            latent_dim = getattr(config, "latent_dim", 256)
            param_dim = getattr(config, "param_dim", 0)

            # in_dim = 4 (dynamic features only), out_dim = 4
            model = PointNetAutoencoder(
                in_dim=n_features_dyn,
                out_dim=4,
                latent_dim=latent_dim,
                param_dim=param_dim,
            ).to(device)

        # --------- Dataloaders ---------
        with log_time("Creating DataLoaders and training model"):
            # Proper train/validation split
            if params_all is not None:
                # Split both data and params consistently
                X_train, X_val, p_train, p_val = train_test_split(
                    train_scaled,
                    params_all,
                    test_size=0.2,
                    random_state=42,
                    shuffle=True,
                )

                train_dataset = SnapshotParamDataset(X_train, p_train)
                val_dataset   = SnapshotParamDataset(X_val,   p_val)

            else:
                # Split only data
                X_train, X_val = train_test_split(
                    train_scaled,
                    test_size=0.2,
                    random_state=42,
                    shuffle=True,
                )

                train_dataset = SnapshotDataset(X_train)
                val_dataset   = SnapshotDataset(X_val)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
            )

            # --------- Train model ---------
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

        # --------- Save artifacts (state_dict + TorchScript) ---------
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, model_path.replace(".pt", "_scaler.pkl"))
        print(f"[INFO] Saved model to {model_path}")
        print(
            f"[INFO] Saved scaler to {model_path.replace('.pt', '_scaler.pkl')}"
        )

        # TorchScript export for CLI/runtime deployment
        try:
            model.eval()
            example_input = torch.randn(1, n_points, n_features_dyn, device=device)
            scripted = torch.jit.trace(model, example_input)
            script_path = model_path.replace(".pt", "_script.pt")
            scripted.save(script_path)
            print(f"[INFO] Saved TorchScript model to {script_path}")
        except Exception as e:
            print(f"[WARN] TorchScript export failed: {e}")

        # Mark that we have just trained the model and have train_times/train_scaled in memory
        trained_here = True

    # --------------- Inference + ROM writing on rev_dirs ---------------

    # By default, inference uses the same rev_dirs as training.
    # If you want to reconstruct only a subset of the training data
    # (e.g., a single Rev with known snapshots per Rev), you can
    # control this via config:
    #
    #   lagrangian_reconstruct_training_only: true
    #   lagrangian_snaps_per_rev: 1001
    #   lagrangian_reconstruct_rev_index: 0  # 0-based index among rev_dirs

    infer_dirs = config.rev_dirs
    reconstruct_training_only = getattr(
        config, "lagrangian_reconstruct_training_only", False
    )

    print("[INFO] Using rev_dirs for Lagrangian inference/ROM output.")

    with log_time("Running Lagrangian inference on rev_dirs"):
        subset_slice = slice(None)  # default = use all snapshots

        if trained_here:
            # We have train_times, train_scaled, train_data in memory from the
            # training branch above. Optionally restrict to a single Rev.
            if reconstruct_training_only:
                snaps_per_rev = getattr(config, "lagrangian_snaps_per_rev", None)
                if snaps_per_rev is None:
                    raise ValueError(
                        "lagrangian_snaps_per_rev must be set in config when "
                        "lagrangian_reconstruct_training_only=True."
                    )

                rev_index = getattr(config, "lagrangian_reconstruct_rev_index", 0)
                if not (0 <= rev_index < len(infer_dirs)):
                    raise ValueError(
                        f"lagrangian_reconstruct_rev_index={rev_index} is out of "
                        f"range for rev_dirs={infer_dirs}"
                    )

                start = rev_index * snaps_per_rev
                end = start + snaps_per_rev

                if end > train_scaled.shape[0]:
                    raise ValueError(
                        f"Requested snapshots {start}..{end-1} but only "
                        f"{train_scaled.shape[0]} total. Check lagrangian_snaps_per_rev "
                        f"and lagrangian_reconstruct_rev_index."
                    )

                subset_slice = slice(start, end)

                print(
                    f"[INFO] Reconstructing training Rev index {rev_index} "
                    f"(snapshots {start}..{end-1})"
                )

                infer_times = train_times[subset_slice]
                infer_scaled = train_scaled[subset_slice]
                infer_data = train_data[subset_slice]
                s_inf, n_points_inf, n_features_inf_total = infer_scaled.shape
            else:
                # Reuse entire training set for inference
                infer_times = train_times
                infer_scaled = train_scaled
                infer_data = train_data
                s_inf, n_points_inf, n_features_inf_total = infer_scaled.shape
        else:
            # skip_training path: load and scale from npy now
            print("[INFO] Loading Lagrangian data from npy for inference...")
            infer_times, infer_data_raw = load_lagrangian_snapshots(infer_dirs)

            # Slice to ROM features using the same columns.txt mapping
            infer_data = _select_rom_features(infer_data_raw, infer_dirs, field_var)
            s_inf, n_points_inf, n_features_inf_total = infer_data.shape

            # Split dynamic vs ID features
            n_features_dyn = 4
            flat_inf = infer_data.reshape(-1, n_features_inf_total)
            flat_inf_dyn = flat_inf[:, :n_features_dyn]
            flat_inf_ids = flat_inf[:, n_features_dyn:]

            flat_inf_dyn_scaled = scaler.transform(flat_inf_dyn)
            flat_inf_scaled = np.concatenate([flat_inf_dyn_scaled, flat_inf_ids], axis=1)
            infer_scaled = flat_inf_scaled.reshape(infer_data.shape)

        # Torch inference in batches
        # Build parameter array for inference if available
        param_mapping = getattr(config, "param_mapping", None)
        params_infer = None
        if param_mapping is not None:
            if trained_here:
                # Reuse params_all from training if available; slice if needed
                if "params_all" in locals() and params_all is not None:
                    params_infer = params_all[subset_slice]
                else:
                    params_infer = build_params_from_rev_dirs(infer_dirs, param_mapping)
            else:
                # skip_training path: build params for inference rev_dirs
                params_infer = build_params_from_rev_dirs(infer_dirs, param_mapping)

            if params_infer.shape[0] != s_inf:
                raise ValueError(
                    f"params_infer has {params_infer.shape[0]} entries but "
                    f"infer_scaled has {s_inf} snapshots"
                )

        model.eval()
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
                x_b_full = x_all[b_start : b_start + infer_batch_size]  # [B, N, 6]
                x_b_dyn = x_b_full[..., :4]                             # [B, N, 4]
                if p_all is not None:
                    p_b = p_all[b_start : b_start + infer_batch_size]   # [B, P]
                    recon_b, _ = model(x_b_dyn, p_b)                    # recon_b: [B, N, 4]
                else:
                    recon_b, _ = model(x_b_dyn)                         # recon_b: [B, N, 4]
                preds_scaled_list.append(recon_b.cpu().numpy())

        preds_scaled = np.concatenate(preds_scaled_list, axis=0)       # [S, N, 4]

        # DEBUG: inspect one training snapshot reconstruction BEFORE inverse scaling / clipping / writing
        if trained_here and reconstruct_training_only:
            # pick first snapshot of the subset
            s0 = 0
            orig_scaled = infer_scaled[s0]   # [N, 6]
            recon_scaled = preds_scaled[s0]  # [N, 4]

            # inverse transform dynamics ONLY
            orig_flat_dyn = orig_scaled[..., :4].reshape(-1, 4)
            recon_flat_dyn = recon_scaled.reshape(-1, 4)

            orig_dyn = scaler.inverse_transform(orig_flat_dyn).reshape(-1, 4)
            recon_dyn = scaler.inverse_transform(recon_flat_dyn).reshape(-1, 4)

            # Print basic stats so we see if x/z are frozen
            for i, name in enumerate(["x", "y", "z", field_var]):
                o_min, o_max = orig_dyn[:, i].min(), orig_dyn[:, i].max()
                r_min, r_max = recon_dyn[:, i].min(), recon_dyn[:, i].max()
                print(f"[DEBUG] {name}: orig [{o_min:.3e}, {o_max:.3e}]  "
                      f"recon [{r_min:.3e}, {r_max:.3e}]")

        # Inverse scale back to physical space for the 4 dynamic features
        flat_pred = preds_scaled.reshape(-1, 4)
        flat_pred_unscaled = scaler.inverse_transform(flat_pred)  # [S*N, 4]

        # Use training min/max explicitly for clipping
        if trained_here:
            # flat_dyn is available from the training branch
            train_min = flat_dyn.min(axis=0)
            train_max = flat_dyn.max(axis=0)
        else:
            # skip_training path: derive bounds from the inference data itself
            train_min = flat_inf_dyn.min(axis=0)
            train_max = flat_inf_dyn.max(axis=0)

        flat_pred_clipped = np.clip(flat_pred_unscaled, train_min, train_max)
        pred_dyn = flat_pred_clipped.reshape(preds_scaled.shape)  # [S, N, 4]

        # Append static IDs (CloudID, CloudID_base) from data in ROM feature space
        ids = infer_data[..., 4:]  # [S, N, 2]
        pred_array = np.concatenate([pred_dyn, ids], axis=-1)  # [S, N, 6]

    # ---------------- Write ROM outputs in particles*.txt format ----------------
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

            from tqdm import tqdm

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
