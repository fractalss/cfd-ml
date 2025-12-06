# cpfd_rom/ml_rom/rom_lagrangian_ml/pipeline.py

import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For geometry projection loop

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import load_lagrangian_snapshots

from cpfd_rom.ml_rom.rom_lagrangian_ml.mlp_encoder import (
    PointNetAutoencoder,
)

from cpfd_rom.ml_rom.rom_lagrangian_ml.geometry_projection import GeometryProjector
from cpfd_rom.ml_rom.rom_lagrangian_ml.evaluation import write_lagrangian_rom_only

from cpfd_rom.ml_rom.rom_lagrangian_ml.training import (
    train_pointnet_torch,
)

from cpfd_rom.ml_rom.rom_lagrangian_ml.datasets import (
    build_params_from_rev_dirs,
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
# Minimal dataset: per-snapshot (x_dyn_scaled, params_aug)
# ---------------------------------------------------------------------------

class SnapshotParamDatasetLagrangian(Dataset):
    """
    Each item is a dict:
        {
            "x":      [N, 4]  dynamic features (scaled) [x, y, z, field],
            "params": [P]     augmented params [physical..., t_norm]
        }
    """

    def __init__(self, dyn_scaled: np.ndarray, params_aug: np.ndarray):
        """
        Parameters
        ----------
        dyn_scaled : np.ndarray [S, N, 4]
            Scaled dynamic fields per snapshot.
        params_aug : np.ndarray [S, P_aug]
            Augmented param vector per snapshot (physical params + t_norm).
        """
        if dyn_scaled.shape[0] != params_aug.shape[0]:
            raise ValueError(
                f"SnapshotParamDatasetLagrangian: dyn_scaled has {dyn_scaled.shape[0]} snapshots "
                f"but params_aug has {params_aug.shape[0]}"
            )
        if dyn_scaled.shape[2] != 4:
            raise ValueError(
                f"Expected dyn_scaled[...,4] for [x,y,z,field], got {dyn_scaled.shape}"
            )

        self.x = dyn_scaled.astype(np.float32)       # [S, N, 4]
        self.params = params_aug.astype(np.float32)  # [S, P_aug]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        x_i = torch.from_numpy(self.x[idx])        # [N, 4]
        p_i = torch.from_numpy(self.params[idx])   # [P_aug]
        return {"x": x_i, "params": p_i}


# ---------------------------------------------------------------------------
# Main pipeline: param+time-conditioned PointNet AE (no baseline)
# ---------------------------------------------------------------------------

def run_lagrangian_ml_pipeline(config, log_time):
    """Param+time conditioned PointNet autoencoder pipeline for Lagrangian ROM.

    Workflow:
      1) Load snapshots & physical params.
      2) Scale CFD dynamic features [x,y,z,field].
      3) Build augmented parameters per snapshot:
             params_aug = [params_phys, t_norm in [0,1]].
      4) Train PointNetAutoencoder:
             x_dyn_scaled | params_aug  ->  x_dyn_scaled  (reconstruction).
      5) Inference at config.user_parameter:
             - choose nearest training parameter group (rev),
             - reconstruct its snapshots with the AE,
             - inverse-scale, clip, write ROM for those snapshots.
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

    epochs = getattr(config, "epochs", 50)
    batch_size = getattr(config, "batch_size", 2)
    patience = getattr(config, "patience", 10)
    lr = getattr(config, "learning_rate", 1e-4)
    latent_dim = getattr(config, "latent_dim", 256)

    field_var = getattr(config, "field_variable", None)
    if field_var is None:
        raise ValueError(
            "[Lagrangian] config.field_variable must be set (e.g., 'pvf', 'speed')."
        )

    if not hasattr(config, "rev_dirs") or not config.rev_dirs:
        raise ValueError("[Lagrangian] config.rev_dirs must be provided.")

    # ==============================================================
    # 1. LOAD SNAPSHOTS & PHYSICAL PARAMS
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

    # Physical parameters per snapshot (no time yet)
    param_mapping = getattr(config, "param_mapping", None)
    if param_mapping is None:
        raise RuntimeError(
            "[Lagrangian] param-conditioned AE requires config.param_mapping."
        )

    params_all = build_params_from_rev_dirs(config.rev_dirs, param_mapping)
    if params_all.shape[0] != n_snaps:
        raise ValueError(
            f"params_all has {params_all.shape[0]} entries but train_data has {n_snaps} snapshots"
        )

    params_all = params_all.astype(np.float32)    # [S, P_phys]
    param_dim_phys = params_all.shape[1]
    config.param_dim_phys = param_dim_phys

    # Basic info about param groups (revs)
    unique_params, rev_indices = np.unique(
        params_all, axis=0, return_inverse=True
    )  # unique_params[G, P_phys], rev_indices[S]
    n_param_groups = unique_params.shape[0]
    print(f"[INFO] Found {n_param_groups} unique parameter vectors across snapshots.")

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

        flat_scaled = np.concatenate([flat_dyn_scaled, flat_ids], axis=1)
        train_scaled_full = flat_scaled.reshape(train_data.shape)  # [S, N, 6]

        train_dyn_scaled = flat_dyn_scaled.reshape(n_snaps, n_points, n_features_dyn)

        # bounds in physical space (for clipping)
        train_min = flat_dyn.min(axis=0)
        train_max = flat_dyn.max(axis=0)

    # ==============================================================
    # 3. BUILD AUGMENTED PARAMS: [params_phys, t_norm]
    # ==============================================================

    t_min = float(train_times.min())
    t_max = float(train_times.max())
    if t_max > t_min:
        t_norm = (train_times - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros_like(train_times)
    t_norm = t_norm.astype(np.float32).reshape(-1, 1)      # [S, 1]

    params_aug = np.concatenate([params_all, t_norm], axis=1).astype(np.float32)
    param_dim_aug = params_aug.shape[1]
    config.param_dim_aug = param_dim_aug

    print(
        f"[INFO] Built augmented params with shape {params_aug.shape} "
        f"(P_phys={param_dim_phys}, +1 time coord)."
    )

    # ==============================================================
    # 4. AE TRAINING: PointNetAutoencoder(x_dyn_scaled | params_aug)
    # ==============================================================

    skip_training = getattr(config, "skip_training", False)

    ae_model = PointNetAutoencoder(
        in_dim=4,
        out_dim=4,
        latent_dim=latent_dim,
        param_dim=param_dim_aug,   # condition on [params_phys, t_norm]
    ).to(device)

    if skip_training and os.path.exists(model_path):
        with log_time("Loading pretrained Lagrangian AE"):
            print(f"[INFO] Loading Lagrangian AE model from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            ae_model.load_state_dict(state_dict)
    else:
        full_dataset = SnapshotParamDatasetLagrangian(
            dyn_scaled=train_dyn_scaled,   # [S, N, 4]
            params_aug=params_aug,         # [S, P_aug]
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

        with log_time("Training PointNet AE (params+time-conditioned)"):
            print("[INFO] Training PointNet autoencoder...")
            ae_model = train_pointnet_torch(
                ae_model,
                train_loader,
                val_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )

        # save model + scaler
        torch.save(ae_model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Saved AE model to {model_path}")
        print(f"[INFO] Saved scaler to {scaler_path}")

        # optional TorchScript export
        try:
            ae_model.eval()
            example_x = torch.randn(1, n_points, 4, device=device)
            example_p = torch.randn(1, param_dim_aug, device=device)
            scripted = torch.jit.trace(ae_model, (example_x, example_p))
            scripted.save(script_path)
            print(f"[INFO] Saved TorchScript Lagrangian AE to {script_path}")
        except Exception as e:
            print(f"[WARN] TorchScript export failed: {e}")

    ae_model.eval()

    # ==============================================================
    # 5. INFERENCE AT user_parameter (nearest training param)
    # ==============================================================

    with log_time("Running Lagrangian AE inference at user_parameter"):
        n_revs = len(config.rev_dirs)
        snaps_per_rev = n_snaps // n_revs  # assume equal snapshots per rev

        # Choose nearest training param group to user_parameter
        infer_param = getattr(config, "user_parameter", None)
        if infer_param is None:
            raise ValueError(
                "[Lagrangian] config.user_parameter must be set for inference."
            )

        infer_param_vec = np.atleast_1d(infer_param).astype(np.float32)  # [P_phys]
        if infer_param_vec.shape[0] != param_dim_phys:
            raise ValueError(
                f"[Lagrangian] user_parameter dim {infer_param_vec.shape[0]} "
                f"does not match param_dim_phys {param_dim_phys}"
            )

        # Find nearest unique parameter (rev) in L2 sense
        dists = np.linalg.norm(unique_params - infer_param_vec[None, :], axis=1)
        best_group = int(np.argmin(dists))
        print(
            f"[INFO] Nearest training parameter group index = {best_group}, "
            f"distance = {dists[best_group]:.3e}"
        )

        # Assume rev_dirs are ordered such that each unique param corresponds
        # to a block of snapshots of length snaps_per_rev.
        if n_param_groups != n_revs:
            print(
                "[WARN] n_param_groups != n_revs; inference rev mapping may be approximate."
            )
        rev_idx = min(best_group, n_revs - 1)

        s0 = rev_idx * snaps_per_rev
        s1 = s0 + snaps_per_rev
        print(f"[INFO] Using snapshots [{s0}:{s1}] for inference.")

        # Inputs for this rev
        dyn_scaled_infer = train_dyn_scaled[s0:s1]   # [T, N, 4]
        params_aug_infer = params_aug[s0:s1]         # [T, P_aug]
        times_unique = train_times[s0:s1].copy()     # [T]
        ids = train_data[s0:s1, :, 4:]               # [T, N, 2]

        # Run AE
        preds_scaled_list = []
        infer_batch_size = getattr(config, "infer_batch_size", batch_size)

        with torch.no_grad():
            for b_start in range(0, dyn_scaled_infer.shape[0], infer_batch_size):
                x_b = torch.from_numpy(
                    dyn_scaled_infer[b_start:b_start + infer_batch_size]
                ).float().to(device)  # [B, N, 4]

                p_b = torch.from_numpy(
                    params_aug_infer[b_start:b_start + infer_batch_size]
                ).float().to(device)  # [B, P_aug]

                recon_b, _ = ae_model(x_b, params=p_b)  # [B, N, 4]
                preds_scaled_list.append(recon_b.cpu().numpy())

        dyn_scaled_full = np.concatenate(preds_scaled_list, axis=0)  # [T, N, 4]

    # ==============================================================
    # 6. INVERSE SCALING, CLIPPING, GEOMETRY, WRITE
    # ==============================================================

    flat_pred = dyn_scaled_full.reshape(-1, 4)
    flat_pred_unscaled = scaler.inverse_transform(flat_pred)
    flat_pred_clipped = np.clip(flat_pred_unscaled, train_min, train_max)
    pred_dyn = flat_pred_clipped.reshape(dyn_scaled_full.shape)      # [T, N, 4]

    pred_array = np.concatenate([pred_dyn, ids], axis=-1)            # [T, N, 6]

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
            times=times_unique,
            columns_dir=columns_dir,
            out_dir=Path(rom_out_dir),
            field_var=field_var,
            train_min=train_min,
            train_max=train_max,
        )
