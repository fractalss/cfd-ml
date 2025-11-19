# cpfd_rom/ml_rom/rom_lagrangian_ml/pipeline.py

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

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
from cpfd_rom.ml_rom.rom_lagrangian_ml.evaluation import write_lagrangian_rom_only

def run_lagrangian_ml_pipeline(config, log_time):
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

    epochs = getattr(config, "epochs", 5)
    batch_size = getattr(config, "batch_size", 2)
    patience = getattr(config, "patience", 10)
    lr = getattr(config, "learning_rate", 1e-3)
    trained_here = False

    # ---------------- TRAIN OR LOAD MODEL ----------------
    if getattr(config, "skip_training", False) and os.path.exists(model_path):
        with log_time("Loading pretrained PyTorch PointNet model"):
            print(f"[INFO] Loading pretrained model from {model_path}")
            param_dim = getattr(config, "param_dim", 0)
            model = PointNetAutoencoder(
                in_dim=config.n_features,
                out_dim=config.n_features,
                latent_dim=config.latent_dim,
                param_dim=param_dim,
            ).to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

        scaler_path = model_path.replace(".pt", "_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Self-consistency check between config and scaler
        if hasattr(scaler, "n_features_in_") and hasattr(config, "n_features"):
            if scaler.n_features_in_ != config.n_features:
                raise ValueError(
                    f"Scaler n_features_in_={scaler.n_features_in_} does not match "
                    f"config.n_features={config.n_features}"
                )

    else:
        # --------- Load + scale training data from npy snapshots ---------
        with log_time("Loading and preprocessing Lagrangian npy data"):
            print("[INFO] Loading Lagrangian training data from npy...")
            train_times, train_data = load_lagrangian_snapshots(config.rev_dirs)
            # train_data: [num_snaps, n_points, n_features]
            n_snaps, n_points, n_features = train_data.shape

            # Optional config/data consistency check
            if hasattr(config, "n_features") and config.n_features != n_features:
                raise ValueError(
                    f"Config n_features={config.n_features} but training data has "
                    f"n_features={n_features}"
                )
            else:
                # Lock n_features into config for downstream use (e.g., skip_training)
                config.n_features = n_features

            scaler = StandardScaler()
            flat = train_data.reshape(-1, n_features)
            flat_scaled = scaler.fit_transform(flat)
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
            latent_dim = getattr(config, "latent_dim", 128)
            param_dim = getattr(config, "param_dim", 0)
            model = PointNetAutoencoder(
                in_dim=n_features,
                out_dim=n_features,
                latent_dim=latent_dim,
                param_dim=param_dim,
            ).to(device)

        # --------- Dataloaders ---------
        with log_time("Creating DataLoaders and training model"):
            if "params_all" in locals() and params_all is not None:
                # Split both data and params consistently
                X_train, X_val, p_train, p_val = train_test_split(
                    train_scaled, params_all, test_size=0.2, random_state=42
                )
                train_dataset = SnapshotParamDataset(X_train, p_train)
                val_dataset = SnapshotParamDataset(X_val, p_val)
            else:
                X_train, X_val = train_test_split(
                    train_scaled, test_size=0.2, random_state=42
                )
                train_dataset = SnapshotDataset(X_train)
                val_dataset = SnapshotDataset(X_val)

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
                example_input = torch.randn(1, n_points, n_features, device=device)
                scripted = torch.jit.trace(model, (example_input,))
                script_path = model_path.replace(".pt", "_script.pt")
                scripted.save(script_path)
                print(f"[INFO] Saved TorchScript model to {script_path}")
            except Exception as e:
                print(f"[WARN] TorchScript export failed: {e}")

            # Mark that we have just trained the model and have train_times/train_scaled in memory
            trained_here = True

    # --------------- Inference + ROM writing on rev_dirs ---------------
    from pathlib import Path

    # We always run inference on the same rev_dirs used for training,
    # unless you later extend this to accept a separate inference set.
    infer_dirs = config.rev_dirs
    print("[INFO] Using rev_dirs for Lagrangian inference/ROM output.")

    with log_time("Running Lagrangian inference on rev_dirs"):
        if trained_here:
            # We already have scaled training data; reuse it for inference
            infer_times = train_times
            infer_scaled = train_scaled
            s_inf, n_points_inf, n_features_inf = infer_scaled.shape
        else:
            # skip_training path: load and scale from npy now
            print("[INFO] Loading Lagrangian data from npy for inference...")
            infer_times, infer_data = load_lagrangian_snapshots(infer_dirs)
            s_inf, n_points_inf, n_features_inf = infer_data.shape

            if n_features_inf != config.n_features:
                raise ValueError(
                    f"Inference data n_features={n_features_inf} does not match "
                    f"config.n_features={config.n_features}"
                )

            flat_inf = infer_data.reshape(-1, n_features_inf)
            flat_inf_scaled = scaler.transform(flat_inf)
            infer_scaled = flat_inf_scaled.reshape(infer_data.shape)

        # Torch inference in batches
        model.eval()
        infer_batch_size = getattr(
            config, "infer_batch_size", getattr(config, "batch_size", 2)
        )
        preds_scaled_list = []
        with torch.no_grad():
            x_all = torch.from_numpy(infer_scaled).float().to(device)
            for b_start in range(0, s_inf, infer_batch_size):
                x_b = x_all[b_start : b_start + infer_batch_size]
                y_b = model(x_b).cpu().numpy()  # [B, n_points, n_features]
                preds_scaled_list.append(y_b)

        preds_scaled = np.concatenate(preds_scaled_list, axis=0)

        # Inverse scale back to physical feature space
        flat_pred = preds_scaled.reshape(-1, n_features_inf)
        flat_pred_unscaled = scaler.inverse_transform(flat_pred)
        pred_array = flat_pred_unscaled.reshape(preds_scaled.shape)

    # ---------------- Write ROM outputs in particles*.txt format ----------------
    with log_time("Writing Lagrangian ROM output"):
        # Determine output directory
        rom_out_dir = getattr(
            config,
            "rom_output_dir_lagrangian",
            None,
        )
        if rom_out_dir is None:
            base_rom_dir = getattr(config, "rom_output_dir", "rom_output")
            rom_out_dir = os.path.join(base_rom_dir, "Lagrangian_ROM")

        print(f"[INFO] Writing Lagrangian ROM snapshots to: {rom_out_dir}")

        # Use the first rev_dir as the schema source (columns.txt).
        # Assumes all Rev*_npy have the same columns, which they should.
        columns_dir = Path(config.rev_dirs[0])

        write_lagrangian_rom_only(
            preds=pred_array,
            times=infer_times,
            columns_dir=columns_dir,
            out_dir=Path(rom_out_dir),
        )

