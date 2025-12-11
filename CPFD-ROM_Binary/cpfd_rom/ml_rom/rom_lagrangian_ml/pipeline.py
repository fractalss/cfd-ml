import os
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import load_lagrangian_snapshots_as_graphs
from cpfd_rom.ml_rom.rom_lagrangian_ml.models.pointnet import PointNetAutoencoder
from cpfd_rom.ml_rom.rom_lagrangian_ml.datasets import GraphSnapshotDataset
from cpfd_rom.ml_rom.rom_lagrangian_ml.training import train_pointnet_torch
from cpfd_rom.ml_rom.rom_lagrangian_ml.latent_regressor import (
    LatentRegressorMLP,
    build_latent_regression_dataloaders,
    train_latent_regressor_torch,
)
from cpfd_rom.ml_rom.rom_lagrangian_ml.evaluation import write_lagrangian_rom_only
from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.util.model_utils import setup_model_paths


def run_lagrangian_ml_pipeline(config, log_time):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    setup_output_dir(config)
    setup_model_paths(config)

    model_path = os.path.join(config.model_dir, "pointnet_lagrangian.pt")
    latent_reg_path = os.path.join(config.model_dir, "latent_regressor.pt")

    with log_time("Loading Lagrangian PyG graph snapshots"):
        graphs = load_lagrangian_snapshots_as_graphs(
            rev_dirs=config.rev_dirs,
            base_data_dir=config.base_data_dir,
            param_mapping=config.param_mapping,
            field_variable=config.field_variable,
            radius=getattr(config, "graph_radius", 0.01),
            sample_ratio=0.01,
        )

    dataset = GraphSnapshotDataset(graphs)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    batch_size = getattr(config, "batch_size", 32)
    latent_dim = getattr(config, "latent_dim", 64)
    epochs = getattr(config, "epochs", 10)
    learning_rate = getattr(config, "learning_rate", 1e-3)
    patience = getattr(config, "patience", 3)
    latent_reg_epochs = getattr(config, "latent_reg_epochs", 50)
    latent_reg_lr = getattr(config, "latent_reg_lr", 1e-3)
    latent_reg_weight_decay = getattr(config, "latent_reg_weight_decay", 1e-4)
    latent_reg_patience = getattr(config, "latent_reg_patience", 5)
    latent_reg_hidden_dims = getattr(config, "latent_reg_hidden_dims", [128, 64])
    user_parameter = getattr(config, "user_parameter", None)
    infer_timesteps = getattr(config, "infer_timesteps", 10)


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    sample = dataset[0]
    param_dim = sample.params.shape[0]


    model = PointNetAutoencoder(input_dim=1, param_dim=param_dim, latent_dim=latent_dim).to(device)

    if getattr(config, "skip_training", False) and os.path.exists(model_path):
        print(f"[INFO] Loading pretrained AE model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        with log_time("Training PointNet Autoencoder"):
            model = train_pointnet_torch(
                model, train_loader, val_loader,
                epochs=epochs,
                lr=learning_rate,
            )
        torch.save(model.state_dict(), model_path)

    model.eval()

    # Latent regression phase
    latents, params_all = [], []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=config.batch_size):
            data = data.to(device)
            _, z = model(data.x, data.edge_index, data.edge_attr, data.batch, data.params)
            latents.append(z.cpu())
            params_all.append(data.params.cpu())

    latents_all = torch.cat(latents, dim=0).numpy()
    params_all = torch.cat(params_all, dim=0).numpy()

    train_loader_reg, val_loader_reg = build_latent_regression_dataloaders(
        params_aug=params_all,
        latents=latents_all,
        batch_size=batch_size,
        val_fraction=0.2,
        random_state=42,
        shuffle=True,
    )

    latent_reg_model = LatentRegressorMLP(
        in_dim=param_dim,
        latent_dim=latent_dim,
        hidden_dims=latent_reg_hidden_dims
    ).to(device)

    if getattr(config, "skip_training", False) and os.path.exists(latent_reg_path):
        print(f"[INFO] Loading pretrained latent regressor from {latent_reg_path}")
        latent_reg_model.load_state_dict(torch.load(latent_reg_path, map_location=device))
    else:
        with log_time("Training latent regressor"):
            latent_reg_model = train_latent_regressor_torch(
                latent_reg_model,
                train_loader_reg,
                val_loader_reg,
                device=device,
                epochs=latent_reg_epochs,
                lr=latent_reg_lr,
                weight_decay=latent_reg_weight_decay,
                patience=latent_reg_patience,
            )
        torch.save(latent_reg_model.state_dict(), latent_reg_path)

    latent_reg_model.eval()

    # Inference at user_parameter
    infer_param = np.atleast_1d(config.user_parameter).astype(np.float32)
    if infer_param.shape[0] != param_dim - 1:
        raise ValueError("user_parameter dimensionality mismatch")

    # Pick representative snapshot set from one Rev
    ref_graphs = [g for g in graphs if torch.allclose(g.params[:-1], torch.tensor(infer_param, dtype=torch.float32), atol=1e-2)]
    if not ref_graphs:
        ref_graphs = graphs[:config.infer_timesteps]  # fallback to any Rev

    pred_graphs = []
    with torch.no_grad():
        for g in ref_graphs:
            t = g.params[-1].item()
            p_aug = torch.tensor(np.append(infer_param, t), dtype=torch.float).unsqueeze(0).to(device)
            z_pred = latent_reg_model(p_aug)
            template = g.x.unsqueeze(0).to(device)
            recon = model.decode(z_pred, template, params=p_aug)
            out = recon.squeeze(0).cpu().numpy()

            # re-attach IDs and metadata
            pred_arr = torch.cat([recon.squeeze(0).cpu(), g.cloud_id.unsqueeze(1), g.cloud_id_base.unsqueeze(1)], dim=1).numpy()
            pred_graphs.append((g.snapshot_name, pred_arr, t))

    out_dir = os.path.join(config.rom_output_dir, "Lagrangian_ROM")
    os.makedirs(out_dir, exist_ok=True)
    write_lagrangian_rom_only(pred_graphs, out_dir, config.field_variable)
    print(f"[INFO] ROM inference complete. Output saved to: {out_dir}")
