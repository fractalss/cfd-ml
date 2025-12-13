import os
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool
from sklearn.model_selection import train_test_split

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import (
    load_lagrangian_snapshots_as_graphs,
    compute_feature_stats,
    extract_scaffold_graphs,
)

from cpfd_rom.ml_rom.rom_lagrangian_ml.mlp_encoder import PointNetAutoencoder
from cpfd_rom.ml_rom.rom_lagrangian_ml.datasets import GraphSnapshotDataset
from cpfd_rom.ml_rom.rom_lagrangian_ml.training import train_pointnet_torch
from cpfd_rom.ml_rom.rom_lagrangian_ml.latent_regressor import (
    LatentRegressorMLP,
    build_latent_regression_dataloaders,
    train_latent_regressor_torch,
)
from cpfd_rom.ml_rom.rom_lagrangian_ml.inference import infer_pointnet_with_latent_regression
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

    with log_time("Loading and normalizing graph snapshots (1-pass)"):
        graphs = load_lagrangian_snapshots_as_graphs(
            rev_dirs=config.rev_dirs,
            base_data_dir=config.base_data_dir,
            param_mapping=config.param_mapping,
            field_variable=config.field_variable,
            radius=getattr(config, "graph_radius", 0.01),
            sample_ratio=0.01,
            feature_stats=None,
        )

        feature_stats = compute_feature_stats(graphs)

        for g in graphs:
            if "x" in g and feature_stats.get("pos_mean") is not None:
                g.x[:, :3] = (g.x[:, :3] - feature_stats["pos_mean"]) / (feature_stats["pos_std"] + 1e-8)
            if "x" in g and feature_stats.get("field_min") is not None:
                g.x[:, 3:4] = (g.x[:, 3:4] - feature_stats["field_min"]) / (
                        feature_stats["field_max"] - feature_stats["field_min"] + 1e-8
                )
            if "y" in g:
                g.y[:, :3] = (g.y[:, :3] - feature_stats["pos_mean"]) / (feature_stats["pos_std"] + 1e-8)
                g.y[:, 3:4] = (g.y[:, 3:4] - feature_stats["field_min"]) / (
                        feature_stats["field_max"] - feature_stats["field_min"] + 1e-8
                )

    dataset = GraphSnapshotDataset(graphs)
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)

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

    sample = dataset[0]
    param_dim = sample.params.shape[1]

    model = PointNetAutoencoder(
        in_dim=4,
        param_dim=param_dim,
        latent_dim=latent_dim,
        out_dim=4,
    )

    if getattr(config, "skip_training", False) and os.path.exists(model_path):
        print(f"[INFO] Loading pretrained AE model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        with log_time("Training PointNet Autoencoder"):
            model = train_pointnet_torch(
                model=model,
                dataset=train_set,
                device=device,
                epochs=epochs,
                lr=learning_rate,
                batch_size=batch_size,
            )
        torch.save(model.state_dict(), model_path)

    model.eval()

    latents, params_all = [], []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=batch_size):
            x_feat = model.encoder_mlp(data.x.to(device))
            latent = global_max_pool(x_feat, data.batch.to(device))
            latents.append(latent.cpu())
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
    scaffold_graphs = extract_scaffold_graphs(
        rev_dirs=config.rev_dirs,
        base_data_dir=config.base_data_dir,
        param_mapping=config.param_mapping,
        param_train_array=params_all,
        user_param_array=config.user_parameter,
        field_variable=config.field_variable,
        radius=getattr(config, "graph_radius", 0.01),
        feature_stats=feature_stats,
        sample_ratio=0.01
    )

    preds, times = infer_pointnet_with_latent_regression(
        user_param=config.user_parameter,
        model=model,
        latent_regressor=latent_reg_model,
        scaffold_graphs=scaffold_graphs,
        output_dir=config.output_dir,
        field_variable=config.field_variable,
        feature_stats=feature_stats,
        device=device
    )

    out_dir = os.path.join(config.output_dir, "Lagrangian_ROM")
    os.makedirs(out_dir, exist_ok=True)

    write_lagrangian_rom_only(preds, times, config.field_variable, out_dir)
    print(f"[INFO] ROM inference complete. Output saved to: {out_dir}")
