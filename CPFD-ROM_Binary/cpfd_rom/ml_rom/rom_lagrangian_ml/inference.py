# cpfd_rom/ml_rom/rom_lagrangian_ml/inference.py

import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from .evaluation import write_lagrangian_rom_only
from .datasets import GraphSnapshotDataset
from cpfd_rom.ml_rom.rom_lagrangian_ml.mlp_encoder import PointNetAutoencoder


def predict_pointnet_torch(
    model,
    data_scaled,
    device,
    batch_size=2,
    params=None,
    feature_stats=None
):
    """
    Direct prediction using PointNet autoencoder on scaled input data.

    Args:
        model (torch.nn.Module): Trained autoencoder model.
        data_scaled (np.ndarray): Scaled input particle data [S, N, 6].
        device (torch.device): Torch device.
        batch_size (int): Batch size for inference.
        params (np.ndarray): Optional conditioning parameters [S, D].
        feature_stats (dict): Normalization statistics.

    Returns:
        np.ndarray: Denormalized predicted output [S, N, 6].
    """
    model.eval()
    dataset = GraphSnapshotDataset(data_scaled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch.to(device)

            if params is not None:
                start = i * batch_size
                end = start + x.shape[0]
                p_batch = torch.from_numpy(params[start:end]).float().to(device)
            else:
                p_batch = None

            recon = model(x, params=p_batch)
            preds.append(recon.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    if feature_stats is not None:
        pos_mean = feature_stats["pos_mean"].cpu().numpy()
        pos_std = feature_stats["pos_std"].cpu().numpy()
        field_min = feature_stats["field_min"]
        field_max = feature_stats["field_max"]

        pos = preds[..., :3] * (pos_std + 1e-8) + pos_mean
        field = preds[..., 3:4] * (field_max - field_min + 1e-8) + field_min
        preds_denorm = np.concatenate([pos, field], axis=-1)
    else:
        preds_denorm = preds

    cloud_ids = data_scaled[..., 4:6]
    final_output = np.concatenate([preds_denorm, cloud_ids], axis=-1)

    return final_output

def infer_pointnet_with_latent_regression(
    user_param: float,
    model: torch.nn.Module,
    latent_regressor: torch.nn.Module,
    scaffold_graphs: list,
    output_dir: str,
    field_variable: str,
    feature_stats: dict = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 16,
):
    """
    Inference using trained autoencoder and latent regressor with output denormalization.

    Args:
        user_param (float): New parameter value for prediction.
        model (torch.nn.Module): Trained PointNet autoencoder.
        latent_regressor (torch.nn.Module): Trained latent regressor.
        scaffold_graphs (list): List of graphs with spatial templates.
        output_dir (str): Path to save results.
        field_variable (str): Field variable name for saving output.
        feature_stats (dict): Normalization statistics.
        device (torch.device): Torch device.
        batch_size (int): Batch size for inference.

    Returns:
        np.ndarray: Denormalized predicted particle states [S, N_i, 6].
        list: List of simulation times.
    """
    model = model.to(device).eval()
    latent_regressor = latent_regressor.to(device).eval()

    loader = DataLoader(scaffold_graphs, batch_size=1, shuffle=False)
    preds, times = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            t_norm = batch.params[:, -1]
            user_param_tensor = torch.full_like(t_norm, fill_value=user_param)
            p_aug = torch.stack([user_param_tensor, t_norm], dim=1)

            z_pred = latent_regressor(p_aug)
            recon = model.decode(z_pred, batch.x[:, :4], batch.batch, p_aug)  # [N, 4]

            pos = recon[:, :3]
            field = recon[:, 3:4]

            if feature_stats is not None:
                pos = pos * (feature_stats["pos_std"].to(device) + 1e-8) + feature_stats["pos_mean"].to(device)
                field = field * (feature_stats["field_max"] - feature_stats["field_min"] + 1e-8) + feature_stats["field_min"]

            cloud_ids = batch.x[:, 4:6]
            output = torch.cat([pos, cloud_ids, field], dim=1)  # [N, 6]

            preds.append(output.cpu().numpy())
            times.append(batch.time.item())

    preds_np = np.stack(preds, axis=0)
    return preds_np, times
