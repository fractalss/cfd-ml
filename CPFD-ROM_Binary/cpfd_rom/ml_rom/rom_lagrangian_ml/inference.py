# cpfd_rom/ml_rom/rom_lagrangian_ml/inference.py

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import SnapshotDataset

def predict_pointnet_torch(model, data_scaled, device, batch_size=2, params=None, scaler=None):
    """
    Run inference on scaled Lagrangian data using the PointNet autoencoder.

     Input x has 6 features: [x, y, z, field, CloudID, CloudID_base]
      Only the first 4 channels are used for prediction.

     Output recon has shape [B, N, 6]: [x, y, z, field, CloudID, CloudID_base]

    Parameters
    ----------
    model : torch.nn.Module
        A PointNetAutoencoder with in_dim=6, out_dim=4.
    data_scaled : np.ndarray
        Scaled input of shape [num_snaps, n_points, 6].
    device : torch.device
        PyTorch device.
    batch_size : int
        Batch size.
    params : np.ndarray or None
        Optional [num_snaps, param_dim] array for conditioning.
    scaler : StandardScaler or None
        Scaler fitted on training data (for inverse_transform and clipping).

    Returns
    -------
    final_output : np.ndarray
        Reconstructed [x, y, z, field, CloudID, CloudID_base] of shape [num_snaps, n_points, 6].
    """
    model.eval()
    dataset = SnapshotDataset(data_scaled)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

            recon, _ = model(x, params=p_batch)  # recon: [B, N, 4]
            preds.append(recon.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # [S, N, 4]

    if scaler is not None:
        flat = preds.reshape(-1, 4)
        preds_inv = scaler.inverse_transform(flat).reshape(preds.shape)
        # Clip each dynamic feature to training min/max
        preds_clipped = np.clip(preds_inv, scaler.data_min_[:4], scaler.data_max_[:4])
    else:
        preds_clipped = preds

    # Reattach CloudID and CloudID_base
    cloud_ids = data_scaled[..., 4:6]
    final_output = np.concatenate([preds_clipped, cloud_ids], axis=-1)  # [S, N, 6]

    return final_output


__all__ = ["predict_pointnet_torch"]