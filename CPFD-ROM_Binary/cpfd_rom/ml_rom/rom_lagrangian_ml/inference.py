# cpfd_rom/ml_rom/rom_lagrangian_ml/inference.py

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import SnapshotDataset


def predict_pointnet_torch(model, data_scaled, device, batch_size=2, params=None):
    """Run inference on scaled data and return reconstructed snapshots (numpy).

    data_scaled: [num_snaps, n_points, n_features]
    params: optional array [num_snaps, param_dim] for conditioning.
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

            recon, _ = model(x, params=p_batch)
            preds.append(recon.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds  # [num_snaps, n_points, n_features]


__all__ = ["predict_pointnet_torch"]
