# cpfd_rom/ml_rom/rom_lagrangian_ml/inference.py

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import SnapshotDataset


def predict_pointnet_torch(model, data_scaled, device, batch_size=2, params=None):
    """Run inference on scaled Lagrangian data using the PointNet autoencoder.

    This helper is aligned with the current Lagrangian ROM logic:

     Per-point scaled input x has 6 features:
        [x, y, z, field, CloudID, CloudID_base]

      where only the first 4 channels are dynamically scaled with
      StandardScaler and used as prediction targets. The last 2 are
      static IDs carried through in the pipeline (not predicted).

     The PointNetAutoencoder is configured with out_dim = 4, so the
      reconstructed output `recon` has shape [B, N, 4] corresponding to
      [x, y, z, field] in scaled space.

    Parameters
    ----------
    model : torch.nn.Module
        A PointNetAutoencoder configured with in_dim=6, out_dim=4.
    data_scaled : np.ndarray
        Scaled input data of shape [num_snaps, n_points, 6].
    device : torch.device
        Device on which to run inference.
    batch_size : int, optional
        Batch size for the DataLoader.
    params : np.ndarray or None, optional
        Optional array of shape [num_snaps, param_dim] providing
        conditioning parameters per snapshot. If provided, they will
        be batched in sync with the snapshots and passed to the model
        as `params`.

    Returns
    -------
    preds : np.ndarray
        Reconstructed dynamic fields with shape [num_snaps, n_points, 4]
        in scaled space corresponding to [x, y, z, field]. It is the
        caller's responsibility to inverse-transform these 4 channels
        with the StandardScaler and to append the static IDs from the
        original data if needed for ROM file writing.
    """
    model.eval()
    dataset = SnapshotDataset(data_scaled)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch: [B, N, 6]
            x = batch.to(device)

            if params is not None:
                start = i * batch_size
                end = start + x.shape[0]
                p_batch = torch.from_numpy(params[start:end]).float().to(device)
            else:
                p_batch = None

            recon, _ = model(x, params=p_batch)  # recon: [B, N, 4]
            preds.append(recon.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds  # [num_snaps, n_points, 4]


__all__ = ["predict_pointnet_torch"]
