# cpfd_rom/ml_rom/rom_lagrangian_ml/training.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

def _forward_pointnet(
    model: nn.Module,
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unified forward helper for PointNet autoencoder.

    Works for both dataset types:
      - SnapshotDataset:       batch is a Tensor [B, n_points, n_features_total]
      - SnapshotParamDataset:  batch is a dict {"x": Tensor, "params": Tensor, ...}

    The network operates only on the first 4 dynamic features:
        [x, y, z, field]

    If the batch contains a key "baseline", it is interpreted as a
    per-point baseline for these 4 channels in the *same scaled space*
    as the inputs, and the model output is treated as a residual to be
    added to this baseline.

    Returns
    -------
    recon : torch.Tensor
        Model output for the 4 dynamic channels, interpreted as either
        full-state or residual depending on whether a baseline is
        provided. Shape [B, N, 4].
    x_full : torch.Tensor
        Full input tensor as provided by the DataLoader, including IDs
        etc. Shape [B, N, n_features_total].
    baseline_dyn : Optional[torch.Tensor]
        Baseline for the 4 dynamic channels, or None if not provided
        in the batch. Shape [B, N, 4] if present.
    """
    baseline_dyn: Optional[torch.Tensor] = None

    if isinstance(batch, dict):
        x_full = batch["x"].to(device)  # [B, N, n_features_total]
        params = batch.get("params")
        if params is not None:
            params = params.to(device)

        # Optional baseline in scaled space
        if "baseline" in batch:
            baseline_full = batch["baseline"].to(device)  # [B, N, >=4]
            baseline_dyn = baseline_full[..., :4]          # [B, N, 4]

        x_dyn = x_full[..., :4]                            # [B, N, 4]
        if params is not None:
            out = model(x_dyn, params=params)
        else:
            out = model(x_dyn)
    else:
        # Plain tensor batch with no params / baseline
        x_full = batch.to(device)                          # [B, N, n_features_total]
        x_dyn = x_full[..., :4]                            # [B, N, 4]
        out = model(x_dyn)

    if isinstance(out, tuple):
        recon = out[0]
    else:
        recon = out

    return recon, x_full, baseline_dyn

def train_pointnet_torch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
) -> nn.Module:
    """
    Train a PointNet-like autoencoder with a clean, symmetric
    train/val loop. Only the first 4 input features are used for loss:

        MSE(recon_scaled, x_scaled[..., :4]).mean()

    CloudID and CloudID_base are ignored in loss.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss: Optional[float] = None
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch:03d}] train", leave=False):
            optimizer.zero_grad()

            # Forward pass: may include optional baseline in scaled space
            recon, x, baseline_dyn = _forward_pointnet(model, batch, device)

            # Target CFD state (scaled) for the 4 dynamic channels
            x_target = x[..., :recon.shape[2]]  # [B, N, 4]

            # If a baseline is provided, interpret recon as residual
            # and add it to the baseline in scaled space. Otherwise
            # recon is the full-state prediction.
            if baseline_dyn is not None:
                pred_dyn = baseline_dyn + recon
            else:
                pred_dyn = recon

            loss = F.mse_loss(pred_dyn, x_target)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch:03d}] val", leave=False):
                recon, x, baseline_dyn = _forward_pointnet(model, batch, device)
                x_target = x[..., :recon.shape[2]]  # [B, N, 4]

                if baseline_dyn is not None:
                    pred_dyn = baseline_dyn + recon
                else:
                    pred_dyn = recon

                loss = F.mse_loss(pred_dyn, x_target)
                batch_size = x.size(0)
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4e}, "
            f"val_loss={val_loss:.4e}"
        )

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[INFO] Early stopping at epoch {epoch} "
                    f"(no improvement in {patience} epochs)."
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model
