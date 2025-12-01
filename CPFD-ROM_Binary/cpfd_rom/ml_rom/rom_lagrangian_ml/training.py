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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unified forward helper that works for both:
      - SnapshotDataset: batch is a Tensor [B, n_points, n_features]
      - SnapshotParamDataset: batch is a dict {"x": Tensor, "params": Tensor}
    """
    if isinstance(batch, dict):
        # Param-conditioned case
        x = batch["x"].to(device)            # [B, n_points, n_features]
        params = batch["params"].to(device)  # [B, param_dim]
        out = model(x, params=params)
    else:
        # No params
        x = batch.to(device)
        out = model(x)

    # Many encoders return (recon, z, ); we only need recon here.
    if isinstance(out, tuple):
        recon = out[0]
    else:
        recon = out

    return recon, x





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
    train/val loop. Both train and val losses are:

        MSE(recon_scaled, x_scaled).mean()

    i.e., they operate on the same scaled space.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss: Optional[float] = None
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in tqdm(train_loader, desc=f"[Epoch {epoch:03d}] train", leave=False):
            optimizer.zero_grad()

            recon, x = _forward_pointnet(model, batch, device)

            # Ensure shapes match on the feature dimension.
            # The model may only reconstruct a subset of the input features.
            if recon.shape != x.shape:
                if x.shape[0:2] == recon.shape[0:2] and x.shape[2] >= recon.shape[2]:
                    x_target = x[..., : recon.shape[2]]
                else:
                    raise ValueError(
                        f"Shape mismatch in train loop: recon {recon.shape}, x {x.shape}"
                    )
            else:
                x_target = x

            # MSE over all elements (mean)
            loss = F.mse_loss(recon, x_target)

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch:03d}] val", leave=False):
                recon, x = _forward_pointnet(model, batch, device)

                if recon.shape != x.shape:
                    if x.shape[0:2] == recon.shape[0:2] and x.shape[2] >= recon.shape[2]:
                        x_target = x[..., : recon.shape[2]]
                    else:
                        raise ValueError(
                            f"Shape mismatch in val loop: recon {recon.shape}, x {x.shape}"
                        )
                else:
                    x_target = x

                loss = F.mse_loss(recon, x_target)

                batch_size = x.size(0)
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4e}, "
            f"val_loss={val_loss:.4e}"
        )

        # ---------------- EARLY STOPPING ----------------
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
