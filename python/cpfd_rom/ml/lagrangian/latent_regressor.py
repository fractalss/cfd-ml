from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)


class LatentRegressorMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        dims = [in_dim] + hidden_dims + [latent_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


LatentRegressor = LatentRegressorMLP


def build_latent_regression_dataloaders(
    params_aug,
    latents,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
):
    """Build train/validation dataloaders for latent regression.

    Args:
        params_aug: Array with shape ``[S, P]``.
        latents: Array with shape ``[S, Z]``.
    """
    params_aug = np.asarray(params_aug, dtype=np.float32)
    latents = np.asarray(latents, dtype=np.float32)

    if params_aug.ndim != 2:
        raise ValueError(f"params_aug must be 2D [S,P], got {params_aug.shape}")
    if latents.ndim != 2:
        raise ValueError(f"latents must be 2D [S,Z], got {latents.shape}")
    if params_aug.shape[0] != latents.shape[0]:
        raise ValueError(
            "params_aug and latents must have same number of samples, got "
            f"{params_aug.shape[0]} and {latents.shape[0]}"
        )

    x = torch.tensor(params_aug, dtype=torch.float32)
    y = torch.tensor(latents, dtype=torch.float32)
    dataset = TensorDataset(x, y)

    n_total = len(dataset)
    n_val = max(1, int(round(val_fraction * n_total)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(
            f"Validation split too large for dataset size {n_total}. "
            f"Got val_fraction={val_fraction}."
        )

    generator = torch.Generator().manual_seed(random_state)
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def train_latent_regressor_torch(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
):
    """Train a latent regressor that maps augmented parameters to latents."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            train_loss += float(loss.item()) * batch_size
            n_train += batch_size

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)

                batch_size = xb.shape[0]
                val_loss += float(loss.item()) * batch_size
                n_val += batch_size

        val_loss /= max(n_val, 1)

        detail(
            logger,
            "[LatentReg][Epoch %d/%d] train=%.6f, val=%.6f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                logger.info(
                    "[LatentReg] Early stopping at epoch %d",
                    epoch + 1,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


__all__ = [
    "LatentRegressorMLP",
    "LatentRegressor",
    "build_latent_regression_dataloaders",
    "train_latent_regressor_torch",
]
