# cpfd_rom/ml_rom/rom_lagrangian_ml/latent_regressor.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Simple dataset: (params_aug -> latent code)
# ---------------------------------------------------------------------------

class LatentRegressionDataset(Dataset):
    """
    Dataset for latent regression:

        params_aug[s] : [P]       (e.g. [param_phys, t_norm])
        latents[s]    : [Z]       (PointNet encoder output)

    Shapes:
        params_aug : np.ndarray [S, P]
        latents    : np.ndarray [S, Z]
    """

    def __init__(self, params_aug: np.ndarray, latents: np.ndarray):
        if params_aug.ndim != 2:
            raise ValueError(f"params_aug must be 2D [S, P], got {params_aug.shape}")
        if latents.ndim != 2:
            raise ValueError(f"latents must be 2D [S, Z], got {latents.shape}")
        if params_aug.shape[0] != latents.shape[0]:
            raise ValueError(
                f"params_aug and latents must have same length along axis 0, "
                f"got {params_aug.shape[0]} vs {latents.shape[0]}"
            )

        self.params_aug = params_aug.astype(np.float32)
        self.latents = latents.astype(np.float32)

    def __len__(self) -> int:
        return self.params_aug.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = torch.from_numpy(self.params_aug[idx])   # [P]
        z = torch.from_numpy(self.latents[idx])      # [Z]
        return p, z


# ---------------------------------------------------------------------------
# Latent regressor network: small MLP
# ---------------------------------------------------------------------------

class LatentRegressorMLP(nn.Module):
    """
    Simple MLP mapping:

        (params_phys, t_norm)  -->  latent code z

    Inputs:
        in_dim      : P  (e.g., P_phys + 1 time coordinate)
        latent_dim  : Z  (PointNet encoder latent dimension)
        hidden_dims : sequence of hidden layer sizes, e.g. (128, 128)
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            last = h
        layers.append(nn.Linear(last, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        params : [B, P]
        returns z_pred : [B, latent_dim]
        """
        return self.net(params)


# ---------------------------------------------------------------------------
# Training utilities for latent regressor
# ---------------------------------------------------------------------------

@dataclass
class LatentRegressorTrainConfig:
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    val_fraction: float = 0.2
    patience: int = 30
    shuffle: bool = True
    random_state: int = 42


def build_latent_regression_dataloaders(
    params_aug: np.ndarray,
    latents: np.ndarray,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Convenience helper to split params/latents into train/val loaders.

    params_aug : [S, P]
    latents    : [S, Z]
    """
    dataset = LatentRegressionDataset(params_aug, latents)
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=shuffle,
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader


def train_latent_regressor_torch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 30,
) -> nn.Module:
    """
    Train the latent regressor MLP with standard MSE loss:

        loss = mean(|| z_pred - z_true ||^2)

    Early stopping is based on validation loss.
    """
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_loss: Optional[float] = None
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # ------------- TRAIN -------------
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for params_b, z_b in train_loader:
            params_b = params_b.to(device)  # [B, P]
            z_b = z_b.to(device)            # [B, Z]

            optimizer.zero_grad()
            z_pred = model(params_b)        # [B, Z]

            loss = F.mse_loss(z_pred, z_b)
            loss.backward()
            optimizer.step()

            batch_size = params_b.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        # ------------- VAL -------------
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for params_b, z_b in val_loader:
                params_b = params_b.to(device)
                z_b = z_b.to(device)

                z_pred = model(params_b)
                loss = F.mse_loss(z_pred, z_b)

                batch_size = params_b.size(0)
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(val_count, 1)

        print(
            f"[LatentReg] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4e}, val_loss={val_loss:.4e}"
        )

        # ------------- EARLY STOPPING -------------
        if best_val_loss is None or val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[LatentReg] Early stopping at epoch {epoch} "
                    f"(no improvement in {patience} epochs)."
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


__all__ = [
    "LatentRegressionDataset",
    "LatentRegressorMLP",
    "LatentRegressorTrainConfig",
    "build_latent_regression_dataloaders",
    "train_latent_regressor_torch",
]
