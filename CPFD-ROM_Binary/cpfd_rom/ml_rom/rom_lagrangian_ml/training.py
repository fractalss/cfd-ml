# cpfd_rom/ml_rom/rom_lagrangian_ml/training.py

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Generic PointNet autoencoder / decoder trainer (still used by other pipelines)
# ---------------------------------------------------------------------------

def _forward_pointnet(
    model: nn.Module,
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unified forward helper for PointNet autoencoder / decoder.

    Works for both dataset types:
      - SnapshotDataset:
            batch is a Tensor [B, N, n_features_total]
      - SnapshotParamDataset:
            batch is a dict {"x": Tensor, "params": Tensor, ...}

    The network operates only on the first 4 dynamic features:
        [x, y, z, field]

    If the batch contains a key "baseline", it is interpreted as a
    per-point baseline for these 4 channels in the *same scaled space*
    as the inputs, and the model output is treated as a residual to be
    added to this baseline. In the current Lagrangian ROM pipeline this
    baseline typically comes from a linear / ridge regression model
    built on flattened [x, y, z, field].
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
            baseline_dyn = baseline_full[..., :4]         # [B, N, 4]

        x_dyn = x_full[..., :4]                           # [B, N, 4]
        if params is not None:
            out = model(x_dyn, params=params)
        else:
            out = model(x_dyn)
    else:
        # Plain tensor batch with no params / baseline
        x_full = batch.to(device)                         # [B, N, n_features_total]
        x_dyn = x_full[..., :4]                           # [B, N, 4]
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
    Train a PointNet-like autoencoder / decoder with a clean, symmetric
    train/val loop. Only the first 4 input features are used for loss:

        MSE(pred_dyn_scaled, x_scaled[..., :4]).mean()

    CloudID and CloudID_base are ignored in the loss.

    If the batch includes a "baseline" tensor (4 dynamic channels in
    scaled space), the model output is treated as a residual on top of
    that baseline (in the Lagrangian ROM this baseline comes from the
    linear/ridge regression model). Otherwise, the model predicts the
    full dynamic state directly.
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

        # ---------------- VAL ----------------
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

        # early stopping
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


# ---------------------------------------------------------------------------
# Residual decoder trainer (linear/ridge baseline + params -> residual)
# ---------------------------------------------------------------------------

def train_pointnet_residual_torch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    importance_alpha: float = 0.1,
) -> nn.Module:
    """
    Train a residual decoder of the form:

        R_pred = model(baseline_dyn_scaled, params)

    where
        baseline_dyn_scaled : [B, N_or_K, 4]  - baseline dynamic features (scaled)
        params              : [B, P]          - global parameters per snapshot
        R_true              : [B, N_or_K, 4]  - target residual (scaled)

    In the Lagrangian ROM pipeline, baseline_dyn_scaled is produced by a
    linear / ridge regression model over flattened [x,y,z,field] using
    the Eulerian baseline infrastructure, then reshaped to [S, N, 4] (or
    [S, K, 4] for coarsened centroids) and scaled with a StandardScaler.

    The DataLoader is assumed to come from BaselineResidualDataset (or a
    compatible variant), whose __getitem__ returns:

        X_b : [N_or_K, 4+P]
              (baseline_dyn_scaled concatenated with tiled params)
        p_b : [P]
        R_b : [N_or_K, 4]

    We only use X_b[..., :4] as baseline_dyn_scaled inside this trainer.

    Loss weighting
    --------------
    If importance_alpha <= 0, we use standard MSE:

        mean( (R_pred - R_true)^2 )

    If importance_alpha > 0, we use a point-wise importance weight based
    on the norm of the true residual R_b:

        r = ||R_b||_2 over channels  -> [B, N_or_K]
        w = 1 + importance_alpha * (r / mean(r))

    and minimize:

        mean( w * sum_c (R_pred - R_b)^2 )

    so that regions with large multi-channel residuals (e.g., bubble
    cavity, strong displacement) contribute more strongly to the loss.

    This trainer is agnostic to the particular residual decoder
    architecture and can be used with both:
      - PointNetResidualDecoder (dense / parcel level)
      - PointNetGATResidualDecoder (coarsened centroid graph, K nodes)
    as long as their forward signature is:

        model(baseline_dyn_scaled_batch, params_batch) -> residual_pred_batch
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    eps = 1e-8  # for numerical stability in normalization

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_losses = []

        for X_b, p_b, R_b in train_loader:
            # X_b: [B, N_or_K, 4+P]
            # p_b: [B, P]
            # R_b: [B, N_or_K, 4]

            X_b = X_b.to(device)
            p_b = p_b.to(device)
            R_b = R_b.to(device)

            baseline_dyn_scaled_b = X_b[..., :4]  # [B, N_or_K, 4]

            optimizer.zero_grad()
            R_pred = model(baseline_dyn_scaled_b, p_b)  # [B, N_or_K, 4]

            if importance_alpha <= 0.0:
                # Plain MSE over all points and channels
                loss = F.mse_loss(R_pred, R_b)
            else:
                # Importance-weighted MSE over points and channels
                diff = R_pred - R_b                          # [B, N, 4]
                se = torch.sum(diff**2, dim=-1)             # [B, N]

                # Residual magnitude per point (all 4 channels)
                r = torch.norm(R_b, dim=-1)                 # [B, N]

                mean_r = torch.mean(r) + eps
                w = 1.0 + importance_alpha * (r / mean_r)   # [B, N]

                loss = torch.mean(w * se)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        # ---------------- VAL ----------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, p_b, R_b in val_loader:
                X_b = X_b.to(device)
                p_b = p_b.to(device)
                R_b = R_b.to(device)

                baseline_dyn_scaled_b = X_b[..., :4]
                R_pred = model(baseline_dyn_scaled_b, p_b)

                if importance_alpha <= 0.0:
                    loss = F.mse_loss(R_pred, R_b)
                else:
                    diff = R_pred - R_b
                    se = torch.sum(diff**2, dim=-1)         # [B, N]
                    r = torch.norm(R_b, dim=-1)             # [B, N]

                    mean_r = torch.mean(r) + eps
                    w = 1.0 + importance_alpha * (r / mean_r)

                    loss = torch.mean(w * se)

                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4e}, val_loss={val_loss:.4e}")

        # ---------------- EARLY STOP ----------------
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
