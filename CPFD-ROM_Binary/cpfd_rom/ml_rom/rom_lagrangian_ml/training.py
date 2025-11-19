# cpfd_rom/ml_rom/rom_lagrangian_ml/training.py

import torch
from tqdm import tqdm


def train_pointnet_torch(model, train_loader, val_loader, device,
                         epochs=100, lr=1e-3, patience=10):
    """Basic training loop with early stopping on val loss.

    Supports optional parameter conditioning if each batch is a dict with
    keys "x" and "params".
    """

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in tqdm(train_loader, desc=f"Epoch {ep:03d} [train]", leave=False):
            # batch can be a tensor [B, N, C] or dict{"x", "params"}
            if isinstance(batch, dict):
                x = batch["x"].to(device)
                params = batch.get("params", None)
                if params is not None:
                    params = params.to(device)
            else:
                x = batch.to(device)
                params = None

            optim.zero_grad()
            recon, _ = model(x, params=params)  # autoencoder: recon vs x
            loss = torch.mean((recon - x) ** 2)
            loss.backward()
            optim.step()

            batch_size = x.shape[0]
            train_loss += loss.item() * batch_size
            n_train += batch_size

        train_loss /= max(n_train, 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {ep:03d} [val]", leave=False):
                if isinstance(batch, dict):
                    x = batch["x"].to(device)
                    params = batch.get("params", None)
                    if params is not None:
                        params = params.to(device)
                else:
                    x = batch.to(device)
                    params = None

                recon, _ = model(x, params=params)
                loss = torch.mean((recon - x) ** 2)

                batch_size = x.shape[0]
                val_loss += loss.item() * batch_size
                n_val += batch_size

        val_loss /= max(n_val, 1)

        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4e}, val_loss={val_loss:.4e}")

        # ---- Early stopping ----
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping at epoch {ep}. Best val_loss={best_val:.4e}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


__all__ = ["train_pointnet_torch"]
