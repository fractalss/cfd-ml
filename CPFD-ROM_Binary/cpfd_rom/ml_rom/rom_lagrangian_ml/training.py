import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def _forward_pointnet(model, batch, device):
    """
    batch: PyG Batch with:
      - batch.x:     [N_total, 4] normalized [x,y,z,field]
      - batch.y:     [N_total, 4] target
      - batch.batch: [N_total] graph index per node
      - batch.params:[B, P_aug] graph-level conditioning params
    """
    batch = batch.to(device)

    x = batch.x                  # [N_total, 4]
    batch_idx = batch.batch      # [N_total]
    params = batch.params        # [B, P_aug]
    target = batch.y[:, :4]      # [N_total, 4]

    if params.dim() != 2:
        raise ValueError(
            f"[train] Expected batch.params to be 2D [B,P], got {tuple(params.shape)}"
        )

    recon, _ = model(x, batch_idx, params)   # model now returns (recon, graph_latent)
    return recon, target


def train_pointnet_torch(
    model,
    dataset,
    device,
    epochs=100,
    lr=1e-3,
    batch_size=4,
    grad_clip_norm: float = 1.0,
    field_loss_weight: float = 1.0,
):
    """
    Train the PointNetAutoencoder to reconstruct normalized [x, y, z, field].

    Args:
        model: PointNetAutoencoder
        dataset: Dataset of PyG Data objects
        device: torch.device
        epochs: number of epochs
        lr: learning rate
        batch_size: batch size
        grad_clip_norm: optional gradient clipping norm
        field_loss_weight: extra weight on field-channel MSE
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"[Epoch {epoch+1}/{epochs}]"):
            optimizer.zero_grad()

            pred, target = _forward_pointnet(model, batch, device)

            # split xyz vs field so field can be weighted if desired
            loss_xyz = F.mse_loss(pred[:, :3], target[:, :3])
            loss_field = F.mse_loss(pred[:, 3:4], target[:, 3:4])
            loss = loss_xyz + field_loss_weight * loss_field

            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(loader), 1)
        print(
            f"  [Epoch {epoch+1}] Avg Loss: {avg_loss:.6f} "
            f"(field_weight={field_loss_weight})"
        )

    return model