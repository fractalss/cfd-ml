import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def _forward_pointnet_gnn(model, batch, device):
    """
    batch: PyG Batch with:
      - batch.x:         [N_total, 4] normalized [x, y, z, field]
      - batch.y:         [N_total, 4] target
      - batch.edge_index:[2, E]
      - batch.batch:     [N_total] graph index per node
      - batch.params:    [B, P_aug] graph-level conditioning params
    """
    batch = batch.to(device)

    x = batch.x
    edge_index = batch.edge_index
    batch_idx = batch.batch
    params = batch.params
    target = batch.y[:, :4]

    if params.dim() != 2:
        raise ValueError(
            f"[train] Expected batch.params to be 2D [B, P], got {tuple(params.shape)}"
        )

    if edge_index is None:
        raise ValueError("[train] batch.edge_index is missing.")

    recon, graph_latent = model(x, edge_index, batch_idx, params)
    return recon, target, graph_latent, batch_idx


def _batch_com_and_spread_loss(pred_xyz, true_xyz, batch_idx):
    """
    Compute COM and spread loss per graph in a PyG batch, then average across graphs.

    Args:
        pred_xyz: [N_total, 3]
        true_xyz: [N_total, 3]
        batch_idx: [N_total] graph id per node

    Returns:
        loss_com, loss_spread
    """
    unique_graphs = torch.unique(batch_idx)

    com_losses = []
    spread_losses = []

    for g in unique_graphs:
        mask = (batch_idx == g)
        pred_g = pred_xyz[mask]
        true_g = true_xyz[mask]

        if pred_g.size(0) < 2:
            continue

        pred_mean = torch.mean(pred_g, dim=0)
        true_mean = torch.mean(true_g, dim=0)
        com_losses.append(F.l1_loss(pred_mean, true_mean))

        pred_std = torch.std(pred_g, dim=0, unbiased=False)
        true_std = torch.std(true_g, dim=0, unbiased=False)
        spread_losses.append(F.l1_loss(pred_std, true_std))

    if len(com_losses) == 0:
        zero = pred_xyz.new_tensor(0.0)
        return zero, zero

    loss_com = torch.stack(com_losses).mean()
    loss_spread = torch.stack(spread_losses).mean()
    return loss_com, loss_spread


def train_pointnet_gnn_torch(
    model,
    dataset,
    device,
    epochs=100,
    lr=1e-3,
    batch_size=4,
    grad_clip_norm: float = 1.0,
    field_loss_weight: float = 1.0,
    com_loss_weight: float = 0.0,
    spread_loss_weight: float = 0.0,
):
    """
    Train the PointNet-GNN autoencoder to reconstruct normalized [x, y, z, field].

    Additional geometry-aware xyz losses:
      - COM loss: match mean x,y,z per graph
      - Spread loss: match std(x), std(y), std(z) per graph
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_xyz = 0.0
        total_field = 0.0
        total_com = 0.0
        total_spread = 0.0

        for batch in tqdm(loader, desc=f"[Epoch {epoch + 1}/{epochs}]"):
            optimizer.zero_grad()

            pred, target, _, batch_idx = _forward_pointnet_gnn(model, batch, device)

            pred_xyz = pred[:, :3]
            true_xyz = target[:, :3]

            loss_xyz = F.mse_loss(pred_xyz, true_xyz)
            loss_field = F.mse_loss(pred[:, 3:4], target[:, 3:4])

            loss_com, loss_spread = _batch_com_and_spread_loss(
                pred_xyz, true_xyz, batch_idx
            )

            loss = (
                loss_xyz
                + field_loss_weight * loss_field
                + com_loss_weight * loss_com
                + spread_loss_weight * loss_spread
            )

            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            total_loss += float(loss.item())
            total_xyz += float(loss_xyz.item())
            total_field += float(loss_field.item())
            total_com += float(loss_com.item())
            total_spread += float(loss_spread.item())

        n_batches = max(len(loader), 1)
        avg_loss = total_loss / n_batches
        avg_xyz = total_xyz / n_batches
        avg_field = total_field / n_batches
        avg_com = total_com / n_batches
        avg_spread = total_spread / n_batches

        print(
            f"  [Epoch {epoch + 1}] Avg Loss: {avg_loss:.6f} "
            f"(xyz={avg_xyz:.6f}, field={avg_field:.6f}, "
            f"com={avg_com:.6f}, spread={avg_spread:.6f}, "
            f"field_weight={field_loss_weight}, "
            f"com_weight={com_loss_weight}, spread_weight={spread_loss_weight})"
        )

    return model