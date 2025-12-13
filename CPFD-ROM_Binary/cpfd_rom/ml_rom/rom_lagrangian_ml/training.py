import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

def _forward_pointnet(model, batch, device):
    x = batch.x.to(device)               # [N, 4]
    batch_idx = batch.batch.to(device)   # [N]
    params = batch.params.to(device)     # [B, param_dim]
    target = batch.y.to(device)          # [N, 4]

    if params.shape[0] == x.shape[0]:
        from torch_scatter import scatter_mean
        params = scatter_mean(params, batch_idx, dim=0)

    if params.dim() == 1:
        params = params.unsqueeze(0)

    recon = model(x, batch_idx, params)  # [N, 4]
    return recon, target


def train_pointnet_torch(model, dataset, device, epochs=100, lr=1e-3, batch_size=4):
    """
    Trains the PointNetAutoencoder to reconstruct normalized [x, y, z, field] per point.

    Args:
        model: PointNetAutoencoder
        dataset: torch.utils.data.Dataset of PyG Data objects
        device: torch.device
        epochs: int, number of epochs
        lr: float, learning rate
        batch_size: int

    Returns:
        model: trained PointNetAutoencoder
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"[Epoch {epoch+1}/{epochs}]"):
            optimizer.zero_grad()

            pred, target = _forward_pointnet(model, batch, device)

            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  [Epoch {epoch+1}] Avg MSE Loss: {avg_loss:.6f}")

    return model
