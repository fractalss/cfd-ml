import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def _forward_pointnet(model, batch, device):
    # Extract inputs
    x = batch.x.to(device)                # [N, input_dim]
    batch_idx = batch.batch.to(device)    # [N]
    params = batch.params.to(device)      # [B, param_dim] or [B]

    if params.dim() == 1:
        params = params.unsqueeze(1)      # Ensure shape is [B, param_dim]

    # Forward pass
    recon = model(x, batch_idx, params)
    return recon, x, None

def train_pointnet_torch(model, dataset, device, epochs=100, lr=1e-3, batch_size=4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            recon, x, _ = _forward_pointnet(model, batch, device)

            # Assume reconstruction target is x (autoencoding)
            loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

    return model

