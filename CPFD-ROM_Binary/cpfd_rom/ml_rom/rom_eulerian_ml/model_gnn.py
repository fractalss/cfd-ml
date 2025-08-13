# Modified model.py with parameter injection in latent space (concatenation logic)
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers, callbacks

# Enable XLA acceleration
tf.config.optimizer.set_jit(True)

# ---------------------------------
# Model
# ---------------------------------

class ThreeLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=1, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)
        self.conv2 = GCNConv(hidden, hidden, cached=True)
        self.conv3 = GCNConv(hidden, out_dim, cached=True)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# ---------------------------------
# Training / Eval / Write
# ---------------------------------

def _load_target_vec(path: Path) -> torch.Tensor:
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    return torch.tensor(arr if arr.ndim>1 else arr.reshape(-1,1), dtype=torch.float32)


def _list_targets(graph_dir: Path):
    snap_dir = graph_dir / 'snapshots'
    files = sorted(list(snap_dir.glob('target_*.parquet')) + list(snap_dir.glob('target_*.csv')))
    if not files:
        raise FileNotFoundError(f"No target_* files found in {snap_dir}")
    rx = re.compile(r'target_(\d+\.?\d*)s\.(?:parquet|csv)')
    times = [(float(m.group(1)), p) for p in files if (m := rx.match(p.name))]
    return sorted(times, key=lambda t: t[0])


def laplacian_smoothness(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    diff = pred[src] - pred[dst]
    return (diff**2).mean()


def train_and_validate(model, edge_index, x, target_paths, device,
                       epochs=50, lr=1e-3, png_path: Path | None = None,
                       live_plotter: LivePlotter | None = None,
                       time_mode: str = 'none', fourier_m: int = 4, lambda_smooth: float = 0.0):
    model.to(device); x = x.to(device); edge_index = edge_index.to(device)
    n = len(target_paths); n_train = max(1, int(0.8*n))

    print(f"[NORM] Computing target normalization from first {n_train} snapshots")
    sum_, sumsq, count = 0.0, 0.0, 0
    for i in tqdm(range(n_train), desc="Loading train targets", leave=False):
        y = _load_target_vec(target_paths[i][1])
        sum_ += float(y.sum().item())
        sumsq += float((y**2).sum().item())
        count += int(y.numel())
        del y
    y_mu = (sum_ / max(1, count))
    var = max((sumsq / max(1, count)) - y_mu * y_mu, 0.0)
    y_sigma = math.sqrt(var) if var > 0 else 1.0

    # Time stats for Option A
    t_stats = _compute_time_stats(target_paths, n_train)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_hist = []
    train_eval_hist = []
    val_hist = []

    for epoch in range(1, epochs+1):
        model.train(); epoch_loss = 0.0
        train_bar = tqdm(torch.randperm(n_train).tolist(), desc=f"Epoch {epoch:03d}/{epochs} [train]", leave=False)
        for i in train_bar:
            t_i, p = target_paths[i]
            y = (_load_target_vec(p).to(device) - y_mu) / y_sigma
            opt.zero_grad(set_to_none=True)
            x_t = _concat_time(x, float(t_i), device, time_mode, t_stats, fourier_m)
            pred = model(x_t, edge_index)
            mse = loss_fn(pred, y)
            reg = lambda_smooth * laplacian_smoothness(pred, edge_index) if lambda_smooth > 0 else 0.0
            loss = mse + (reg if isinstance(reg, torch.Tensor) else torch.tensor(reg, device=device, dtype=pred.dtype))
            loss.backward(); opt.step()
            epoch_loss += mse.item()
            train_bar.set_postfix(mse=f"{mse.item():.3e}", reg=(f"{reg.item():.2e}" if isinstance(reg, torch.Tensor) else f"{reg:.2e}"))
            del y, pred, mse
        train_mse_n = epoch_loss / max(1, n_train)

        # Option D: Train metric in eval() (dropout off)
        model.eval()
        with torch.no_grad():
            te_losses = []
            for i in range(n_train):
                t_i, p = target_paths[i]
                y = (_load_target_vec(p).to(device) - y_mu) / y_sigma
                x_t = _concat_time(x, float(t_i), device, time_mode, t_stats, fourier_m)
                pred = model(x_t, edge_index)
                te_losses.append(loss_fn(pred, y).item())
            train_eval_mse_n = float(np.mean(te_losses)) if te_losses else float('nan')

        val_losses = []
        if n_train < n:
            val_bar = tqdm(range(n_train, n), desc=f"Epoch {epoch:03d}/{epochs} [val]", leave=False)
            with torch.no_grad():
                for j in val_bar:
                    t_j, p = target_paths[j]
                    y = (_load_target_vec(p).to(device) - y_mu) / y_sigma
                    x_t = _concat_time(x, float(t_j), device, time_mode, t_stats, fourier_m)
                    pred = model(x_t, edge_index)
                    l = loss_fn(pred, y).item()
                    val_losses.append(l)
                    val_bar.set_postfix(mse_n=f"{l:.3e}")
                    del y, pred
        val_mse_n = float(np.mean(val_losses)) if val_losses else float('nan')
        val_rmse = (math.sqrt(val_mse_n) * y_sigma) if val_losses else float('nan')

        train_hist.append(train_mse_n)
        train_eval_hist.append(train_eval_mse_n)
        val_hist.append(val_mse_n)

        print(f"[EPOCH {epoch:03d}] train_MSE_n={train_mse_n:.6e}  train_eval_MSE_n={train_eval_mse_n:.6e}  val_MSE_n={val_mse_n:.6e}  val_RMSE_phys={val_rmse:.6e}")

