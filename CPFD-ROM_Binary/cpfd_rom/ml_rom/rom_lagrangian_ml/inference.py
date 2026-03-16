# cpfd_rom/ml_rom/rom_lagrangian_ml/inference.py

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def infer_pointnet_with_latent_regression(
    user_param,
    model: torch.nn.Module,
    latent_regressor: torch.nn.Module,
    scaffold_graphs: list,
    output_dir: str,   # kept for API compatibility
    field_variable: str,  # kept for API compatibility
    feature_stats: dict | None = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 1,
):
    """
    Inference for raw-particle Lagrangian ROM.

    Raw-particle contract:
      - model predicts normalized [x, y, z, field]
      - Cloud ID is passed through from scaffold graph
      - output rows are [x, y, z, Cloud ID, field]

    Returns
    -------
    preds_list : list[np.ndarray]
        Each entry has shape [N_i, 5] with columns:
            [x, y, z, Cloud ID, field]
    times : list[float]
        Simulation times, one per snapshot.
    """
    model = model.to(device).eval()
    latent_regressor = latent_regressor.to(device).eval()

    loader = DataLoader(scaffold_graphs, batch_size=batch_size, shuffle=False, drop_last=False)

    preds_list = []
    times = []

    user_param_vec = np.asarray(user_param, dtype=np.float32).reshape(-1)  # [P_phys]
    P_phys = user_param_vec.shape[0]

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if not hasattr(batch, "params"):
                raise AttributeError("[Inference] batch is missing 'params' attribute.")
            if not hasattr(batch, "cloud_id"):
                raise AttributeError("[Inference] batch is missing 'cloud_id' attribute.")

            # batch.params is [B, P_aug], last col is t_norm
            t_norm = batch.params[:, -1:]   # [B,1]
            B = t_norm.shape[0]

            user_phys = (
                torch.tensor(user_param_vec, dtype=torch.float32, device=device)
                .view(1, P_phys)
                .repeat(B, 1)
            )  # [B,P_phys]

            p_aug = torch.cat([user_phys, t_norm], dim=1)  # [B, P_phys+1]

            # latent prediction
            z_pred = latent_regressor(p_aug)  # [B, latent_dim]

            # decode with scaffold/template features
            recon = model.decode(z_pred, batch.x[:, :4], batch.batch, p_aug)  # [N_total,4]

            batch_vec = batch.batch  # [N_total]

            for g_idx in range(B):
                mask = (batch_vec == g_idx)
                recon_g = recon[mask]  # [N_i,4]

                pos = recon_g[:, :3]
                field = recon_g[:, 3:4]

                # denormalize
                if feature_stats is not None:
                    pos_mean = feature_stats["pos_mean"].to(device)
                    pos_std = feature_stats["pos_std"].to(device)
                    pos = pos * (pos_std + 1e-8) + pos_mean

                    fmin = float(feature_stats["field_min"])
                    fmax = float(feature_stats["field_max"])
                    field = field * (fmax - fmin + 1e-8) + fmin

                # pass-through Cloud ID from scaffold graph
                cid = batch.cloud_id[mask].to(device).float().view(-1, 1)  # [N_i,1]

                # final raw-particle output: [x, y, z, Cloud ID, field]
                out_g = torch.cat([pos, cid, field], dim=1)  # [N_i,5]

                preds_list.append(out_g.cpu().numpy())

                if hasattr(batch, "time"):
                    if batch.time.numel() >= B:
                        tval = float(batch.time[g_idx].item())
                    else:
                        tval = float(batch.time.item())
                else:
                    tval = float(g_idx)

                times.append(tval)

    return preds_list, times