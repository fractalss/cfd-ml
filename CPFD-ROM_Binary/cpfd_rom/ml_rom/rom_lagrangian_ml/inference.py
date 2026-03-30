# cpfd_rom/ml_rom/rom_lagrangian_ml/inference.py

from __future__ import annotations

from typing import Optional, Sequence

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
    override_times: Optional[Sequence[float]] = None,
):
    """
    Inference for raw-particle Lagrangian ROM.

    Raw-particle contract:
      - model predicts normalized [x, y, z, field]
      - Cloud ID is passed through from scaffold graph
      - output rows are [x, y, z, Cloud ID, field]

    Time handling:
      - if override_times is provided, those times are used for outputs
      - otherwise scaffold graph times (batch.time) are used
      - if neither exists, graph index is used as fallback

    Returns
    -------
    preds_list : list[np.ndarray]
        Each entry has shape [N_i, 5] with columns:
            [x, y, z, Cloud ID, field]
    times : list[float]
        Output times, one per snapshot.
    """
    model = model.to(device).eval()
    latent_regressor = latent_regressor.to(device).eval()

    if override_times is not None:
        override_times = [float(t) for t in override_times]
        if len(override_times) != len(scaffold_graphs):
            raise ValueError(
                f"[Inference] override_times length ({len(override_times)}) must match "
                f"number of scaffold_graphs ({len(scaffold_graphs)})."
            )

    loader = DataLoader(
        scaffold_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    preds_list = []
    times = []

    user_param_vec = np.asarray(user_param, dtype=np.float32).reshape(-1)  # [P_phys]
    p_phys_dim = user_param_vec.shape[0]

    graph_counter = 0  # global graph index across batches

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if not hasattr(batch, "params"):
                raise AttributeError("[Inference] batch is missing 'params' attribute.")
            if not hasattr(batch, "cloud_id"):
                raise AttributeError("[Inference] batch is missing 'cloud_id' attribute.")
            if not hasattr(batch, "edge_index"):
                raise AttributeError("[Inference] batch is missing 'edge_index' attribute.")
            if not hasattr(batch, "batch"):
                raise AttributeError("[Inference] batch is missing 'batch' attribute.")

            if batch.params.dim() != 2:
                raise ValueError(
                    f"[Inference] Expected batch.params to be 2D [B, P_aug], got {tuple(batch.params.shape)}"
                )

            # batch.params is [B, P_aug], with last column = t_norm
            t_norm = batch.params[:, -1:]   # [B, 1]
            B = t_norm.shape[0]

            user_phys = (
                torch.tensor(user_param_vec, dtype=torch.float32, device=device)
                .view(1, p_phys_dim)
                .repeat(B, 1)
            )  # [B, P_phys]

            p_aug = torch.cat([user_phys, t_norm], dim=1)  # [B, P_phys + 1]

            # Predict latent code from parameter + normalized time
            z_pred = latent_regressor(p_aug)  # [B, latent_dim]

            # Decode using scaffold/template features
            recon = model.decode(
                z_pred,
                batch.x[:, :4],
                batch.edge_index,
                batch.batch,
                p_aug,
            )  # [N_total, 4]

            batch_vec = batch.batch  # [N_total]

            # Fallback graph-level times from scaffold graphs
            if hasattr(batch, "time"):
                t_batch = batch.time.view(-1)
                if t_batch.numel() != B:
                    raise ValueError(
                        f"[Inference] Expected {B} graph-level times in batch.time, "
                        f"got shape {tuple(batch.time.shape)}"
                    )
            else:
                t_batch = None

            for g_idx in range(B):
                mask = (batch_vec == g_idx)
                recon_g = recon[mask]  # [N_i, 4]

                pos = recon_g[:, :3]
                field = recon_g[:, 3:4]

                # Denormalize back to physical units
                if feature_stats is not None:
                    pos_mean = feature_stats["pos_mean"].to(device)
                    pos_std = feature_stats["pos_std"].to(device)
                    pos = pos * (pos_std + 1e-8) + pos_mean

                    fmin = float(feature_stats["field_min"])
                    fmax = float(feature_stats["field_max"])
                    field = field * (fmax - fmin + 1e-8) + fmin

                # Pass through Cloud ID from scaffold graph
                cid = batch.cloud_id[mask].to(device).float().view(-1, 1)  # [N_i, 1]

                # Final raw-particle output: [x, y, z, Cloud ID, field]
                out_g = torch.cat([pos, cid, field], dim=1)  # [N_i, 5]
                preds_list.append(out_g.cpu().numpy())

                # Output time selection priority:
                # 1. override_times
                # 2. scaffold graph batch.time
                # 3. global graph index
                if override_times is not None:
                    tval = float(override_times[graph_counter])
                elif t_batch is not None:
                    tval = float(t_batch[g_idx].item())
                else:
                    tval = float(graph_counter)

                times.append(tval)
                graph_counter += 1

    return preds_list, times