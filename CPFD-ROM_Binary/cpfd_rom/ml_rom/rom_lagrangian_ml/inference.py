# cpfd_rom/ml_rom/rom_lagrangian_ml/inference.py

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def _graphs_are_time_sorted(graphs: list) -> bool:
    if len(graphs) <= 1:
        return True
    times = np.array([float(g.time.item()) for g in graphs], dtype=np.float64)
    return bool(np.all(np.diff(times) >= 0.0))


def _validate_template_graphs(scaffold_graphs: list) -> None:
    if len(scaffold_graphs) == 0:
        raise ValueError("[Inference] No template graphs provided.")

    required_attrs = ["x", "pos", "edge_index", "params", "time", "cloud_id"]
    for i, g in enumerate(scaffold_graphs):
        for attr in required_attrs:
            if not hasattr(g, attr):
                raise AttributeError(
                    f"[Inference] Template graph index {i} is missing required attribute '{attr}'."
                )

    if not _graphs_are_time_sorted(scaffold_graphs):
        raise ValueError(
            "[Inference] Template graphs are not sorted by time. "
            "Sort graphs by time before calling inference."
        )


def _print_template_summary(scaffold_graphs: list, override_times: Optional[Sequence[float]]) -> None:
    g0 = scaffold_graphs[0]
    g_last = scaffold_graphs[-1]

    t0 = float(g0.time.item())
    t_last = float(g_last.time.item())

    snap0 = getattr(g0, "snapshot_name", "<missing>")
    snap_last = getattr(g_last, "snapshot_name", "<missing>")

    print("[Inference] Template graph summary")
    print(f"  num_graphs              : {len(scaffold_graphs)}")
    print(f"  first_template_time     : {t0:.6f}")
    print(f"  first_template_snapshot : {snap0}")
    print(f"  last_template_time      : {t_last:.6f}")
    print(f"  last_template_snapshot  : {snap_last}")

    if override_times is not None and len(override_times) > 0:
        print(f"  first_override_time     : {float(override_times[0]):.6f}")
        print(f"  last_override_time      : {float(override_times[-1]):.6f}")


def _check_override_alignment(scaffold_graphs: list, override_times: Optional[Sequence[float]], atol: float) -> None:
    if override_times is None:
        return

    if len(override_times) != len(scaffold_graphs):
        raise ValueError(
            f"[Inference] override_times length ({len(override_times)}) must match "
            f"number of template graphs ({len(scaffold_graphs)})."
        )

    t_graph_0 = float(scaffold_graphs[0].time.item())
    t_override_0 = float(override_times[0])

    dt0 = abs(t_graph_0 - t_override_0)
    if dt0 > atol:
        raise ValueError(
            f"[Inference] First override time does not match first template graph time within tolerance.\n"
            f"  template time  = {t_graph_0:.6f}\n"
            f"  override time  = {t_override_0:.6f}\n"
            f"  abs diff       = {dt0:.6f}\n"
            f"  allowed atol   = {atol:.6f}"
        )


def _denormalize_prediction(
    pos: torch.Tensor,
    field: torch.Tensor,
    feature_stats: dict | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_stats is None:
        return pos, field

    pos_mean = feature_stats["pos_mean"].to(device)
    pos_std = feature_stats["pos_std"].to(device)
    pos = pos * (pos_std + 1e-8) + pos_mean

    fmin = float(feature_stats["field_min"])
    fmax = float(feature_stats["field_max"])
    field = field * (fmax - fmin + 1e-8) + fmin

    return pos, field


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
    time_alignment_atol: float = 1e-8,
    debug: bool = True,
):
    """
    Inference for the old raw-particle Lagrangian ROM path.

    Contract
    --------
    - model predicts normalized [x, y, z, field]
    - Cloud ID is passed through from template graph
    - output rows are [x, y, z, Cloud ID, field]
    - template graphs are expected to be time-sorted
    - the nearest rev's true first snapshot should be the first template graph

    Time handling
    -------------
    - if override_times is provided, those times are used for outputs
    - otherwise template graph times (batch.time) are used
    - if neither exists, graph index is used as fallback
    """
    model = model.to(device).eval()
    latent_regressor = latent_regressor.to(device).eval()

    if override_times is not None:
        override_times = [float(t) for t in override_times]

    _validate_template_graphs(scaffold_graphs)
    _check_override_alignment(scaffold_graphs, override_times, atol=time_alignment_atol)

    if debug:
        _print_template_summary(scaffold_graphs, override_times)

    loader = DataLoader(
        scaffold_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    preds_list: list[np.ndarray] = []
    times: list[float] = []

    user_param_vec = np.asarray(user_param, dtype=np.float32).reshape(-1)
    p_phys_dim = user_param_vec.shape[0]

    graph_counter = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
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

            # batch.params is [B, P_aug], with last column assumed to be t_norm
            t_norm = batch.params[:, -1:]   # [B, 1]
            B = t_norm.shape[0]

            user_phys = (
                torch.tensor(user_param_vec, dtype=torch.float32, device=device)
                .view(1, p_phys_dim)
                .repeat(B, 1)
            )  # [B, P_phys]

            p_aug = torch.cat([user_phys, t_norm], dim=1)  # [B, P_phys + 1]

            # Predict latent code from physical parameter + normalized time
            z_pred = latent_regressor(p_aug)  # [B, latent_dim]

            # Decode using template particle features
            recon = model.decode(
                z_pred,
                batch.x[:, :4],
                batch.edge_index,
                batch.batch,
                p_aug,
            )  # [N_total, 4]

            if recon.dim() != 2 or recon.shape[1] < 4:
                raise ValueError(
                    f"[Inference] Expected recon to have shape [N, >=4], got {tuple(recon.shape)}"
                )

            batch_vec = batch.batch  # [N_total]

            # Graph-level times from template graphs
            # Graph-level times from template graphs
            if hasattr(batch, "time"):
                t_batch = batch.time.view(-1)
                if t_batch.numel() != B:
                    raise ValueError(
                        f"[Inference] Expected {B} graph-level times in batch.time, "
                        f"got shape {tuple(batch.time.shape)}"
                    )
            else:
                t_batch = None

            recon = model.decode(
                z_pred,
                batch.x[:, :4],
                batch.edge_index,
                batch.batch,
                p_aug,
                time=t_batch,
            )
            # Optional graph-level names if preserved by batch collation
            has_snapshot_name = hasattr(batch, "snapshot_name")

            for g_idx in range(B):
                mask = (batch_vec == g_idx)
                recon_g = recon[mask]  # [N_i, 4]

                pos = recon_g[:, :3]
                field = recon_g[:, 3:4]

                pos, field = _denormalize_prediction(
                    pos=pos,
                    field=field,
                    feature_stats=feature_stats,
                    device=device,
                )

                cid = batch.cloud_id[mask].to(device).float().view(-1, 1)

                out_g = torch.cat([pos, cid, field], dim=1)  # [N_i, 5]
                preds_list.append(out_g.cpu().numpy())

                if override_times is not None:
                    tval = float(override_times[graph_counter])
                elif t_batch is not None:
                    tval = float(t_batch[g_idx].item())
                else:
                    tval = float(graph_counter)

                times.append(tval)

                if debug and graph_counter < 3:
                    try:
                        snap_name = (
                            batch.snapshot_name[g_idx]
                            if has_snapshot_name and isinstance(batch.snapshot_name, (list, tuple))
                            else getattr(scaffold_graphs[graph_counter], "snapshot_name", "<missing>")
                        )
                    except Exception:
                        snap_name = getattr(scaffold_graphs[graph_counter], "snapshot_name", "<missing>")

                    template_time = (
                        float(t_batch[g_idx].item()) if t_batch is not None
                        else getattr(scaffold_graphs[graph_counter], "time", torch.tensor([np.nan])).item()
                    )

                    print(f"[Inference][Graph {graph_counter}]")
                    print(f"  template_snapshot      : {snap_name}")
                    print(f"  template_time          : {template_time:.6f}")
                    print(f"  output_time            : {tval:.6f}")
                    print(f"  template_nodes         : {int(mask.sum().item())}")
                    print(f"  predicted_xyz_min      : {pos.min(dim=0).values.detach().cpu().numpy()}")
                    print(f"  predicted_xyz_max      : {pos.max(dim=0).values.detach().cpu().numpy()}")
                    print(f"  predicted_field_minmax : ({field.min().item():.6e}, {field.max().item():.6e})")

                graph_counter += 1

    return preds_list, times