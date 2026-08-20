# cpfd_rom/ml/lagrangian/inference.py

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader


logger = logging.getLogger(__name__)


def _graphs_are_time_sorted(graphs: list) -> bool:
    if len(graphs) <= 1:
        return True
    times = np.array([float(g.time.item()) for g in graphs], dtype=np.float64)
    return bool(np.all(np.diff(times) >= 0.0))


def _validate_template_graphs(scaffold_graphs: list) -> None:
    if len(scaffold_graphs) == 0:
        raise ValueError("[Inference] No template graphs provided.")

    required_attrs = ["x", "pos", "edge_index", "params", "time", "cloud_id"]
    for i, graph in enumerate(scaffold_graphs):
        for attr in required_attrs:
            if not hasattr(graph, attr):
                raise AttributeError(
                    f"[Inference] Template graph index {i} is missing required "
                    f"attribute '{attr}'."
                )

    if not _graphs_are_time_sorted(scaffold_graphs):
        raise ValueError(
            "[Inference] Template graphs are not sorted by time. "
            "Sort graphs by time before calling inference."
        )


def _log_template_summary(
    scaffold_graphs: list,
    override_times: Optional[Sequence[float]],
) -> None:
    first_graph = scaffold_graphs[0]
    last_graph = scaffold_graphs[-1]

    lines = [
        "Template graph summary:",
        f"  num_graphs              : {len(scaffold_graphs)}",
        f"  first_template_time     : {float(first_graph.time.item()):.6f}",
        "  first_template_snapshot : "
        f"{getattr(first_graph, 'snapshot_name', '<missing>')}",
        f"  last_template_time      : {float(last_graph.time.item()):.6f}",
        "  last_template_snapshot  : "
        f"{getattr(last_graph, 'snapshot_name', '<missing>')}",
    ]

    if override_times is not None and len(override_times) > 0:
        lines.extend(
            [
                f"  first_override_time     : {float(override_times[0]):.6f}",
                f"  last_override_time      : {float(override_times[-1]):.6f}",
            ]
        )

    logger.debug("\n".join(lines))


def _check_override_alignment(
    scaffold_graphs: list,
    override_times: Optional[Sequence[float]],
    atol: float,
) -> None:
    if override_times is None:
        return

    if len(override_times) != len(scaffold_graphs):
        raise ValueError(
            f"[Inference] override_times length ({len(override_times)}) must match "
            f"number of template graphs ({len(scaffold_graphs)})."
        )

    template_time = float(scaffold_graphs[0].time.item())
    override_time = float(override_times[0])
    difference = abs(template_time - override_time)

    if difference > atol:
        raise ValueError(
            "[Inference] First override time does not match first template graph "
            "time within tolerance.\n"
            f"  template time  = {template_time:.6f}\n"
            f"  override time  = {override_time:.6f}\n"
            f"  abs diff       = {difference:.6f}\n"
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

    field_min = float(feature_stats["field_min"])
    field_max = float(feature_stats["field_max"])
    field = field * (field_max - field_min + 1e-8) + field_min

    return pos, field


def infer_pointnet_with_latent_regression(
    user_param,
    model: torch.nn.Module,
    latent_regressor: torch.nn.Module,
    scaffold_graphs: list,
    output_dir: str,  # kept for API compatibility
    field_variable: str,  # kept for API compatibility
    feature_stats: dict | None = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 1,
    override_times: Optional[Sequence[float]] = None,
    time_alignment_atol: float = 1e-8,
    debug: bool = True,  # kept for API compatibility; logging level controls output
):
    """
    Run inference for the old raw-particle Lagrangian ROM path.

    The model predicts normalized ``[x, y, z, field]`` values. Cloud IDs pass
    through from the time-sorted template graphs, producing output rows in the
    form ``[x, y, z, Cloud ID, field]``.

    If ``override_times`` is supplied, those values are used for the output.
    Otherwise, template graph times are used, with graph indices as a fallback.
    The retained ``debug`` argument no longer bypasses CLI logging controls;
    diagnostics are emitted only when DEBUG logging is enabled (``-vv``).
    """
    # del output_dir, field_variable, debug

    model = model.to(device).eval()
    latent_regressor = latent_regressor.to(device).eval()

    if override_times is not None:
        override_times = [float(time) for time in override_times]

    _validate_template_graphs(scaffold_graphs)
    _check_override_alignment(
        scaffold_graphs,
        override_times,
        atol=time_alignment_atol,
    )

    if logger.isEnabledFor(logging.DEBUG):
        _log_template_summary(scaffold_graphs, override_times)

    loader = DataLoader(
        scaffold_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    predictions: list[np.ndarray] = []
    times: list[float] = []

    user_param_vec = np.asarray(user_param, dtype=np.float32).reshape(-1)
    physical_param_dim = user_param_vec.shape[0]
    graph_counter = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            for attr in ("params", "cloud_id", "edge_index", "batch"):
                if not hasattr(batch, attr):
                    raise AttributeError(
                        f"[Inference] batch is missing '{attr}' attribute."
                    )

            if batch.params.dim() != 2:
                raise ValueError(
                    "[Inference] Expected batch.params to be 2D [B, P_aug], "
                    f"got {tuple(batch.params.shape)}"
                )

            # batch.params is [B, P_aug], with its last column assumed to be t_norm.
            normalized_time = batch.params[:, -1:]
            num_graphs = normalized_time.shape[0]

            user_physical_params = (
                torch.as_tensor(user_param_vec, dtype=torch.float32, device=device)
                .view(1, physical_param_dim)
                .repeat(num_graphs, 1)
            )
            augmented_params = torch.cat(
                [user_physical_params, normalized_time],
                dim=1,
            )

            latent_prediction = latent_regressor(augmented_params)
            batch_vector = batch.batch

            if hasattr(batch, "time"):
                batch_times = batch.time.view(-1)
                if batch_times.numel() != num_graphs:
                    raise ValueError(
                        f"[Inference] Expected {num_graphs} graph-level times in "
                        f"batch.time, got shape {tuple(batch.time.shape)}"
                    )
            else:
                batch_times = None

            # Decode once, using graph-level time when it is available.
            reconstruction = model.decode(
                latent_prediction,
                batch.x[:, :4],
                batch.edge_index,
                batch.batch,
                augmented_params,
                time=batch_times,
            )

            if reconstruction.dim() != 2 or reconstruction.shape[1] < 4:
                raise ValueError(
                    "[Inference] Expected recon to have shape [N, >=4], "
                    f"got {tuple(reconstruction.shape)}"
                )

            has_snapshot_name = hasattr(batch, "snapshot_name")

            for graph_index in range(num_graphs):
                mask = batch_vector == graph_index
                graph_reconstruction = reconstruction[mask]

                pos = graph_reconstruction[:, :3]
                field = graph_reconstruction[:, 3:4]
                pos, field = _denormalize_prediction(
                    pos=pos,
                    field=field,
                    feature_stats=feature_stats,
                    device=device,
                )

                cloud_id = batch.cloud_id[mask].to(device).float().view(-1, 1)
                graph_output = torch.cat([pos, cloud_id, field], dim=1)
                predictions.append(graph_output.cpu().numpy())

                if override_times is not None:
                    output_time = float(override_times[graph_counter])
                elif batch_times is not None:
                    output_time = float(batch_times[graph_index].item())
                else:
                    output_time = float(graph_counter)

                times.append(output_time)

                if logger.isEnabledFor(logging.DEBUG) and graph_counter < 3:
                    try:
                        snapshot_name = (
                            batch.snapshot_name[graph_index]
                            if has_snapshot_name
                            and isinstance(batch.snapshot_name, (list, tuple))
                            else getattr(
                                scaffold_graphs[graph_counter],
                                "snapshot_name",
                                "<missing>",
                            )
                        )
                    except Exception:
                        snapshot_name = getattr(
                            scaffold_graphs[graph_counter],
                            "snapshot_name",
                            "<missing>",
                        )

                    if batch_times is not None:
                        template_time = float(batch_times[graph_index].item())
                    else:
                        fallback_time = getattr(
                            scaffold_graphs[graph_counter],
                            "time",
                            torch.tensor([np.nan]),
                        )
                        template_time = float(fallback_time.item())

                    xyz_min = pos.min(dim=0).values.detach().cpu().numpy()
                    xyz_max = pos.max(dim=0).values.detach().cpu().numpy()
                    field_min = field.min().item()
                    field_max = field.max().item()
                    template_nodes = int(mask.sum().item())

                    logger.debug(
                        "Graph %d prediction:\n"
                        "  template_snapshot      : %s\n"
                        "  template_time          : %.6f\n"
                        "  output_time            : %.6f\n"
                        "  template_nodes         : %d\n"
                        "  predicted_xyz_min      : %s\n"
                        "  predicted_xyz_max      : %s\n"
                        "  predicted_field_minmax : (%.6e, %.6e)",
                        graph_counter,
                        snapshot_name,
                        template_time,
                        output_time,
                        template_nodes,
                        xyz_min,
                        xyz_max,
                        field_min,
                        field_max,
                    )

                graph_counter += 1

    return predictions, times
