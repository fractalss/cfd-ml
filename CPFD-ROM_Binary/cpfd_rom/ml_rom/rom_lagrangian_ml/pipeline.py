from __future__ import annotations

import gc
import logging
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from cpfd_rom.ml_rom.rom_lagrangian_ml.data_loader import (
    compute_feature_stats,
    extract_scaffold_graphs,
    get_initial_template_graph,
    load_lagrangian_snapshot_times,
    load_lagrangian_snapshots_as_graphs,
    sort_graphs_by_time,
)
from cpfd_rom.ml_rom.rom_lagrangian_ml.datasets import GraphSnapshotDataset
from cpfd_rom.ml_rom.rom_lagrangian_ml.evaluation import write_lagrangian_rom_only
from cpfd_rom.ml_rom.rom_lagrangian_ml.latent_regressor import (
    LatentRegressorMLP,
    build_latent_regression_dataloaders,
    train_latent_regressor_torch,
)
from cpfd_rom.ml_rom.rom_lagrangian_ml.model_pointnet_gnn import (
    PointNetGNNAutoencoder,
)
from cpfd_rom.ml_rom.rom_lagrangian_ml.training import train_pointnet_gnn_torch
from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)


def _normalize_graphs_in_place(graphs, feature_stats):
    eps = 1e-8
    pos_mean = feature_stats["pos_mean"]
    pos_std = feature_stats["pos_std"]
    fmin = float(feature_stats["field_min"])
    fmax = float(feature_stats["field_max"])

    for g in graphs:
        g.x[:, :3] = (g.x[:, :3] - pos_mean) / (pos_std + eps)
        g.x[:, 3:4] = (g.x[:, 3:4] - fmin) / (fmax - fmin + eps)

        if hasattr(g, "y") and g.y is not None:
            g.y[:, :3] = (g.y[:, :3] - pos_mean) / (pos_std + eps)
            g.y[:, 3:4] = (g.y[:, 3:4] - fmin) / (fmax - fmin + eps)


def _extract_graph_times(graphs) -> np.ndarray:
    return np.array([float(g.time.item()) for g in graphs], dtype=np.float64)


def _as_single_parameter_request_list(user_parameter) -> list[np.ndarray]:
    """
    Interpret config.user_parameter for a single-parameter Lagrangian ROM.

    Supported YAML examples:

        user_parameter: 0.300

    and:

        user_parameter: [0.300, 0.400, 0.500]

    Both are converted into a list of 1D parameter vectors:

        [array([0.300]), array([0.400]), array([0.500])]

    This intentionally does NOT interpret [0.300, 0.400, 0.500]
    as one 3-component parameter vector. This file is for the current
    single-parameter Lagrangian ROM only.
    """
    if np.isscalar(user_parameter):
        return [np.array([float(user_parameter)], dtype=np.float32)]

    arr = np.asarray(user_parameter, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("[Lagrangian/Raw] config.user_parameter is empty.")

    return [np.array([float(v)], dtype=np.float32) for v in arr]


def _format_rom_param_dir(user_param_vec: np.ndarray) -> str:
    """Return folder name such as ROM_param_0.300 for a scalar parameter."""
    vals = np.asarray(user_param_vec, dtype=np.float32).reshape(-1)
    if vals.size != 1:
        raise ValueError(
            f"[Lagrangian/Raw] Expected one scalar parameter, got shape {vals.shape}."
        )
    return f"ROM_param_{float(vals[0]):.3f}"


def _validate_reference_times(ref_times: np.ndarray) -> np.ndarray:
    ref_times = np.asarray(ref_times, dtype=np.float64).reshape(-1)
    if ref_times.size == 0:
        raise ValueError("[Lagrangian/Raw] Reference time axis is empty.")
    if ref_times.size >= 2 and not np.all(np.diff(ref_times) >= 0.0):
        raise ValueError("[Lagrangian/Raw] Reference times are not sorted.")
    return ref_times


def _log_template_alignment_summary(scaffold_graphs, matched_times):
    g0 = get_initial_template_graph(scaffold_graphs)
    logger.debug(
        "[TimingMatch] Template alignment summary: num_template_graphs=%d, "
        "first_template_time=%.6f, first_template_snapshot=%s, "
        "first_output_time=%.6f, last_output_time=%.6f",
        len(scaffold_graphs),
        float(g0.time.item()),
        getattr(g0, "snapshot_name", "<missing>"),
        float(matched_times[0]),
        float(matched_times[-1]),
    )


def _compute_time_bounds_from_graphs(graphs) -> tuple[float, float]:
    times = _extract_graph_times(graphs)
    t_min = float(times.min())
    t_max = float(times.max())
    if t_max <= t_min:
        raise ValueError("[Lagrangian/Raw] Invalid time range: all graph times are identical.")
    return t_min, t_max


def _normalize_time_array(times: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if t_max <= t_min:
        raise ValueError("[Lagrangian/Raw] Cannot normalize time: t_max <= t_min.")
    return ((times - t_min) / (t_max - t_min)).astype(np.float32)


def _denormalize_prediction(
    pos: torch.Tensor,
    field: torch.Tensor,
    feature_stats: dict | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_stats is None:
        return pos, field

    pos_mean = feature_stats["pos_mean"].to(device=device, dtype=pos.dtype)
    pos_std = feature_stats["pos_std"].to(device=device, dtype=pos.dtype)
    pos = pos * (pos_std + 1e-8) + pos_mean

    fmin = float(feature_stats["field_min"])
    fmax = float(feature_stats["field_max"])
    field = field * (fmax - fmin + 1e-8) + fmin
    return pos, field


def _extract_latents_and_regression_inputs(
    model,
    dataset,
    device,
    batch_size: int,
    t_min: float,
    t_max: float,
):
    latents = []
    reg_inputs = []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            recon, z = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.params,
                time=batch.time,
            )
            del recon

            latents.append(z.detach().cpu())

            time_np = batch.time.view(-1).detach().cpu().numpy()
            t_norm_np = _normalize_time_array(time_np, t_min=t_min, t_max=t_max)
            t_norm_t = torch.from_numpy(t_norm_np).view(-1, 1)

            phys_params = batch.params.detach().cpu()
            reg_inputs.append(torch.cat([phys_params, t_norm_t], dim=1))

    latents_all = torch.cat(latents, dim=0).numpy()
    reg_inputs_all = torch.cat(reg_inputs, dim=0).numpy()
    return latents_all, reg_inputs_all


def _infer_with_fourier_time(
    *,
    user_param,
    model: torch.nn.Module,
    latent_regressor: torch.nn.Module,
    template_graphs: list,
    feature_stats: dict | None,
    device: torch.device,
    batch_size: int,
    override_times,
    t_min: float,
    t_max: float,
):
    model = model.to(device).eval()
    latent_regressor = latent_regressor.to(device).eval()

    override_times = [float(t) for t in override_times]
    if len(template_graphs) != len(override_times):
        raise ValueError(
            f"[Inference] template_graphs ({len(template_graphs)}) and override_times "
            f"({len(override_times)}) must have the same length."
        )

    loader = DataLoader(
        template_graphs,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    preds_list: list[np.ndarray] = []
    times_out: list[float] = []

    user_param_vec = np.asarray(user_param, dtype=np.float32).reshape(-1)
    if user_param_vec.size != 1:
        raise ValueError(
            f"[Inference] This revised pipeline expects one scalar user parameter. "
            f"Got shape {user_param_vec.shape}."
        )

    graph_counter = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if not hasattr(batch, "params"):
                raise AttributeError("[Inference] batch is missing 'params'.")
            if not hasattr(batch, "time"):
                raise AttributeError("[Inference] batch is missing 'time'.")
            if not hasattr(batch, "cloud_id"):
                raise AttributeError("[Inference] batch is missing 'cloud_id'.")

            B = int(batch.params.shape[0])

            t_actual = torch.tensor(
                override_times[graph_counter:graph_counter + B],
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)

            t_norm = torch.tensor(
                _normalize_time_array(
                    np.asarray(
                        override_times[graph_counter:graph_counter + B],
                        dtype=np.float64,
                    ),
                    t_min=t_min,
                    t_max=t_max,
                ),
                dtype=torch.float32,
                device=device,
            ).view(-1, 1)

            user_phys = (
                torch.tensor(user_param_vec, dtype=torch.float32, device=device)
                .view(1, -1)
                .repeat(B, 1)
            )

            reg_in = torch.cat([user_phys, t_norm], dim=1)
            z_pred = latent_regressor(reg_in)

            recon = model.decode(
                latent_z=z_pred,
                template_x=batch.x[:, :4],
                edge_index=batch.edge_index,
                batch=batch.batch,
                params=user_phys,
                time=t_actual,
            )

            batch_vec = batch.batch

            for g_idx in range(B):
                mask = batch_vec == g_idx
                recon_g = recon[mask]

                pos = recon_g[:, :3]
                field = recon_g[:, 3:4]

                pos, field = _denormalize_prediction(
                    pos=pos,
                    field=field,
                    feature_stats=feature_stats,
                    device=device,
                )

                cid = batch.cloud_id[mask].to(device).float().view(-1, 1)
                out_g = torch.cat([pos, cid, field], dim=1)

                preds_list.append(out_g.cpu().numpy())

                tval = float(override_times[graph_counter])
                times_out.append(tval)

                if logger.isEnabledFor(logging.DEBUG) and graph_counter < 3:
                    logger.debug(
                        "[Inference][Graph %d] user_parameter=%.6f, "
                        "template_time=%.6f, output_time=%.6f, template_nodes=%d, "
                        "predicted_xyz_min=%s, predicted_xyz_max=%s, "
                        "predicted_field_minmax=(%.6e, %.6e)",
                        graph_counter,
                        float(user_param_vec[0]),
                        float(batch.time.view(-1)[g_idx].item()),
                        tval,
                        int(mask.sum().item()),
                        pos.min(dim=0).values.detach().cpu().numpy(),
                        pos.max(dim=0).values.detach().cpu().numpy(),
                        field.min().item(),
                        field.max().item(),
                    )

                graph_counter += 1

    return preds_list, times_out



# ------------------------------
# Lightweight inference artifacts
# ------------------------------

def _lagrangian_artifact_path(config) -> str:
    """Return the first-pass Lagrangian inference artifact path."""
    return os.path.join(config.model_dir, "lagrangian_inference_artifacts.pt")


def _save_lagrangian_inference_artifacts(
    *,
    config,
    feature_stats,
    t_min,
    t_max,
    param_dim,
    latent_reg_in_dim,
    latent_dim,
    hidden_dim,
    num_gnn_layers,
    num_time_bands,
    max_time_freq,
    include_raw_time,
    use_bn,
    graph_radius,
    sample_ratio,
    max_num_neighbors,
    batch_size,
    latent_reg_hidden_dims,
):
    """Save the minimum metadata needed for first-pass lightweight inference.

    This intentionally does not cache template/scaffold graphs yet. The first pass skips
    training-graph loading, dataset construction, AE latent extraction, and model training,
    while still extracting nearest-rev scaffold graphs during inference.
    """
    artifact_path = _lagrangian_artifact_path(config)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    payload = {
        "feature_stats": feature_stats,
        "t_min": float(t_min),
        "t_max": float(t_max),
        "param_dim": int(param_dim),
        "latent_reg_in_dim": int(latent_reg_in_dim),
        "latent_dim": int(latent_dim),
        "hidden_dim": int(hidden_dim),
        "num_gnn_layers": int(num_gnn_layers),
        "num_time_bands": int(num_time_bands),
        "max_time_freq": float(max_time_freq),
        "include_raw_time": bool(include_raw_time),
        "use_bn": bool(use_bn),
        "graph_radius": float(graph_radius),
        "sample_ratio": float(sample_ratio),
        "max_num_neighbors": int(max_num_neighbors),
        "batch_size": int(batch_size),
        "latent_reg_hidden_dims": list(latent_reg_hidden_dims),
        "field_variable": getattr(config, "field_variable", None),
        "type_of_field": getattr(config, "type_of_field", None),
        "rom_type": getattr(config, "rom_type", None),
    }

    torch.save(payload, artifact_path)
    logger.info("Wrote Lagrangian inference artifacts to %s", artifact_path)


def _load_lagrangian_inference_artifacts(config) -> dict:
    """Load first-pass Lagrangian inference artifact bundle."""
    artifact_path = _lagrangian_artifact_path(config)
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Missing Lagrangian inference artifact: {artifact_path}. "
            "Run a normal Lagrangian pipeline once before --infer-only."
        )

    logger.info("Loading Lagrangian inference artifacts from %s", artifact_path)
    return torch.load(artifact_path, map_location="cpu", weights_only=False)


def run_lagrangian_infer_only_pipeline(config, log_time, model_path: str, latent_reg_path: str):
    """First-pass lightweight Lagrangian inference path.

    This branch skips raw training graph loading, feature-stat computation from training
    graphs, dataset construction, AE latent extraction, and both training loops. It still
    extracts nearest-rev scaffold/template graphs because the current decoder path needs
    a template particle cloud for each output time.
    """
    logger.info("Lightweight Lagrangian inference mode enabled")
    detail(
        logger,
        "Skipping raw training graph loading, dataset construction, AE latent "
        "extraction, and training",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detail(logger, "Using device: %s", device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[INFER-ONLY] Missing AE model checkpoint: {model_path}")
    if not os.path.exists(latent_reg_path):
        raise FileNotFoundError(f"[INFER-ONLY] Missing latent regressor checkpoint: {latent_reg_path}")

    with log_time("Loading Lagrangian inference artifacts"):
        art = _load_lagrangian_inference_artifacts(config)

    feature_stats = art["feature_stats"]
    t_min = float(art["t_min"])
    t_max = float(art["t_max"])

    param_dim = int(art["param_dim"])
    latent_reg_in_dim = int(art["latent_reg_in_dim"])
    latent_dim = int(art["latent_dim"])
    hidden_dim = int(art["hidden_dim"])
    num_gnn_layers = int(art["num_gnn_layers"])
    num_time_bands = int(art["num_time_bands"])
    max_time_freq = float(art["max_time_freq"])
    include_raw_time = bool(art["include_raw_time"])
    use_bn = bool(art["use_bn"])

    graph_radius = float(art["graph_radius"])
    sample_ratio = float(art["sample_ratio"])
    max_num_neighbors = int(art["max_num_neighbors"])
    latent_reg_hidden_dims = art["latent_reg_hidden_dims"]

    if not getattr(config, "rev_dirs", None):
        raise ValueError("[Lagrangian/InferOnly] config.rev_dirs must be provided.")
    if not hasattr(config, "base_data_dir"):
        raise ValueError("[Lagrangian/InferOnly] config.base_data_dir must be provided.")
    if getattr(config, "param_mapping", None) is None:
        raise ValueError("[Lagrangian/InferOnly] config.param_mapping must be provided.")
    if getattr(config, "field_variable", None) is None:
        raise ValueError("[Lagrangian/InferOnly] config.field_variable must be provided.")
    if not hasattr(config, "user_parameter"):
        raise ValueError("[Lagrangian/InferOnly] config.user_parameter must be set.")

    user_param_requests = _as_single_parameter_request_list(config.user_parameter)
    detail(
        logger,
        "Requested Lagrangian scalar user_parameter values: %s",
        ", ".join(f"{float(p[0]):.6f}" for p in user_param_requests),
    )

    model = PointNetGNNAutoencoder(
        in_dim=4,
        param_dim=param_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        out_dim=4,
        num_gnn_layers=num_gnn_layers,
        num_time_bands=num_time_bands,
        max_time_freq=max_time_freq,
        include_raw_time=include_raw_time,
        use_bn=use_bn,
    ).to(device)

    with log_time("Loading pretrained AE model"):
        logger.info("Loading pretrained AE model from %s", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    latent_reg_model = LatentRegressorMLP(
        in_dim=latent_reg_in_dim,
        latent_dim=latent_dim,
        hidden_dims=latent_reg_hidden_dims,
    ).to(device)

    with log_time("Loading pretrained latent regressor"):
        logger.info("Loading pretrained latent regressor from %s", latent_reg_path)
        latent_reg_model.load_state_dict(torch.load(latent_reg_path, map_location=device))
    latent_reg_model.eval()

    reference_time_rev = getattr(config, "reference_time_rev", config.rev_dirs[0])

    ref_times = load_lagrangian_snapshot_times(
        rev_dir=reference_time_rev,
        base_data_dir=config.base_data_dir,
        sample_ratio=sample_ratio,
        field_variable=config.field_variable,
    )
    ref_times = _validate_reference_times(ref_times)

    detail(
        logger,
        "Using reference_time_rev='%s' with %d sorted reference times",
        reference_time_rev,
        len(ref_times),
    )

    lagrangian_root_out = os.path.join(config.output_dir, "Lagrangian_ROM")
    os.makedirs(lagrangian_root_out, exist_ok=True)

    for user_param_vec in user_param_requests:
        param_dir_name = _format_rom_param_dir(user_param_vec)
        out_dir = os.path.join(lagrangian_root_out, param_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(
            "Running Lagrangian ROM inference for user_parameter=%.6f",
            float(user_param_vec[0]),
        )
        detail(logger, "Output directory: %s", out_dir)

        with log_time(f"Extracting nearest-rev template graphs for {param_dir_name}"):
            template_graphs = extract_scaffold_graphs(
                rev_dirs=config.rev_dirs,
                base_data_dir=config.base_data_dir,
                param_mapping=config.param_mapping,
                param_train_array=None,
                user_param_array=user_param_vec,
                field_variable=config.field_variable,
                radius=graph_radius,
                sample_ratio=sample_ratio,
                feature_stats=feature_stats,
                max_num_neighbors=max_num_neighbors,
                verbose_timing=False,
            )

        template_graphs = sort_graphs_by_time(template_graphs)
        g0 = get_initial_template_graph(template_graphs)

        logger.debug(
            "[TemplateInit] user_parameter=%.6f, first_template_snapshot=%s, "
            "first_template_time=%.6f, num_template_graphs=%d",
            float(user_param_vec[0]),
            getattr(g0, "snapshot_name", "<missing>"),
            float(g0.time.item()),
            len(template_graphs),
        )

        if len(template_graphs) != len(ref_times):
            raise ValueError(
                f"[Lagrangian/InferOnly] template graphs ({len(template_graphs)}) and "
                f"reference times ({len(ref_times)}) have different lengths "
                f"for user_parameter={float(user_param_vec[0]):.6f}."
            )

        matched_graphs = list(template_graphs)
        matched_times = [float(t) for t in ref_times]

        _log_template_alignment_summary(matched_graphs, matched_times)

        del template_graphs
        gc.collect()

        detail(
            logger,
            "Index-aligned %d template graphs to %d reference times",
            len(matched_graphs),
            len(matched_times),
        )

        with log_time(f"Running Lagrangian ROM inference for {param_dir_name}"):
            preds, times = _infer_with_fourier_time(
                user_param=user_param_vec,
                model=model,
                latent_regressor=latent_reg_model,
                template_graphs=matched_graphs,
                feature_stats=feature_stats,
                device=device,
                batch_size=1,
                override_times=matched_times,
                t_min=t_min,
                t_max=t_max,
            )

        write_lagrangian_rom_only(
            preds=preds,
            times=times,
            field_variable=config.field_variable,
            out_dir=out_dir,
            time_mode="raw",
        )

        logger.info(
            "ROM inference complete for user_parameter=%.6f. Output saved to %s",
            float(user_param_vec[0]),
            out_dir,
        )

        del matched_graphs
        del preds
        del times
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "All Lagrangian ROM inference cases complete. Root output: %s",
        lagrangian_root_out,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None, None


def run_lagrangian_ml_pipeline(config, log_time):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detail(logger, "Using device: %s", device)

    setup_output_dir(config)
    setup_model_paths(config)

    model_path = os.path.join(config.model_dir, "pointnet_gnn_lagrangian.pt")
    latent_reg_path = os.path.join(config.model_dir, "latent_regressor.pt")

    if bool(getattr(config, "infer_only", False)):
        return run_lagrangian_infer_only_pipeline(
            config=config,
            log_time=log_time,
            model_path=model_path,
            latent_reg_path=latent_reg_path,
        )

    batch_size = int(getattr(config, "batch_size", 2))
    latent_dim = int(getattr(config, "latent_dim", 32))
    hidden_dim = int(getattr(config, "hidden_dim", 64))
    num_gnn_layers = int(getattr(config, "num_gnn_layers", 2))

    epochs = int(getattr(config, "epochs", 10))
    learning_rate = float(getattr(config, "learning_rate", 1e-3))
    sample_ratio = float(getattr(config, "sample_ratio", 0.05))
    graph_radius = float(getattr(config, "graph_radius", 0.001))
    max_num_neighbors = int(getattr(config, "max_num_neighbors", 16))

    grad_clip_norm = float(getattr(config, "grad_clip_norm", 1.0))
    field_loss_weight = float(getattr(config, "field_loss_weight", 1.0))
    com_loss_weight = float(getattr(config, "com_loss_weight", 0.0))
    spread_loss_weight = float(getattr(config, "spread_loss_weight", 0.0))

    latent_reg_epochs = int(getattr(config, "latent_reg_epochs", 50))
    latent_reg_lr = float(getattr(config, "latent_reg_lr", 1e-3))
    latent_reg_weight_decay = float(getattr(config, "latent_reg_weight_decay", 1e-4))
    latent_reg_patience = int(getattr(config, "latent_reg_patience", 5))
    latent_reg_hidden_dims = getattr(config, "latent_reg_hidden_dims", [128, 64])

    num_time_bands = int(getattr(config, "num_time_bands", 8))
    max_time_freq = float(getattr(config, "max_time_freq", 10.0))
    include_raw_time = bool(getattr(config, "include_raw_time", True))
    use_bn = bool(getattr(config, "use_bn", True))

    if not getattr(config, "rev_dirs", None):
        raise ValueError("[Lagrangian/Raw] config.rev_dirs must be provided.")
    if not hasattr(config, "base_data_dir"):
        raise ValueError("[Lagrangian/Raw] config.base_data_dir must be provided.")
    if getattr(config, "param_mapping", None) is None:
        raise ValueError("[Lagrangian/Raw] config.param_mapping must be provided.")
    if getattr(config, "field_variable", None) is None:
        raise ValueError("[Lagrangian/Raw] config.field_variable must be provided.")
    if not hasattr(config, "user_parameter"):
        raise ValueError("[Lagrangian/Raw] config.user_parameter must be set.")

    user_param_requests = _as_single_parameter_request_list(config.user_parameter)
    detail(
        logger,
        "Requested Lagrangian scalar user_parameter values: %s",
        ", ".join(f"{float(p[0]):.6f}" for p in user_param_requests),
    )

    with log_time("Loading raw-particle training graphs"):
        graphs = load_lagrangian_snapshots_as_graphs(
            rev_dirs=config.rev_dirs,
            base_data_dir=config.base_data_dir,
            param_mapping=config.param_mapping,
            field_variable=config.field_variable,
            radius=graph_radius,
            sample_ratio=sample_ratio,
            feature_stats=None,
            max_num_neighbors=max_num_neighbors,
            verbose_timing=False,
        )

        if len(graphs) == 0:
            raise RuntimeError("[Lagrangian/Raw] No graphs loaded.")

    t_min, t_max = _compute_time_bounds_from_graphs(graphs)
    detail(logger, "Training time range: [%.6f, %.6f]", t_min, t_max)

    with log_time("Computing feature stats and normalizing training graphs"):
        feature_stats = compute_feature_stats(graphs)
        _normalize_graphs_in_place(graphs, feature_stats)

    dataset = GraphSnapshotDataset(graphs)

    indices = np.arange(len(dataset))
    train_idx, _val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    train_set = torch.utils.data.Subset(dataset, train_idx)

    sample = dataset[0]
    param_dim = int(sample.params.shape[1])
    if param_dim != 1:
        raise ValueError(
            f"[Lagrangian/Raw] This revised pipeline supports single-parameter ROMs only. "
            f"The loaded dataset has param_dim={param_dim}."
        )

    latent_reg_in_dim = param_dim + 1

    model = PointNetGNNAutoencoder(
        in_dim=4,
        param_dim=param_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        out_dim=4,
        num_gnn_layers=num_gnn_layers,
        num_time_bands=num_time_bands,
        max_time_freq=max_time_freq,
        include_raw_time=include_raw_time,
        use_bn=use_bn,
    ).to(device)

    if getattr(config, "skip_training", False) and os.path.exists(model_path):
        with log_time("Loading pretrained AE model"):
            logger.info("Loading pretrained AE model from %s", model_path)
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        with log_time("Training PointNet-GNN autoencoder"):
            model = train_pointnet_gnn_torch(
                model=model,
                dataset=train_set,
                device=device,
                epochs=epochs,
                lr=learning_rate,
                batch_size=batch_size,
                grad_clip_norm=grad_clip_norm,
                field_loss_weight=field_loss_weight,
                com_loss_weight=com_loss_weight,
                spread_loss_weight=spread_loss_weight,
            )
        torch.save(model.state_dict(), model_path)
        logger.info("Saved AE model to %s", model_path)

    model.eval()

    with log_time("Extracting AE latents"):
        latents_all, reg_inputs_all = _extract_latents_and_regression_inputs(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            t_min=t_min,
            t_max=t_max,
        )

    detail(logger, "Latent matrix shape: %s", latents_all.shape)
    detail(logger, "Regressor input shape: %s", reg_inputs_all.shape)

    train_loader_reg, val_loader_reg = build_latent_regression_dataloaders(
        params_aug=reg_inputs_all,
        latents=latents_all,
        batch_size=batch_size,
        val_fraction=0.2,
        random_state=42,
        shuffle=True,
    )

    latent_reg_model = LatentRegressorMLP(
        in_dim=latent_reg_in_dim,
        latent_dim=latent_dim,
        hidden_dims=latent_reg_hidden_dims,
    ).to(device)

    if getattr(config, "skip_training", False) and os.path.exists(latent_reg_path):
        with log_time("Loading pretrained latent regressor"):
            logger.info("Loading pretrained latent regressor from %s", latent_reg_path)
            latent_reg_model.load_state_dict(torch.load(latent_reg_path, map_location=device))
    else:
        with log_time("Training latent regressor"):
            latent_reg_model = train_latent_regressor_torch(
                latent_reg_model,
                train_loader_reg,
                val_loader_reg,
                device=device,
                epochs=latent_reg_epochs,
                lr=latent_reg_lr,
                weight_decay=latent_reg_weight_decay,
                patience=latent_reg_patience,
            )
        torch.save(latent_reg_model.state_dict(), latent_reg_path)
        logger.info("Saved latent regressor to %s", latent_reg_path)

    latent_reg_model.eval()

    _save_lagrangian_inference_artifacts(
        config=config,
        feature_stats=feature_stats,
        t_min=t_min,
        t_max=t_max,
        param_dim=param_dim,
        latent_reg_in_dim=latent_reg_in_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        num_time_bands=num_time_bands,
        max_time_freq=max_time_freq,
        include_raw_time=include_raw_time,
        use_bn=use_bn,
        graph_radius=graph_radius,
        sample_ratio=sample_ratio,
        max_num_neighbors=max_num_neighbors,
        batch_size=batch_size,
        latent_reg_hidden_dims=latent_reg_hidden_dims,
    )

    del train_set
    del train_loader_reg
    del val_loader_reg
    del latents_all
    del reg_inputs_all
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reference_time_rev = getattr(config, "reference_time_rev", config.rev_dirs[0])

    ref_times = load_lagrangian_snapshot_times(
        rev_dir=reference_time_rev,
        base_data_dir=config.base_data_dir,
        sample_ratio=sample_ratio,
        field_variable=config.field_variable,
    )
    ref_times = _validate_reference_times(ref_times)

    detail(
        logger,
        "Using reference_time_rev='%s' with %d sorted reference times",
        reference_time_rev,
        len(ref_times),
    )

    lagrangian_root_out = os.path.join(config.output_dir, "Lagrangian_ROM")
    os.makedirs(lagrangian_root_out, exist_ok=True)

    for user_param_vec in user_param_requests:
        param_dir_name = _format_rom_param_dir(user_param_vec)
        out_dir = os.path.join(lagrangian_root_out, param_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(
            "Running Lagrangian ROM inference for user_parameter=%.6f",
            float(user_param_vec[0]),
        )
        detail(logger, "Output directory: %s", out_dir)

        with log_time(f"Extracting nearest-rev template graphs for {param_dir_name}"):
            template_graphs = extract_scaffold_graphs(
                rev_dirs=config.rev_dirs,
                base_data_dir=config.base_data_dir,
                param_mapping=config.param_mapping,
                param_train_array=None,
                user_param_array=user_param_vec,
                field_variable=config.field_variable,
                radius=graph_radius,
                sample_ratio=sample_ratio,
                feature_stats=feature_stats,
                max_num_neighbors=max_num_neighbors,
                verbose_timing=False,
            )

        template_graphs = sort_graphs_by_time(template_graphs)
        g0 = get_initial_template_graph(template_graphs)

        logger.debug(
            "[TemplateInit] user_parameter=%.6f, first_template_snapshot=%s, "
            "first_template_time=%.6f, num_template_graphs=%d",
            float(user_param_vec[0]),
            getattr(g0, "snapshot_name", "<missing>"),
            float(g0.time.item()),
            len(template_graphs),
        )

        if len(template_graphs) != len(ref_times):
            raise ValueError(
                f"[Lagrangian/Raw] template graphs ({len(template_graphs)}) and "
                f"reference times ({len(ref_times)}) have different lengths "
                f"for user_parameter={float(user_param_vec[0]):.6f}."
            )

        matched_graphs = list(template_graphs)
        matched_times = [float(t) for t in ref_times]

        _log_template_alignment_summary(matched_graphs, matched_times)

        del template_graphs
        gc.collect()

        detail(
            logger,
            "Index-aligned %d template graphs to %d reference times",
            len(matched_graphs),
            len(matched_times),
        )

        with log_time(f"Running Lagrangian ROM inference for {param_dir_name}"):
            preds, times = _infer_with_fourier_time(
                user_param=user_param_vec,
                model=model,
                latent_regressor=latent_reg_model,
                template_graphs=matched_graphs,
                feature_stats=feature_stats,
                device=device,
                batch_size=1,
                override_times=matched_times,
                t_min=t_min,
                t_max=t_max,
            )

        write_lagrangian_rom_only(
            preds=preds,
            times=times,
            field_variable=config.field_variable,
            out_dir=out_dir,
            time_mode="raw",
        )

        logger.info(
            "ROM inference complete for user_parameter=%.6f. Output saved to %s",
            float(user_param_vec[0]),
            out_dir,
        )

        del matched_graphs
        del preds
        del times
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del dataset
    del graphs
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "All Lagrangian ROM inference cases complete. Root output: %s",
        lagrangian_root_out,
    )
