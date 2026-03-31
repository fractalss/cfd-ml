# cpfd_rom/ml_rom/rom_lagrangian_ml/pipeline.py

from __future__ import annotations

import gc
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
from cpfd_rom.ml_rom.rom_lagrangian_ml.inference import (
    infer_pointnet_with_latent_regression,
)
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


def _normalize_graphs_in_place(graphs, feature_stats):
    """
    Normalize x/y in-place using feature_stats, while keeping .pos physical.

    Assumes:
      - g.x, g.y are [N,4] = [x,y,z,field]
      - g.pos is physical xyz
    """
    eps = 1e-8
    pos_mean = feature_stats["pos_mean"]
    pos_std = feature_stats["pos_std"]
    fmin = feature_stats["field_min"]
    fmax = feature_stats["field_max"]

    for g in graphs:
        g.x[:, :3] = (g.x[:, :3] - pos_mean) / (pos_std + eps)
        g.x[:, 3:4] = (g.x[:, 3:4] - fmin) / (fmax - fmin + eps)

        if hasattr(g, "y") and g.y is not None:
            g.y[:, :3] = (g.y[:, :3] - pos_mean) / (pos_std + eps)
            g.y[:, 3:4] = (g.y[:, 3:4] - fmin) / (fmax - fmin + eps)


def _extract_graph_times(graphs):
    return np.array([float(g.time.item()) for g in graphs], dtype=np.float64)


def _as_1d_user_param_array(user_parameter) -> np.ndarray:
    if np.isscalar(user_parameter):
        return np.array([float(user_parameter)], dtype=np.float32)
    return np.asarray(user_parameter, dtype=np.float32).reshape(-1)


def _validate_reference_times(ref_times: np.ndarray) -> np.ndarray:
    ref_times = np.asarray(ref_times, dtype=np.float64).reshape(-1)
    if ref_times.size == 0:
        raise ValueError("[Lagrangian/Raw] Reference time axis is empty.")
    if ref_times.size >= 2 and not np.all(np.diff(ref_times) >= 0.0):
        raise ValueError("[Lagrangian/Raw] Reference times are not sorted.")
    return ref_times


def _print_template_alignment_summary(scaffold_graphs, matched_times):
    g0 = get_initial_template_graph(scaffold_graphs)
    print("[TimingMatch] Template alignment summary")
    print(f"  num_template_graphs     : {len(scaffold_graphs)}")
    print(f"  first_template_time     : {float(g0.time.item()):.6f}")
    print(f"  first_template_snapshot : {getattr(g0, 'snapshot_name', '<missing>')}")
    print(f"  first_output_time       : {float(matched_times[0]):.6f}")
    print(f"  last_output_time        : {float(matched_times[-1]):.6f}")


def run_lagrangian_ml_pipeline(config, log_time):
    """
    Raw-particle Lagrangian ROM pipeline (old particle path, Step-2 cleaned).

    Contract
    --------
    - Input graphs come from Raw.particle.*.json + .npy
    - Model learns normalized [x, y, z, field]
    - Cloud ID is passed through from template graphs
    - Inference output rows are [x, y, z, Cloud ID, field]
    - Template graphs come from the nearest training rev
    - Template graphs are explicitly sorted by time
    - The nearest rev's true first snapshot is treated as the initialization template
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    setup_output_dir(config)
    setup_model_paths(config)

    model_path = os.path.join(config.model_dir, "pointnet_gnn_lagrangian.pt")
    latent_reg_path = os.path.join(config.model_dir, "latent_regressor.pt")

    # ----------------------------------------------------------
    # Config
    # ----------------------------------------------------------
    batch_size = int(getattr(config, "batch_size", 2))
    latent_dim = int(getattr(config, "latent_dim", 32))
    hidden_dim = int(getattr(config, "hidden_dim", 64))
    num_gnn_layers = int(getattr(config, "num_gnn_layers", 2))

    epochs = int(getattr(config, "epochs", 10))
    learning_rate = float(getattr(config, "learning_rate", 1e-3))
    sample_ratio = float(getattr(config, "sample_ratio", 0.05))
    graph_radius = float(getattr(config, "graph_radius", 0.001))

    # AE training knobs
    grad_clip_norm = float(getattr(config, "grad_clip_norm", 1.0))
    field_loss_weight = float(getattr(config, "field_loss_weight", 1.0))
    com_loss_weight = float(getattr(config, "com_loss_weight", 0.0))
    spread_loss_weight = float(getattr(config, "spread_loss_weight", 0.0))

    # Latent regressor knobs
    latent_reg_epochs = int(getattr(config, "latent_reg_epochs", 50))
    latent_reg_lr = float(getattr(config, "latent_reg_lr", 1e-3))
    latent_reg_weight_decay = float(getattr(config, "latent_reg_weight_decay", 1e-4))
    latent_reg_patience = int(getattr(config, "latent_reg_patience", 5))
    latent_reg_hidden_dims = getattr(config, "latent_reg_hidden_dims", [128, 64])

    # Inference/debug knobs
    inference_debug = bool(getattr(config, "inference_debug", True))
    inference_time_alignment_atol = float(
        getattr(config, "inference_time_alignment_atol", 1e-8)
    )

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

    user_param_arr = _as_1d_user_param_array(config.user_parameter)
    print(f"[INFO] user_parameter = {user_param_arr}")

    # ==========================================================
    # 1) LOAD TRAINING GRAPHS ONCE (RAW), COMPUTE STATS, NORMALIZE
    # ==========================================================
    with log_time("Loading raw-particle training graphs"):
        graphs = load_lagrangian_snapshots_as_graphs(
            rev_dirs=config.rev_dirs,
            base_data_dir=config.base_data_dir,
            param_mapping=config.param_mapping,
            field_variable=config.field_variable,
            radius=graph_radius,
            sample_ratio=sample_ratio,
            feature_stats=None,
            verbose_timing=False,
        )

        if len(graphs) == 0:
            raise RuntimeError("[Lagrangian/Raw] No graphs loaded.")

    with log_time("Computing feature stats and normalizing training graphs"):
        feature_stats = compute_feature_stats(graphs)
        _normalize_graphs_in_place(graphs, feature_stats)

    dataset = GraphSnapshotDataset(graphs)

    # ==========================================================
    # 2) TRAIN / VAL SPLIT
    # ==========================================================
    indices = np.arange(len(dataset))
    train_idx, _val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    train_set = torch.utils.data.Subset(dataset, train_idx)

    # ==========================================================
    # 3) BUILD AE
    # ==========================================================
    sample = dataset[0]
    param_dim = int(sample.params.shape[1])

    model = PointNetGNNAutoencoder(
        in_dim=4,
        param_dim=param_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        out_dim=4,
        num_gnn_layers=num_gnn_layers,
    ).to(device)

    # ==========================================================
    # 4) TRAIN AE (or load)
    # ==========================================================
    if getattr(config, "skip_training", False) and os.path.exists(model_path):
        with log_time("Loading pretrained AE model"):
            print(f"[INFO] Loading pretrained AE model from {model_path}")
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
        print(f"[INFO] Saved AE model to {model_path}")

    model.eval()

    # ==========================================================
    # 5) EXTRACT LATENTS FOR LATENT REGRESSION
    # ==========================================================
    latents = []
    params_all = []

    with log_time("Extracting AE latents"):
        loader_lat = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        with torch.no_grad():
            for batch in loader_lat:
                batch = batch.to(device)
                _recon, z = model(batch.x, batch.edge_index, batch.batch, batch.params)
                latents.append(z.detach().cpu())
                params_all.append(batch.params.detach().cpu())

    latents_all = torch.cat(latents, dim=0).numpy()
    params_all = torch.cat(params_all, dim=0).numpy()

    print(f"[INFO] Latent matrix shape     : {latents_all.shape}")
    print(f"[INFO] Params matrix shape     : {params_all.shape}")

    # ==========================================================
    # 6) TRAIN LATENT REGRESSOR
    # ==========================================================
    train_loader_reg, val_loader_reg = build_latent_regression_dataloaders(
        params_aug=params_all,
        latents=latents_all,
        batch_size=batch_size,
        val_fraction=0.2,
        random_state=42,
        shuffle=True,
    )

    latent_reg_model = LatentRegressorMLP(
        in_dim=param_dim,
        latent_dim=latent_dim,
        hidden_dims=latent_reg_hidden_dims,
    ).to(device)

    if getattr(config, "skip_training", False) and os.path.exists(latent_reg_path):
        with log_time("Loading pretrained latent regressor"):
            print(f"[INFO] Loading pretrained latent regressor from {latent_reg_path}")
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
        print(f"[INFO] Saved latent regressor to {latent_reg_path}")

    latent_reg_model.eval()

    # ----------------------------------------------------------
    # Cleanup training data memory
    # ----------------------------------------------------------
    del dataset
    del train_set
    del latents
    del latents_all
    del params_all
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    del graphs
    gc.collect()

    # ==========================================================
    # 7) EXTRACT TEMPLATE PARTICLE GRAPHS FROM NEAREST REV
    # ==========================================================
    with log_time("Extracting nearest-rev template graphs"):
        template_graphs = extract_scaffold_graphs(
            rev_dirs=config.rev_dirs,
            base_data_dir=config.base_data_dir,
            param_mapping=config.param_mapping,
            param_train_array=None,  # kept for API compatibility
            user_param_array=config.user_parameter,
            field_variable=config.field_variable,
            radius=graph_radius,
            sample_ratio=sample_ratio,
            feature_stats=feature_stats,
            verbose_timing=False,
        )

    # Explicit time sort and initialization confirmation
    template_graphs = sort_graphs_by_time(template_graphs)
    g0 = get_initial_template_graph(template_graphs)

    print("[TemplateInit] Nearest-rev template initialization confirmed")
    print(f"  first_template_snapshot : {getattr(g0, 'snapshot_name', '<missing>')}")
    print(f"  first_template_time     : {float(g0.time.item()):.6f}")
    print(f"  num_template_graphs     : {len(template_graphs)}")

    # ==========================================================
    # 7b) BUILD REFERENCE TIME AXIS
    # ==========================================================
    reference_time_rev = getattr(config, "reference_time_rev", config.rev_dirs[0])

    ref_times = load_lagrangian_snapshot_times(
        rev_dir=reference_time_rev,
        base_data_dir=config.base_data_dir,
        sample_ratio=sample_ratio,
    )
    ref_times = _validate_reference_times(ref_times)

    print(
        f"[TimingRef] Using reference_time_rev='{reference_time_rev}' "
        f"with {len(ref_times)} sorted reference times"
    )

    # ==========================================================
    # 7c) ALIGN TEMPLATE GRAPHS TO REFERENCE TIMES
    # ==========================================================
    if len(template_graphs) != len(ref_times):
        raise ValueError(
            f"[Lagrangian/Raw] template graphs ({len(template_graphs)}) and "
            f"reference times ({len(ref_times)}) have different lengths."
        )

    # For the old particle code path, keep strict index-based alignment,
    # but now after explicit sorting on both sides.
    matched_graphs = list(template_graphs)
    matched_times = [float(t) for t in ref_times]

    _print_template_alignment_summary(matched_graphs, matched_times)

    del template_graphs
    gc.collect()

    print(
        f"[TimingMatch] Index-aligned {len(matched_graphs)} template graphs "
        f"to {len(matched_times)} reference times"
    )

    # ==========================================================
    # 8) INFERENCE
    # ==========================================================
    with log_time("Running Lagrangian ROM inference"):
        preds, times = infer_pointnet_with_latent_regression(
            user_param=config.user_parameter,
            model=model,
            latent_regressor=latent_reg_model,
            scaffold_graphs=matched_graphs,
            output_dir=config.output_dir,
            field_variable=config.field_variable,
            feature_stats=feature_stats,
            device=device,
            batch_size=1,
            override_times=matched_times,
            time_alignment_atol=inference_time_alignment_atol,
            debug=inference_debug,
        )

    del matched_graphs
    gc.collect()

    # ==========================================================
    # 9) WRITE OUTPUT
    # ==========================================================
    out_dir = os.path.join(config.output_dir, "Lagrangian_ROM")
    os.makedirs(out_dir, exist_ok=True)

    write_lagrangian_rom_only(
        preds=preds,
        times=times,
        field_variable=config.field_variable,
        out_dir=out_dir,
        time_mode="raw",
    )

    print(f"[INFO] ROM inference complete. Output saved to: {out_dir}")