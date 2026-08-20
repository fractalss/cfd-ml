# ===============================
# File: cpfd_rom/ml/eulerian/pipeline.py
# Purpose: Thin orchestrator that wires together modular Eulerian ML-ROM steps
#          with parameter-specific transient ROM output directories and metadata.
# ===============================
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
import torch

from cpfd_rom.util.model_utils import setup_model_paths
from cpfd_rom.util.output_utils import setup_output_dir
from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)

# graph + datasets
from .loader import prepare_graph_and_datasets, list_targets

# model + inference
from .model_gnn import build_gcn_model, train_gcn_model, _time_feature_dim, build_node_features_xyz
from .inference import predict_gcn

# alignment & ordering
from .steps.align import (
    maybe_load_coords_file_df,
    build_file_to_node_colmap,
    canonicalize_all,
    to_file_order,
)

# time configuration
from .steps.timecfg import resolve_time_features

# baseline & residual flow
from .steps.baseline import (
    fit_user_baseline,
    build_train_val_baseline,
    make_baseline_model,  # may not exist in older drops; fallback below if unavailable
)

# loss weighting
from .steps.weights import make_training_weights

# model i/o helpers
from .steps.modelio import torch_model_path, load_or_build_model, save_state_dict

# output writer
from .evaluation import _write_rom_only


# ------------------------------
# Local helpers  baseline
# ------------------------------

def _poly_design(p: float, deg: int) -> np.ndarray:
    """Return [1, p, p^2, ..., p^deg] as (D,) float32."""
    d = [1.0]
    for k in range(1, int(deg) + 1):
        d.append(float(p) ** k)
    return np.array(d, dtype=np.float32)


def _param_scalar(P: np.ndarray) -> np.ndarray:
    """Extract one scalar operating parameter per sample from P.

    Accepts:
      - P shaped (S, Pd)
      - P shaped (S, N, Pd)

    Returns:
      - p shaped (S,), using the first parameter column.
    """
    P = np.asarray(P)
    if P.ndim == 2:  # (S, Pd)
        return P[:, 0].astype(np.float32)
    if P.ndim == 3:  # (S, N, Pd)
        return P[:, :, 0].mean(axis=1).astype(np.float32)
    raise ValueError(f"Bad P shape {P.shape}; expected (S,Pd) or (S,N,Pd)")


class _BaselineFromW:
    """Lightweight baseline wrapper built from W_by_time.

    The object exposes the methods used by this pipeline:
      - predict_samples(times, P) -> (S, N)
      - predict_user(times, user_param) -> (S, N)

    W_by_time maps each time to polynomial coefficients shaped (D, N), where
    D = poly_deg + 1 and N is the number of Eulerian nodes/cells.
    """

    def __init__(self, W_by_time: dict[float, np.ndarray], poly_deg: int, atol: float = 1e-8):
        self.W_by_time = {float(k): np.asarray(v) for k, v in W_by_time.items()}
        self.poly_deg = int(poly_deg)
        self.atol = float(atol)
        self._times = np.array(sorted(self.W_by_time.keys()), dtype=np.float64)

    def _select_time(self, t: float) -> float:
        """Select exact time within tolerance, otherwise nearest available time."""
        diffs = np.abs(self._times - float(t))
        idx = int(np.argmin(diffs))
        return float(self._times[idx])

    def _eval_single(self, t: float, p_scalar: float) -> np.ndarray:
        t_key = self._select_time(t)
        W = self.W_by_time[t_key]  # (D, N)
        d = _poly_design(p_scalar, self.poly_deg).reshape(-1)  # (D,)
        return (d @ W).astype(np.float32)  # (N,)

    def predict_samples(self, times: np.ndarray, P: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float32).reshape(-1)
        p = _param_scalar(P)
        S = times.shape[0]

        y0 = self._eval_single(times[0], float(p[0]))
        out = np.empty((S, y0.shape[0]), dtype=np.float32)
        out[0] = y0

        for s in range(1, S):
            out[s] = self._eval_single(times[s], float(p[s]))

        return out

    def predict_user(self, times: np.ndarray, user_param: float) -> np.ndarray:
        times = np.asarray(times, dtype=np.float32).reshape(-1)
        S = times.shape[0]

        y0 = self._eval_single(times[0], float(user_param))
        out = np.empty((S, y0.shape[0]), dtype=np.float32)
        out[0] = y0

        for s in range(1, S):
            out[s] = self._eval_single(times[s], float(user_param))

        return out


def _maybe_build_baseline_features(
    *,
    use_feature: bool,
    baseline_model,
    times_train: Optional[np.ndarray],
    times_val: Optional[np.ndarray],
    P_train: np.ndarray,
    P_val: np.ndarray,
    user_param: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return baseline feature fields for train and validation.

    Returns:
      - B_train shaped (S_train, N) or None
      - B_val shaped (S_val, N) or None

    If baseline-as-feature is disabled or no baseline model exists, both are None.
    """
    if not use_feature or baseline_model is None:
        return None, None

    B_train = None
    if times_train is not None and len(times_train):
        B_train = baseline_model.predict_samples(times_train, P_train)

    B_val = None
    if times_val is not None and len(times_val):
        B_val = baseline_model.predict_samples(times_val, P_val)

    return B_train, B_val


# ------------------------------
# Local helpers  parameter-specific output paths and metadata
# ------------------------------

def _param_tag(p: float, ndigits: int = 3) -> str:
    """Return a filesystem-friendly operating-parameter tag.

    Examples:
      10.0   -> "10.000"
      42.25  -> "42.250"
      -5.0   -> "m5.000"
    """
    tag = f"{float(p):.{int(ndigits)}f}"
    return tag.replace("-", "m")


def _rom_param_output_dir(cfg, user_param: float) -> Path:
    """Return parameter-specific ROM output directory.

    This path is used by transient ML / residual-corrector ROM inference, so
    baseline+ transient ROM data is also associated with user_parameter.
    """
    ndigits = int(getattr(cfg, "param_tag_digits", 3))
    tag = _param_tag(user_param, ndigits=ndigits)
    return Path(cfg.output_dir) / "ML" / f"ROM_param_{tag}"


def _resolve_user_parameters(cfg) -> list[float]:
    """Resolve one or many inference parameters from config.

    Supported config forms:

      # Single operating point
      user_parameter: 45.0

      # Multiple operating points, legacy/user style
      user_parameter:
        - 35.0
        - 40.0
        - 45.0

      # Multiple operating points, explicit style
      user_parameters:
        - 35.0
        - 40.0
        - 45.0

    If user_parameters is present, it takes precedence over user_parameter.
    """
    vals = getattr(cfg, "user_parameters", None)

    if vals is None:
        vals = getattr(cfg, "user_parameter", 0.0)

    if vals is None:
        return [0.0]

    if isinstance(vals, (float, int, str)):
        return [float(vals)]

    if isinstance(vals, Sequence):
        return [float(v) for v in vals]

    raise ValueError(
        "cfg.user_parameter/user_parameters must be a scalar or sequence of scalars. "
        f"Got type={type(vals)!r}"
    )


def _write_inference_metadata(
    *,
    out_dir: Path,
    user_param: float,
    cfg,
    baseline_mode: str,
    use_residual: bool,
    use_baseline_as_feature: bool,
    time_cfg,
    model_path_pt: Path,
    train_param_min: float,
    train_param_max: float,
    y_min: float,
    y_max: float,
    n_nodes: int,
    n_edges: int,
    n_snapshots: int,
) -> None:
    """Write small self-describing metadata files inside each ROM output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "user_parameter": float(user_param),
        "field_variable": getattr(cfg, "field_variable", None),
        "rom_type": getattr(cfg, "rom_type", None),
        "type_of_field": getattr(cfg, "type_of_field", None),
        "baseline": baseline_mode,
        "use_residual_corrector": bool(use_residual),
        "use_baseline_as_feature": bool(use_baseline_as_feature),
        "time_features": {
            "add_time": bool(getattr(time_cfg, "add_time", False)),
            "time_mode": getattr(time_cfg, "time_mode", None),
            "fourier_m": int(getattr(time_cfg, "fourier_m", 0)),
            "time_gain": float(getattr(time_cfg, "time_gain", 1.0)),
        },
        "model": {
            "model_path_pt": str(model_path_pt),
            "conv_type": getattr(time_cfg, "conv_type", getattr(cfg, "conv_type", None)),
            "hidden": int(getattr(cfg, "hidden", 256)),
            "dropout": float(getattr(cfg, "dropout", 0.0)),
        },
        "training_ranges": {
            "parameter_min": float(train_param_min),
            "parameter_max": float(train_param_max),
            "target_min": float(y_min),
            "target_max": float(y_max),
            "user_inside_train_parameter_range": bool(train_param_min <= float(user_param) <= train_param_max),
        },
        "data_shape": {
            "n_nodes": int(n_nodes),
            "n_edges": int(n_edges),
            "n_snapshots_written": int(n_snapshots),
        },
    }

    json_path = out_dir / "rom_inference_meta.json"
    txt_path = out_dir / "rom_inference_meta.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"user_parameter = {float(user_param):.12g}\n")
        f.write(f"field_variable = {getattr(cfg, 'field_variable', None)}\n")
        f.write(f"rom_type = {getattr(cfg, 'rom_type', None)}\n")
        f.write(f"type_of_field = {getattr(cfg, 'type_of_field', None)}\n")
        f.write(f"baseline = {baseline_mode}\n")
        f.write(f"use_residual_corrector = {bool(use_residual)}\n")
        f.write(f"use_baseline_as_feature = {bool(use_baseline_as_feature)}\n")
        f.write(f"add_time = {bool(getattr(time_cfg, 'add_time', False))}\n")
        f.write(f"time_mode = {getattr(time_cfg, 'time_mode', None)}\n")
        f.write(f"fourier_m = {int(getattr(time_cfg, 'fourier_m', 0))}\n")
        f.write(f"time_gain = {float(getattr(time_cfg, 'time_gain', 1.0))}\n")
        f.write(f"model_path_pt = {str(model_path_pt)}\n")
        f.write(f"train_parameter_min = {float(train_param_min):.12g}\n")
        f.write(f"train_parameter_max = {float(train_param_max):.12g}\n")
        f.write(f"user_inside_train_parameter_range = {bool(train_param_min <= float(user_param) <= train_param_max)}\n")
        f.write(f"target_min = {float(y_min):.12g}\n")
        f.write(f"target_max = {float(y_max):.12g}\n")
        f.write(f"n_nodes = {int(n_nodes)}\n")
        f.write(f"n_edges = {int(n_edges)}\n")
        f.write(f"n_snapshots_written = {int(n_snapshots)}\n")



# ------------------------------
# Local helpers  lightweight Eulerian inference artifacts
# ------------------------------

def _model_dir_from_cfg(cfg) -> Path:
    """Return the directory used to store Eulerian model/inference artifacts."""
    raw = Path(getattr(cfg, "model_path_eulerian", "model_files"))
    return raw.parent if raw.suffix else raw


def _eulerian_artifact_path(cfg) -> Path:
    """Path to the cached bundle needed for lightweight --infer-only."""
    return _model_dir_from_cfg(cfg) / "eulerian_inference_artifacts.pt"


def _time_cfg_to_dict(time_cfg) -> dict:
    """Serialize the time feature configuration without depending on its class type."""
    if hasattr(time_cfg, "__dict__"):
        return dict(time_cfg.__dict__)
    return {
        "add_time": bool(getattr(time_cfg, "add_time", False)),
        "time_mode": getattr(time_cfg, "time_mode", "none"),
        "fourier_m": int(getattr(time_cfg, "fourier_m", 0)),
        "time_gain": float(getattr(time_cfg, "time_gain", 1.0)),
        "t_feat_dim": int(getattr(time_cfg, "t_feat_dim", 0)),
        "conv_type": str(getattr(time_cfg, "conv_type", getattr(time_cfg, "conv", "sage"))).lower(),
        "gat_heads": int(getattr(time_cfg, "gat_heads", 4)),
        "attn_drop": float(getattr(time_cfg, "attn_drop", 0.1)),
    }


def _save_eulerian_inference_artifacts(
    *,
    cfg,
    ref_rev,
    ref_graph_dir,
    nodes_df,
    edge_index,
    X_node,
    colmap_canon,
    time_cfg,
    times_train,
    p_mu,
    p_std,
    res_mu,
    res_std,
    y_lo: float,
    y_hi: float,
    pmin: float,
    pmax: float,
    baseline_mode: str,
    use_residual: bool,
    use_baseline_as_feature: bool,
    baseline_model,
    n_nodes: int,
    n_edges: int,
) -> None:
    """Persist the minimum cached state needed by lightweight --infer-only.

    This avoids reading raw cells_*.txt files, rebuilding graph artifacts, and
    regenerating snapshot target arrays when only new operating-parameter
    inference is requested.
    """
    artifact_path = _eulerian_artifact_path(cfg)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "field_variable": getattr(cfg, "field_variable", None),
        "type_of_field": getattr(cfg, "type_of_field", None),
        "rom_type": getattr(cfg, "rom_type", None),
        "ref_rev": str(ref_rev),
        "ref_graph_dir": str(ref_graph_dir),
        "nodes_df": nodes_df,
        "edge_index": edge_index.cpu() if hasattr(edge_index, "cpu") else torch.as_tensor(edge_index, dtype=torch.long),
        "X_node": X_node.cpu() if hasattr(X_node, "cpu") else torch.as_tensor(X_node, dtype=torch.float32),
        "colmap_canon": np.asarray(colmap_canon),
        "time_cfg": _time_cfg_to_dict(time_cfg),
        "times_train": np.asarray(times_train, dtype=np.float32) if times_train is not None else None,
        "p_mu": np.asarray(p_mu, dtype=np.float32),
        "p_std": np.asarray(p_std, dtype=np.float32),
        "res_mu": np.asarray(res_mu, dtype=np.float32),
        "res_std": np.asarray(res_std, dtype=np.float32),
        "y_lo": float(y_lo),
        "y_hi": float(y_hi),
        "pmin": float(pmin),
        "pmax": float(pmax),
        "baseline_mode": str(baseline_mode),
        "use_residual": bool(use_residual),
        "use_baseline_as_feature": bool(use_baseline_as_feature),
        "baseline_model": baseline_model,
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "model_config": {
            "conv_type": str(getattr(cfg, "conv_type", "sage")).lower(),
            "hidden": int(getattr(cfg, "hidden", 256)),
            "dropout": float(getattr(cfg, "dropout", 0.0)),
            "gat_heads": int(getattr(cfg, "gat_heads", 4)),
            "attn_dropout": float(getattr(cfg, "attn_dropout", 0.1)),
            "early_stopping": bool(getattr(cfg, "early_stopping", True)),
            "es_patience": int(getattr(cfg, "es_patience", 10)),
            "es_min_delta": float(getattr(cfg, "es_min_delta", 0.0)),
            "es_restore_best": bool(getattr(cfg, "es_restore_best", True)),
        },
    }

    try:
        torch.save(payload, artifact_path)
        detail(logger, "Wrote Eulerian inference artifacts to %s", artifact_path)
    except Exception as e:
        # Do not make a successful training run fail just because artifact caching
        # could not serialize an optional helper object such as a custom baseline.
        logger.warning("Failed to write Eulerian inference artifacts: %s", e)


def _load_eulerian_inference_artifacts(cfg) -> dict:
    artifact_path = _eulerian_artifact_path(cfg)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing Eulerian inference artifact: {artifact_path}. "
            "Run a normal training/full pipeline once before using --infer-only."
        )

    logger.info("Loading Eulerian inference artifacts from %s", artifact_path)
    return torch.load(artifact_path, map_location="cpu", weights_only=False)


def _run_eulerian_inference_outputs(
    *,
    cfg,
    log_time,
    model,
    model_path_pt: Path,
    ref_rev: str,
    edge_index,
    X_node,
    nodes_df,
    colmap_canon,
    time_cfg,
    times_train,
    user_parameters: list[float],
    p_mu,
    p_std,
    res_mu,
    res_std,
    y_lo: float,
    y_hi: float,
    pmin: float,
    pmax: float,
    baseline_mode: str,
    use_residual: bool,
    use_baseline_as_feature: bool,
    baseline_model,
    n_nodes: int,
    n_edges: int,
):
    """Shared inference writer used by full mode and lightweight --infer-only."""
    with log_time("Inference at user_parameter(s): parameter-specific ROM outputs"):
        out_root = Path(cfg.output_dir) / "graph"
        ref_dir = out_root / ref_rev
        targets = list_targets(ref_dir)
        times = np.array([t for t, _ in targets], dtype=float)

        logger.info(
            "Running Eulerian inference for %d snapshot(s) and %d operating parameter(s)",
            len(times),
            len(user_parameters),
        )
        detail(logger, "Inference operating parameters: %s", user_parameters)

        # Time statistics must match the training time feature construction.
        if time_cfg.add_time and times_train is not None and len(times_train) > 0:
            t_mu = float(np.mean(times_train))
            t_sigma = float(np.std(times_train)) if np.std(times_train) > 0 else 1.0
            t_min = float(np.min(times_train))
            t_max_train = float(np.max(times_train))
            t_max = t_max_train if t_max_train > t_min else (t_min + 1.0)
        else:
            t_mu = None
            t_sigma = None
            t_min, t_max = 0.0, 1.0

        ei_torch = torch.as_tensor(edge_index, dtype=torch.long)
        X_node_t = X_node.clone() if hasattr(X_node, "clone") else torch.tensor(X_node, dtype=torch.float32)

        # FILE-order node dataframe is independent of operating parameter.
        nodes_df_file = to_file_order(nodes_df, colmap_canon, dataframe=True)

        detail(logger, "Clipping predictions to training range [%.6e, %.6e]", y_lo, y_hi)

        for user_param_i in user_parameters:
            detail(logger, "Predicting user_parameter=%.6f", user_param_i)

            params_row = ((np.array([user_param_i], dtype=np.float32) - p_mu) / p_std).astype(np.float32)

            # Optional baseline feature channel at inference.
            baseline_feats = None
            if use_baseline_as_feature and (baseline_model is not None):
                baseline_feats = baseline_model.predict_user(times, user_param_i)  # (S, N)

            preds_list = []

            for s, t in enumerate(times):
                b_feat_row = baseline_feats[s] if baseline_feats is not None else None

                y = predict_gcn(
                    model,
                    edge_index=ei_torch,
                    X_node=X_node_t,
                    params_row=params_row,
                    add_time=time_cfg.add_time,
                    time_val=(float(t) if time_cfg.add_time else None),
                    time_mode=time_cfg.time_mode,
                    t_min=t_min,
                    t_max=t_max,
                    t_mu=t_mu,
                    t_sigma=t_sigma,
                    fourier_m=time_cfg.fourier_m,
                    time_gain=time_cfg.time_gain,
                    baseline_chan=b_feat_row,
                )

                preds_list.append(y.detach().cpu().numpy())

            preds = np.stack(preds_list, axis=0)  # (S, N)

            # De-normalize. If residual mode is enabled, this is the predicted residual field.
            preds = preds * res_std + res_mu

            # If residual mode, add the user-parameter-specific baseline field back.
            if use_residual and (baseline_model is not None):
                base_user = baseline_model.predict_user(times, user_param_i)
                preds = preds + base_user

            # Clip to training target range.
            np.clip(preds, y_lo, y_hi, out=preds)

            # NODE canonical order -> FILE order.
            preds_file = to_file_order(preds, colmap_canon)

            # Parameter-specific ROM directory for transient ML / baseline+ output.
            out_dir = _rom_param_output_dir(cfg, user_param_i)
            logger.info(
                "Writing Eulerian ROM output for user_parameter=%.6f to %s",
                user_param_i,
                out_dir,
            )

            _write_rom_only(
                preds_file,
                times,
                nodes_df_file,
                getattr(cfg, "field_variable", None),
                out_dir,
            )

            _write_inference_metadata(
                out_dir=out_dir,
                user_param=user_param_i,
                cfg=cfg,
                baseline_mode=baseline_mode,
                use_residual=use_residual,
                use_baseline_as_feature=use_baseline_as_feature,
                time_cfg=time_cfg,
                model_path_pt=Path(model_path_pt),
                train_param_min=pmin,
                train_param_max=pmax,
                y_min=y_lo,
                y_max=y_hi,
                n_nodes=n_nodes,
                n_edges=n_edges,
                n_snapshots=len(times),
            )


def run_ml_rom_infer_only_pipeline(cfg, log_time, model_path_pt: Path):
    """Lightweight Eulerian inference path.

    This path avoids graph rebuild, raw snapshot target generation, dataset
    construction, normalization fitting, and model training. It requires that a
    normal/full run has already written the checkpoint and inference artifact.
    """
    logger.info("Running lightweight Eulerian inference-only pipeline")
    detail(
        logger,
        "Skipping graph rebuild, snapshot target generation, dataset construction, and training",
    )

    from types import SimpleNamespace

    with log_time("Loading Eulerian inference artifacts"):
        art = _load_eulerian_inference_artifacts(cfg)

    user_parameters = _resolve_user_parameters(cfg)
    time_cfg = SimpleNamespace(**art["time_cfg"])

    edge_index = art["edge_index"]
    X_node = art["X_node"]
    nodes_df = art["nodes_df"]
    colmap_canon = art["colmap_canon"]

    p_mu = art["p_mu"]
    p_std = art["p_std"]
    res_mu = art["res_mu"]
    res_std = art["res_std"]

    use_baseline_as_feature = bool(art["use_baseline_as_feature"])
    use_residual = bool(art["use_residual"])
    baseline_model = art.get("baseline_model", None)

    p_dim = int(np.asarray(p_mu).reshape(-1).shape[0])
    in_dim = int(X_node.shape[1]) + p_dim + (int(time_cfg.t_feat_dim) if bool(time_cfg.add_time) else 0)
    if use_baseline_as_feature:
        in_dim += 1

    detail(
        logger,
        "Inference model dimensions: in_dim=%d, X=%d, P=%d, time=%d, baseline=%s",
        in_dim,
        X_node.shape[1],
        p_dim,
        int(time_cfg.t_feat_dim) if bool(time_cfg.add_time) else 0,
        "yes" if use_baseline_as_feature else "no",
    )

    mc = art.get("model_config", {})
    model = load_or_build_model(
        model_path_pt=model_path_pt,
        in_dim=in_dim,
        conv_type=str(mc.get("conv_type", getattr(cfg, "conv_type", "sage"))).lower(),
        hidden=int(mc.get("hidden", getattr(cfg, "hidden", 256))),
        dropout=float(mc.get("dropout", getattr(cfg, "dropout", 0.0))),
        gat_heads=int(mc.get("gat_heads", getattr(cfg, "gat_heads", 4))),
        attn_drop=float(mc.get("attn_dropout", getattr(cfg, "attn_dropout", 0.1))),
        early_stopping=bool(mc.get("early_stopping", getattr(cfg, "early_stopping", True))),
        es_patience=int(mc.get("es_patience", getattr(cfg, "es_patience", 10))),
        es_min_delta=float(mc.get("es_min_delta", getattr(cfg, "es_min_delta", 0.0))),
        es_restore_best=bool(mc.get("es_restore_best", getattr(cfg, "es_restore_best", True))),
        skip_training=True,
    )

    _run_eulerian_inference_outputs(
        cfg=cfg,
        log_time=log_time,
        model=model,
        model_path_pt=Path(model_path_pt),
        ref_rev=art["ref_rev"],
        edge_index=edge_index,
        X_node=X_node,
        nodes_df=nodes_df,
        colmap_canon=colmap_canon,
        time_cfg=time_cfg,
        times_train=art["times_train"],
        user_parameters=user_parameters,
        p_mu=p_mu,
        p_std=p_std,
        res_mu=res_mu,
        res_std=res_std,
        y_lo=float(art["y_lo"]),
        y_hi=float(art["y_hi"]),
        pmin=float(art["pmin"]),
        pmax=float(art["pmax"]),
        baseline_mode=str(art["baseline_mode"]),
        use_residual=use_residual,
        use_baseline_as_feature=use_baseline_as_feature,
        baseline_model=baseline_model,
        n_nodes=int(art["n_nodes"]),
        n_edges=int(art["n_edges"]),
    )

    return None, None


# ------------------------------
# Main pipeline
# ------------------------------

def run_ml_rom_pipeline(cfg, log_time):
    """Main entry point for Eulerian ML-ROM pipeline.

    This orchestrator supports:
      - baseline-only ROM path
      - residual-corrector / baseline+ transient GNN path
      - optional baseline-as-feature path
      - single or multiple inference operating parameters
      - parameter-specific transient ROM output directories
      - per-ROM metadata files
    """
    # --- Setup ---
    logger.info("Preparing Eulerian ML ROM pipeline")
    setup_output_dir(cfg)
    setup_model_paths(cfg)
    model_path_pt = torch_model_path(cfg.model_path_eulerian)

    # --- Lightweight inference-only path. Must occur before raw-data preparation. ---
    if bool(getattr(cfg, "infer_only", False)):
        return run_ml_rom_infer_only_pipeline(
            cfg=cfg,
            log_time=log_time,
            model_path_pt=model_path_pt,
        )

    # --- Build canonical graph + datasets. First rev defines graph. ---
    logger.info("Preparing Eulerian graph and datasets")
    with log_time("Preparing graph & datasets from rev_dirs"):
        (
            ref_rev,
            ref_graph_dir,
            nodes_df,
            edge_index,
            _X_node_loader,
            Y_train,
            Y_val,
            P_train,
            P_val,
            times_train,
            times_val,
        ) = prepare_graph_and_datasets(cfg.__dict__, neighbor_set="n6")

    # --- FILE -> NODE mapping based on coords_file_order, if available. ---
    coords_path = Path(ref_graph_dir) / "coords_file_order.parquet"
    coords_file_df = None

    if coords_path.exists():
        try:
            import pandas as pd

            coords_file_df = pd.read_parquet(coords_path)[["i", "j", "k"]]
            detail(logger, "Using coords_file_order.parquet for FILE-to-NODE mapping")
        except Exception as e:
            logger.warning("Failed to read %s: %s", coords_path.name, e)

    if coords_file_df is None:
        coords_file_df = maybe_load_coords_file_df(
            cfg,
            Path(ref_graph_dir) if ref_graph_dir else None,
            N_cols=(Y_train.shape[1] if Y_train.size else Y_val.shape[1]),
        )

    colmap_file_to_node = build_file_to_node_colmap(coords_file_df, nodes_df, Y_train)

    # --- Canonicalize (i,j,k) ordering and rebuild X_node accordingly. ---
    with log_time("Canonicalizing (i,j,k) order for nodes, Y_*, and edges"):
        nodes_df, Y_train, Y_val, edge_index, X_node, colmap_canon = canonicalize_all(
            nodes_df,
            Y_train,
            Y_val,
            edge_index,
            build_node_features_xyz,
            colmap_file_to_node,
        )

    # --- Dataset stats and parameter range. ---
    N = int(Y_train.shape[1]) if Y_train.size else 0
    E = int(edge_index.shape[1])
    S_tr, S_va = int(Y_train.shape[0]), int(Y_val.shape[0])
    P_dim = int(P_train.shape[1]) if P_train.size else 0

    detail(
        logger,
        "Dataset: nodes=%d, edges=%d, training samples=%d, validation samples=%d, parameter dimensions=%d",
        N,
        E,
        S_tr,
        S_va,
        P_dim,
    )
    detail(
        logger,
        "Dataset shapes: X_node=%s, Y_train=%s, P_train=%s",
        tuple(X_node.shape),
        tuple(Y_train.shape),
        tuple(P_train.shape),
    )

    user_parameters = _resolve_user_parameters(cfg)
    user_param = float(user_parameters[0])
    pmin, pmax = float(P_train.min()), float(P_train.max())
    detail(
        logger,
        "Training parameter range=[%.6f, %.6f]; first inference parameter=%.6f; inside_range=%s",
        pmin,
        pmax,
        user_param,
        pmin <= user_param <= pmax,
    )
    detail(logger, "Inference operating parameters: %s", user_parameters)

    # --- Normalization: target and parameter. ---
    y_mu = float(np.mean(Y_train))
    y_std_raw = float(np.std(Y_train))
    y_std = y_std_raw if y_std_raw > 0 else 1.0

    detail(logger, "Target normalization: mean=%.6e, std=%.6e (scalar)", y_mu, y_std)

    Y_train_n_scalar = (Y_train - y_mu) / y_std
    Y_val_n_scalar = (Y_val - y_mu) / y_std if Y_val.size else Y_val

    p_mu = P_train.mean(axis=0)
    p_std = P_train.std(axis=0)
    p_std = np.where(p_std > 0, p_std, 1.0)

    P_train_n = (P_train - p_mu) / p_std
    P_val_n = (P_val - p_mu) / p_std

    detail(
        logger,
        "Parameter normalization: mean=%s, std=%s",
        p_mu.ravel().tolist(),
        p_std.ravel().tolist(),
    )

    # --- Time feature configuration. ---
    time_cfg = resolve_time_features(cfg, times_train)

    # --- Flags. ---
    baseline_mode = str(getattr(cfg, "baseline", getattr(cfg, "baseline_mode", "none")) or "none").lower()
    use_residual = bool(getattr(cfg, "use_residual_corrector", False))
    use_baseline_as_feature = bool(getattr(cfg, "use_baseline_as_feature", False))

    # --- Optional baseline-only path. ---
    # Existing behavior is preserved. This path already attaches user_parameter through the
    # baseline writer/helper stack in older usage. The transient GNN path below now also
    # uses parameter-specific ROM directories.
    if baseline_mode in ("linear", "linear_over_param", "ridge") and not use_residual:
        with log_time("Baseline: linear over parameter (per time)"):
            _preds_file, _times = fit_user_baseline(
                user_param=user_param,
                ref_rev=ref_rev,
                output_dir=cfg.output_dir,
                Y_train=Y_train,
                Y_val=Y_val,
                P_train=P_train,
                P_val=P_val,
                times_train=times_train,
                times_val=times_val,
                colmap_canon=colmap_canon,
                nodes_df=nodes_df,
                field_variable=getattr(cfg, "field_variable", None),
                poly_deg=int(getattr(cfg, "baseline_poly_degree", 1)),
                ridge_alpha=float(getattr(cfg, "baseline_ridge_alpha", 0.0)),
                atol=float(getattr(cfg, "times_match_atol", 1e-8)),
            )
            return None, None

    # --- Residual-corrector path: fit baseline, train GNN on residuals. ---
    baseline_model = None
    base_tr_fields = None
    base_va_fields = None

    if use_residual:
        with log_time("Residual corrector: building per-time baseline on TRAIN/VAL"):
            poly_deg = int(getattr(cfg, "baseline_poly_degree", 1))
            ridge_alpha = float(getattr(cfg, "baseline_ridge_alpha", 0.0))
            atol = float(getattr(cfg, "times_match_atol", 1e-8))

            ret = build_train_val_baseline(
                times_train,
                P_train,
                Y_train,
                times_val,
                P_val,
                Y_val,
                poly_deg=poly_deg,
                ridge_alpha=ridge_alpha,
                atol=atol,
            )

            if isinstance(ret, tuple) and len(ret) == 3:
                W_by_time, baseline_model, _meta = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                W_by_time, _meta = ret
                try:
                    baseline_model = make_baseline_model(
                        W_by_time,
                        poly_deg=poly_deg,
                        ridge_alpha=ridge_alpha,
                        atol=atol,
                    )
                except Exception:
                    baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)
            else:
                W_by_time = ret
                try:
                    baseline_model = make_baseline_model(
                        W_by_time,
                        poly_deg=poly_deg,
                        ridge_alpha=ridge_alpha,
                        atol=atol,
                    )
                except Exception:
                    baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)

            base_tr_fields = None
            if times_train is not None and len(times_train):
                base_tr_fields = baseline_model.predict_samples(times_train, P_train)

            base_va_fields = None
            if times_val is not None and len(times_val):
                base_va_fields = baseline_model.predict_samples(times_val, P_val)

            detail(
                logger,
                "Baseline field shapes: training=%s, validation=%s",
                None if base_tr_fields is None else base_tr_fields.shape,
                None if base_va_fields is None else base_va_fields.shape,
            )

            if base_tr_fields is None:
                raise RuntimeError("Baseline fields for training could not be constructed.")

            if base_va_fields is None:
                logger.warning("No validation baseline fields; using zeros as a fallback")
                base_va_fields = np.zeros_like(Y_val, dtype=np.float32)

            R_tr = (Y_train - base_tr_fields).astype(np.float32)
            R_va = (Y_val - base_va_fields).astype(np.float32)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Raw training residual mean/std: %.3e/%.3e", R_tr.mean(), R_tr.std())
                logger.debug("Raw validation residual mean/std: %.3e/%.3e", R_va.mean(), R_va.std())
                logger.debug(
                    "Residual node std range: training=[%.3e, %.3e], validation=[%.3e, %.3e]",
                    np.min(R_tr.std(axis=0)),
                    np.max(R_tr.std(axis=0)),
                    np.min(R_va.std(axis=0)),
                    np.max(R_va.std(axis=0)),
                )
                snap_tr_std = np.std(R_tr, axis=1)
                snap_va_std = np.std(R_va, axis=1)
                logger.debug(
                    "Residual snapshot std range: training=[%.3e, %.3e], validation=[%.3e, %.3e]",
                    snap_tr_std.min(),
                    snap_tr_std.max(),
                    snap_va_std.min(),
                    snap_va_std.max(),
                )

            pernode = bool(getattr(cfg, "residual_pernode_norm", True))
            eps = 1e-8

            if pernode:
                mu = R_tr.mean(axis=0)  # (N,)
                std = R_tr.std(axis=0) + eps  # (N,)

                small_nodes = int(np.sum(std < 1e-2))
                logger.debug(
                    "Residual node std: min=%.3e, max=%.3e, mean=%.3e; %d/%d nodes below 1e-2",
                    std.min(),
                    std.max(),
                    std.mean(),
                    small_nodes,
                    std.size,
                )

                std_safe = np.where(std < 1e-2, 1e-2, std)

                Y_train_n = (R_tr - mu) / std_safe
                Y_val_n = (R_va - mu) / std_safe
                res_mu = mu.astype(np.float32)
                res_std = std_safe.astype(np.float32)

                detail(logger, "Residual corrector enabled with per-node normalization")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Safe residual std: min=%.3e, max=%.3e, mean=%.3e",
                        std_safe.min(),
                        std_safe.max(),
                        std_safe.mean(),
                    )
            else:
                mu = float(R_tr.mean())
                std = float(R_tr.std() + eps)

                Y_train_n = (R_tr - mu) / std
                Y_val_n = (R_va - mu) / std
                res_mu = np.array(mu, dtype=np.float32)
                res_std = np.array(std, dtype=np.float32)

                detail(logger, "Residual corrector enabled with scalar normalization")

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Normalized residual mean/std: training=%.3e/%.3e, validation=%.3e/%.3e",
                    Y_train_n.mean(),
                    Y_train_n.std(),
                    Y_val_n.mean(),
                    Y_val_n.std(),
                )

    else:
        # No residual path: train directly on scalar-normalized target fields.
        Y_train_n, Y_val_n = Y_train_n_scalar, Y_val_n_scalar
        res_mu = y_mu
        res_std = y_std

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Normalized training target mean/std: %.3e/%.3e",
                Y_train_n.mean(),
                Y_train_n.std(),
            )
            if Y_val_n is not None and isinstance(Y_val_n, np.ndarray) and Y_val_n.size:
                logger.debug(
                    "Normalized validation target mean/std: %.3e/%.3e",
                    Y_val_n.mean(),
                    Y_val_n.std(),
                )

    # --- Ensure baseline exists if baseline-as-feature is requested, even without residuals. ---
    if baseline_model is None and use_baseline_as_feature:
        poly_deg = int(getattr(cfg, "baseline_poly_degree", 1))
        ridge_alpha = float(getattr(cfg, "baseline_ridge_alpha", 0.0))
        atol = float(getattr(cfg, "times_match_atol", 1e-8))

        ret = build_train_val_baseline(
            times_train,
            P_train,
            Y_train,
            times_val,
            P_val,
            Y_val,
            poly_deg=poly_deg,
            ridge_alpha=ridge_alpha,
            atol=atol,
        )

        if isinstance(ret, tuple) and len(ret) == 3:
            W_by_time, baseline_model, _meta = ret
        else:
            W_by_time = ret[0] if isinstance(ret, tuple) else ret
            try:
                baseline_model = make_baseline_model(
                    W_by_time,
                    poly_deg=poly_deg,
                    ridge_alpha=ridge_alpha,
                    atol=atol,
                )
            except Exception:
                baseline_model = _BaselineFromW(W_by_time, poly_deg=poly_deg, atol=atol)

    # --- Baseline-as-feature channels for TRAIN/VAL. ---
    B_tr, B_va = _maybe_build_baseline_features(
        use_feature=use_baseline_as_feature,
        baseline_model=baseline_model,
        times_train=times_train,
        times_val=times_val,
        P_train=P_train,
        P_val=P_val,
        user_param=user_param,
    )

    # Model training expects optional baseline arrays shaped (S, N, 1).
    if B_tr is not None and B_tr.ndim == 2:
        B_tr = B_tr[..., None]
    if B_va is not None and B_va.ndim == 2:
        B_va = B_va[..., None]

    # --- Training weights. ---
    node_weights = None
    sample_weights = None
    loss_weighting = str(getattr(cfg, "loss_weighting", "none")).strip().lower()

    if loss_weighting not in (None, "", "none"):
        node_weights, sample_weights = make_training_weights(
            strategy=loss_weighting,
            Y_train=Y_train,
            Y_val=Y_val,
            times_train=times_train,
            times_val=times_val,
            P_train=P_train,
            P_val=P_val,
            W_by_time=None,
        )

    # --- Train or load model. ---
    in_dim = X_node.shape[1] + P_train_n.shape[1] + (time_cfg.t_feat_dim if time_cfg.add_time else 0)
    if B_tr is not None:
        in_dim += 1  # baseline scalar field as extra node feature

    detail(
        logger,
        "Training model dimensions: in_dim=%d, X=%d, P=%d, time=%d, baseline=%s",
        in_dim,
        X_node.shape[1],
        P_train_n.shape[1],
        time_cfg.t_feat_dim if time_cfg.add_time else 0,
        "yes" if B_tr is not None else "no",
    )

    model = load_or_build_model(
        model_path_pt=model_path_pt,
        in_dim=in_dim,
        conv_type=str(getattr(cfg, "conv_type", "sage")).lower(),
        hidden=int(getattr(cfg, "hidden", 256)),
        dropout=float(getattr(cfg, "dropout", 0.0)),
        gat_heads=int(getattr(cfg, "gat_heads", 4)),
        attn_drop=float(getattr(cfg, "attn_dropout", 0.1)),
        early_stopping=bool(getattr(cfg, "early_stopping", True)),
        es_patience=int(getattr(cfg, "es_patience", 10)),
        es_min_delta=float(getattr(cfg, "es_min_delta", 0.0)),
        es_restore_best=bool(getattr(cfg, "es_restore_best", True)),
        skip_training=bool(getattr(cfg, "skip_training", False)),
    )

    if not (bool(getattr(cfg, "skip_training", False)) and os.path.exists(model_path_pt)):
        logger.info("Training Eulerian GNN model")
        with log_time("Training GNN (graph-first)"):
            Y_tr = Y_train_n
            Y_va = Y_val_n

            assert isinstance(Y_tr, np.ndarray) and Y_tr.ndim == 2, (
                f"Y_tr must be (S_train, N); got {None if Y_tr is None else Y_tr.shape}"
            )
            assert isinstance(Y_va, np.ndarray) and Y_va.ndim == 2, (
                f"Y_va must be (S_val, N); got {None if Y_va is None else Y_va.shape}"
            )
            assert Y_tr.shape[1] == Y_va.shape[1], (
                f"Node dim mismatch: Y_tr N={Y_tr.shape[1]} vs Y_va N={Y_va.shape[1]}"
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Pipeline normalized targets: training mean/std=%.3e/%.3e, validation mean/std=%.3e/%.3e",
                    Y_tr.mean(),
                    Y_tr.std(),
                    Y_va.mean(),
                    Y_va.std(),
                )

            shuffle_seed = getattr(cfg, "shuffle_seed", None)

            model, hist = train_gcn_model(
                Y_tr,
                Y_va,
                P_train_n,
                P_val_n,
                edge_index,
                X_node=X_node,
                epochs=int(getattr(cfg, "epochs", 10)),
                lr=float(getattr(cfg, "lr", 1e-3)),
                hidden=int(getattr(cfg, "hidden", 256)),
                dropout=float(getattr(cfg, "dropout", 0.0)),
                lambda_smooth=float(getattr(cfg, "lambda_smooth", 0.0)),
                ignore_head_frac=float(getattr(cfg, "ignore_head_frac", 0.0)),
                shuffle=True,
                shuffle_seed=shuffle_seed,
                device=None,
                add_time=time_cfg.add_time,
                times_train=times_train,
                times_test=times_val,
                time_mode=time_cfg.time_mode,
                fourier_m=time_cfg.fourier_m,
                time_gain=time_cfg.time_gain,
                conv_type=time_cfg.conv_type,
                heads=time_cfg.gat_heads,
                attn_dropout=time_cfg.attn_drop,
                baseline_train=B_tr,
                baseline_val=B_va,
                node_weights=node_weights,
                sample_weights=sample_weights,
            )

            save_state_dict(model, model_path_pt, hist)

    # --- Save lightweight inference artifacts for future --infer-only runs. ---
    y_lo = float(np.min(Y_train))
    y_hi = float(np.max(Y_train))

    _save_eulerian_inference_artifacts(
        cfg=cfg,
        ref_rev=ref_rev,
        ref_graph_dir=ref_graph_dir,
        nodes_df=nodes_df,
        edge_index=edge_index,
        X_node=X_node,
        colmap_canon=colmap_canon,
        time_cfg=time_cfg,
        times_train=times_train,
        p_mu=p_mu,
        p_std=p_std,
        res_mu=res_mu,
        res_std=res_std,
        y_lo=y_lo,
        y_hi=y_hi,
        pmin=pmin,
        pmax=pmax,
        baseline_mode=baseline_mode,
        use_residual=use_residual,
        use_baseline_as_feature=use_baseline_as_feature,
        baseline_model=baseline_model,
        n_nodes=N,
        n_edges=E,
    )

    # --- Inference at one or many user parameters. ---
    _run_eulerian_inference_outputs(
        cfg=cfg,
        log_time=log_time,
        model=model,
        model_path_pt=Path(model_path_pt),
        ref_rev=ref_rev,
        edge_index=edge_index,
        X_node=X_node,
        nodes_df=nodes_df,
        colmap_canon=colmap_canon,
        time_cfg=time_cfg,
        times_train=times_train,
        user_parameters=user_parameters,
        p_mu=p_mu,
        p_std=p_std,
        res_mu=res_mu,
        res_std=res_std,
        y_lo=y_lo,
        y_hi=y_hi,
        pmin=pmin,
        pmax=pmax,
        baseline_mode=baseline_mode,
        use_residual=use_residual,
        use_baseline_as_feature=use_baseline_as_feature,
        baseline_model=baseline_model,
        n_nodes=N,
        n_edges=E,
    )

    return None, None
