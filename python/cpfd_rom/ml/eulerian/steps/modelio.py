"""Model path normalization, model construction, and checkpoint I/O."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from cpfd_rom.util.logging_config import detail

from ..model_gnn import build_gcn_model


logger = logging.getLogger(__name__)


def torch_model_path(base_path: str) -> str:
    """Return a PyTorch checkpoint path derived from ``base_path``."""
    if base_path.endswith(".keras"):
        return base_path[:-6] + ".pt"
    if not base_path.endswith(".pt"):
        return base_path + ".pt"
    return base_path


def load_or_build_model(
    *,
    model_path_pt: str,
    in_dim: int,
    conv_type: str,
    hidden: int,
    dropout: float,
    gat_heads: int,
    attn_drop: float,
    early_stopping: bool,
    es_patience: int,
    es_min_delta: float,
    es_restore_best: bool,
    skip_training: bool,
):
    """Build the configured model and optionally load an existing checkpoint."""
    extra: dict[str, Any] = {}
    if conv_type == "gat":
        extra.update({"heads": gat_heads, "attn_dropout": attn_drop})

    detail(
        logger,
        "Building %s model: in_dim=%d, hidden=%d, dropout=%.4g",
        conv_type.upper(),
        in_dim,
        hidden,
        dropout,
    )

    model = build_gcn_model(
        in_dim=in_dim,
        hidden=hidden,
        out_dim=1,
        dropout=dropout,
        conv_type=conv_type,
        early_stopping=early_stopping,
        es_patience=es_patience,
        es_min_delta=es_min_delta,
        es_restore_best=es_restore_best,
        **extra,
    )

    if skip_training:
        if os.path.exists(model_path_pt):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            detail(logger, "Loading checkpoint on device %s", device)
            state = torch.load(model_path_pt, map_location=device)

            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
                detail(logger, "Loaded rich checkpoint containing state_dict")
            else:
                model.load_state_dict(state)

            logger.info("Loaded model checkpoint from %s", model_path_pt)
        else:
            logger.warning(
                "Training is disabled, but model checkpoint was not found: %s",
                model_path_pt,
            )

    return model


def save_state_dict(
    model,
    model_path_pt: str,
    hist: dict | None = None,
) -> None:
    """Save a model state dictionary and report optional training history."""
    torch.save(model.state_dict(), model_path_pt)
    logger.info("Saved model checkpoint to %s", model_path_pt)

    if hist is None:
        return

    try:
        val_mse = np.asarray(hist.get("val_mse", []), dtype=float)
        if val_mse.size == 0:
            detail(logger, "Training history contains no validation MSE values")
            return

        finite_val_mse = val_mse[np.isfinite(val_mse)]
        min_val = (
            float(np.min(finite_val_mse))
            if finite_val_mse.size
            else float("nan")
        )
        last_val = float(val_mse[-1])

        detail(
            logger,
            "Validation MSE: minimum=%.6e, final=%.6e",
            min_val,
            last_val,
        )
        detail(logger, "Training epochs completed: %d", val_mse.size)
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        logger.debug(
            "Could not summarize training history: %s",
            exc,
            exc_info=True,
        )
