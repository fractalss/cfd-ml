# ===============================
# File: cpfd_rom/ml_rom/rom_eulerian_ml/pipeline_refactored/steps/modelio.py
# Purpose: Model path normalization, load/build, save
# ===============================
from __future__ import annotations
import os
import torch

from ..model_gnn import build_gcn_model


def torch_model_path(base_path: str) -> str:
    if base_path.endswith(".keras"):
        return base_path[:-6] + ".pt"
    if not base_path.endswith(".pt"):
        return base_path + ".pt"
    return base_path


def load_or_build_model(*, model_path_pt: str, in_dim: int, conv_type: str, hidden: int, dropout: float,
                        gat_heads: int, attn_drop: float, early_stopping: bool, es_patience: int,
                        es_min_delta: float, es_restore_best: bool, skip_training: bool):
    extra = {}
    if conv_type == "gat":
        extra.update({"heads": gat_heads, "attn_dropout": attn_drop})

    model = build_gcn_model(
        in_dim=in_dim,
        hidden=hidden,
        out_dim=1,
        dropout=dropout,
        conv_type=conv_type,
        early_stopping=early_stopping, es_patience=es_patience, es_min_delta=es_min_delta, es_restore_best=es_restore_best,
        **extra,
    )

    if skip_training and os.path.exists(model_path_pt):
        state = torch.load(model_path_pt, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])  # support rich checkpoints
        else:
            model.load_state_dict(state)
        print("[LOAD] checkpoint loaded")
    return model


def save_state_dict(model, model_path_pt: str, hist: dict | None = None):
    torch.save(model.state_dict(), model_path_pt)
    if hist is not None:
        try:
            import numpy as np
            min_val = float(np.nanmin(np.array(hist.get("val_mse", []))))
            last_val = float(hist.get("val_mse", [float('nan')])[-1])
            print(f"[TRAIN] min val_MSE={min_val:.6e}  last val_MSE={last_val:.6e}")
            print(f"[ES] epochs_ran={len(hist.get('val_mse', []))}")
        except Exception:
            pass
    print(f"[SAVE] wrote model state_dict to {model_path_pt}")
