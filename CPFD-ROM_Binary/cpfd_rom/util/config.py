# User-configurable parameters + robust YAML loader for the ROM pipeline
# Updated to match the revised `rom_inputs.yaml` schema (top-level keys)
# and to play nicely with CLI overrides.

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

# --------------------------------------------------------------------------------------
# Back-compat module-level placeholders (some legacy code may import these)
# --------------------------------------------------------------------------------------
rom_type: Optional[str] = None
field_variable: Optional[str] = None
type_of_field: Optional[str] = None
user_parameter: Optional[float] = None
target_times: Optional[Iterable[float]] = None
base_data_dir: Optional[str] = None
rev_dirs: Optional[Iterable[str]] = None
param_mapping: Optional[Mapping[str, float]] = None
test_directory: Optional[str] = None
test_df: Any = None
test_times: Optional[Iterable[float]] = None
model_path: Optional[str] = None

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _lower_strip(v: Any, default: str = "") -> str:
    return str(v).strip().lower() if v is not None else default


# --------------------------------------------------------------------------------------
# Config object
# --------------------------------------------------------------------------------------

class ROMConfig(SimpleNamespace):
    """Lightweight config container with sensible defaults and normalization.

    Fields are attached as attributes so existing code can use `getattr(cfg, "key", default)`.
    """

    # Normalization for time features
    def _normalize_time(self) -> None:
        tm = _lower_strip(getattr(self, "time_mode", None), "none")
        fm = _to_int(getattr(self, "fourier_m", 4), 4)
        setattr(self, "time_mode", tm)
        setattr(self, "fourier_m", fm)
        # Derive add_time if not explicitly provided
        if hasattr(self, "add_time"):
            setattr(self, "add_time", _to_bool(getattr(self, "add_time")))
        else:
            setattr(self, "add_time", tm != "none")

    # Normalization for backbone
    def _normalize_backbone(self) -> None:
        ct = _lower_strip(getattr(self, "conv_type", None), "sage")
        setattr(self, "conv_type", ct)
        setattr(self, "gat_heads", _to_int(getattr(self, "gat_heads", 4), 4))
        setattr(self, "attn_dropout", float(getattr(self, "attn_dropout", 0.1)))

    # Numeric casts for key fields
    def _normalize_core(self) -> None:
        if hasattr(self, "user_parameter"):
            setattr(self, "user_parameter", _to_float(getattr(self, "user_parameter")))
        # param_mapping ? float values
        pm = getattr(self, "param_mapping", None)
        if isinstance(pm, dict):
            setattr(self, "param_mapping", {str(k): _to_float(v) for k, v in pm.items()})
        # rev_dirs ? list of strings
        rd = getattr(self, "rev_dirs", None)
        if rd is not None and not isinstance(rd, (list, tuple)):
            setattr(self, "rev_dirs", [str(rd)])

        # booleans
        setattr(self, "skip_training", _to_bool(getattr(self, "skip_training", False)))
        setattr(self, "rebuild_graph", _to_bool(getattr(self, "rebuild_graph", False)))

    def normalize(self) -> "ROMConfig":
        self._normalize_time()
        self._normalize_backbone()
        self._normalize_core()
        return self

    # Convenience: dict view
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------------------
# YAML loader and CLI overlay
# --------------------------------------------------------------------------------------

def load_config(path: str) -> ROMConfig:
    """Load YAML at `path` and return a normalized ROMConfig.

    Accepts the revised schema from `rom_inputs.yaml` with top-level keys like:
      - rom_type, type_of_field, field_variable, user_parameter
      - base_data_dir, rev_dirs, param_mapping
      - conv_type, gat_heads, attn_dropout
      - time_mode, fourier_m, (optional) add_time
      - skip_training, rebuild_graph
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = ROMConfig(**data)
    return cfg.normalize()


def overlay_cli(cfg: ROMConfig, **overrides: Any) -> ROMConfig:
    """Overlay CLI-style keyword overrides onto an existing config and normalize."""
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        # Coerce common keys
        if k in {"user_parameter", "fourier_m", "gat_heads"}:
            v = _to_int(v) if k in {"fourier_m", "gat_heads"} else _to_float(v)
        if k in {"add_time", "skip_training", "rebuild_graph"}:
            v = _to_bool(v)
        if k in {"conv_type", "time_mode"}:
            v = _lower_strip(v)
        setattr(cfg, k, v)
    return cfg.normalize()


# --------------------------------------------------------------------------------------
# Optional: module-level loader (legacy)
# --------------------------------------------------------------------------------------

def load_into_module_globals(path: str) -> ROMConfig:
    """Load config and also populate legacy module globals for back-compat."""
    cfg = load_config(path)
    globals().update({
        "rom_type": getattr(cfg, "rom_type", None),
        "field_variable": getattr(cfg, "field_variable", None),
        "type_of_field": getattr(cfg, "type_of_field", None),
        "user_parameter": getattr(cfg, "user_parameter", None),
        "base_data_dir": getattr(cfg, "base_data_dir", None),
        "rev_dirs": getattr(cfg, "rev_dirs", None),
        "param_mapping": getattr(cfg, "param_mapping", None),
        "model_path": getattr(cfg, "model_path", None),
    })
    return cfg
