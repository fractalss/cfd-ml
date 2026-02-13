# cpfd_rom/main_entry.py

from __future__ import annotations

import argparse
import gc
import os
import time
from contextlib import contextmanager
from typing import Any, Mapping, Optional

from cpfd_rom.util.config import load_config, overlay_cli


@contextmanager
def log_time(task_name: str):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"[Timing] {task_name} took {end - start:.2f} seconds")


def clear_memory():
    gc.collect()


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")


def _normalize_overrides(overrides: Mapping[str, Any]) -> dict:
    out = dict(overrides)

    # bools
    for k in ("add_time", "dry_run"):
        if k in out and out[k] is not None:
            out[k] = _truthy(out[k])

    # ints
    for k in ("fourier_m", "gat_heads"):
        if k in out and out[k] is not None:
            out[k] = int(out[k])

    # floats
    if "attn_dropout" in out and out["attn_dropout"] is not None:
        out["attn_dropout"] = float(out["attn_dropout"])
    # in _normalize_overrides
    if "dry_run" in out and out["dry_run"] is not None:
        out["dry_run"] = _truthy(out["dry_run"])

    return out


def run_from_config_path(config_path: str, overrides: Optional[Mapping[str, Any]] = None) -> int:
    """
    Stable internal entrypoint for the C++ wrapper to protect with licensing.

    Return codes:
      0 success
      1 failure
    """
    clear_memory()

    try:
        cfg = load_config(config_path)

        if overrides:
            cfg = overlay_cli(cfg, **_normalize_overrides(overrides))

        # Prepend base_data_dir to rev_dirs if relative
        if getattr(cfg, "base_data_dir", None) and getattr(cfg, "rev_dirs", None):
            cfg.rev_dirs = [
                d if os.path.isabs(d) else os.path.join(cfg.base_data_dir, d)
                for d in cfg.rev_dirs
            ]

        print(
            "[MAIN] Effective:",
            "dry_run=", getattr(cfg, "dry_run", None),
            "add_time=", getattr(cfg, "add_time", None),
            "time_mode=", getattr(cfg, "time_mode", None),
            "fourier_m=", getattr(cfg, "fourier_m", None),
            "conv_type=", getattr(cfg, "conv_type", None),
            "gat_heads=", getattr(cfg, "gat_heads", None),
            "attn_dropout=", getattr(cfg, "attn_dropout", None),
        )

        # Optional: clean green run without data/GPU
        if getattr(cfg, "dry_run", False):
            print("[MAIN] dry_run=true -> exiting before pipeline routing.")
            return 0

        # LAZY IMPORT + ROUTING
        rom_type = str(getattr(cfg, "rom_type", "")).strip()
        field_type = str(getattr(cfg, "type_of_field", "")).strip()

        if rom_type == "ML" and field_type == "Eulerian":
            from cpfd_rom.ml_rom.rom_eulerian_ml.pipeline import run_ml_rom_pipeline
            run_ml_rom_pipeline(cfg, log_time)

        elif rom_type == "ML" and field_type == "Lagrangian":
            from cpfd_rom.ml_rom.rom_lagrangian_ml.pipeline import run_lagrangian_ml_pipeline
            run_lagrangian_ml_pipeline(cfg, log_time)

        else:
            raise ValueError(
                f"Unsupported ROM type / field type: rom_type={rom_type}, type_of_field={field_type}"
            )

        clear_memory()
        return 0

    except Exception as e:
        print(f"[ERROR] {e}")
        clear_memory()
        return 1
