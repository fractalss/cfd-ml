# cpfd_rom/main_entry.py
from __future__ import annotations

import gc
import os
import time
from contextlib import contextmanager

from cpfd_rom.util.config import load_config


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


def run_from_config_path(config_path: str) -> int:
    """
    Internal ROM engine entrypoint used by rom-cli-bin.

    Return codes:
      0 success
      1 failure
    """
    clear_memory()

    try:
        cfg = load_config(config_path)

        # Prepend base_data_dir to rev_dirs if relative
        if getattr(cfg, "base_data_dir", None) and getattr(cfg, "rev_dirs", None):
            cfg.rev_dirs = [
                d if os.path.isabs(d) else os.path.join(cfg.base_data_dir, d)
                for d in cfg.rev_dirs
            ]

        print(
            "[MAIN] Effective:",
            "rom_type=", getattr(cfg, "rom_type", None),
            "type_of_field=", getattr(cfg, "type_of_field", None),
            "field_variable=", getattr(cfg, "field_variable", None),
        )

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