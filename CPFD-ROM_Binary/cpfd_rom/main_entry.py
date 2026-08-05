# cpfd_rom/main_entry.py
from __future__ import annotations

import gc
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from cpfd_rom.util.config import load_config
from cpfd_rom.util.logging_config import detail


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set global random seeds for reproducible or ensemble training.

    Different seed values produce different model initializations and
    training trajectories. This is useful for ensemble-based UQ.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Keep these False for performance unless strict reproducibility is needed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    logger.debug("Global seed set to %d", seed)


@contextmanager
def log_time(task_name: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info("%s completed in %.2f seconds", task_name, elapsed)


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_from_config_path(config_path: str, infer_only: bool = False) -> int:
    """
    Internal ROM engine entrypoint used by rom-cli-bin.

    Parameters
    ----------
    config_path:
        Path to ROM YAML configuration file.

    infer_only:
        If True, skip model training and run inference using an existing model.

    Return codes
    ------------
    0:
        Success.
    1:
        Failure.
    """
    clear_memory()

    try:
        logger.info("Loading ROM configuration")

        cfg = load_config(config_path)

        # CLI/YAML-level mode override
        effective_infer_only = bool(
            infer_only or getattr(cfg, "infer_only", False)
        )

        if effective_infer_only:
            setattr(cfg, "infer_only", True)
            setattr(cfg, "skip_training", True)
            setattr(cfg, "rebuild_graph", False)
            setattr(cfg, "rebuild_targets", False)
            setattr(cfg, "use_cached_artifacts", True)
        else:
            setattr(cfg, "infer_only", False)

        # Global seed for reproducibility and UQ ensemble runs
        seed = int(
            getattr(cfg, "seed", getattr(cfg, "shuffle_seed", 42))
        )
        set_seed(seed)

        # Prepend base_data_dir to relative revision directories
        if getattr(cfg, "base_data_dir", None) and getattr(
            cfg, "rev_dirs", None
        ):
            cfg.rev_dirs = [
                directory
                if os.path.isabs(directory)
                else os.path.join(cfg.base_data_dir, directory)
                for directory in cfg.rev_dirs
            ]

        rom_type = str(getattr(cfg, "rom_type", "")).strip()
        field_type = str(getattr(cfg, "type_of_field", "")).strip()

        detail(
            logger,
            (
                "Configuration: rom_type=%s, field_type=%s, "
                "field_variable=%s, infer_only=%s, skip_training=%s"
            ),
            rom_type,
            field_type,
            getattr(cfg, "field_variable", None),
            getattr(cfg, "infer_only", False),
            getattr(cfg, "skip_training", False),
        )

        logger.debug(
            (
                "Artifact settings: seed=%d, rebuild_graph=%s, "
                "rebuild_targets=%s, use_cached_artifacts=%s"
            ),
            seed,
            getattr(cfg, "rebuild_graph", False),
            getattr(cfg, "rebuild_targets", False),
            getattr(cfg, "use_cached_artifacts", False),
        )

        if rom_type == "ML" and field_type == "Eulerian":
            logger.info("Starting Eulerian ML ROM pipeline")

            from cpfd_rom.ml_rom.rom_eulerian_ml.pipeline import (
                run_ml_rom_pipeline,
            )

            run_ml_rom_pipeline(cfg, log_time)

        elif rom_type == "ML" and field_type == "Lagrangian":
            logger.info("Starting Lagrangian ML ROM pipeline")

            from cpfd_rom.ml_rom.rom_lagrangian_ml.pipeline import (
                run_lagrangian_ml_pipeline,
            )

            run_lagrangian_ml_pipeline(cfg, log_time)

        else:
            raise ValueError(
                "Unsupported ROM type / field type: "
                f"rom_type={rom_type}, type_of_field={field_type}"
            )

        clear_memory()
        logger.info("ROM execution completed successfully")
        return 0

    except Exception as error:
        logger.error(
            "ROM execution failed: %s",
            error,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        clear_memory()
        return 1