import os
import gc
import time
from contextlib import contextmanager
import argparse

from cpfd_rom.util.config import load_config, overlay_cli


@contextmanager
def log_time(task_name):
    start = time.time()
    yield
    end = time.time()
    print(f"[Timing] {task_name} took {end - start:.2f} seconds")


def clear_memory():
    gc.collect()


def _truthy(x):
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run ROM pipelines with configuration from a YAML file.\n\n"
            "Example:\n  rom-cli-bin --config rom_inputs.yaml --add_time true --time_mode fourier --fourier_m 8\n\n"
            "YAML (revised schema) top-level keys include: rom_type, type_of_field, field_variable,\n"
            "base_data_dir, rev_dirs, param_mapping, user_parameter, conv_type, gat_heads, attn_dropout,\n"
            "time_mode, fourier_m, [add_time], skip_training, rebuild_graph."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Accept both --config and --config_yaml for backward compatibility
    parser.add_argument(
        "--config", "--config_yaml",
        dest="config_path",
        required=True,
        help="Path to YAML file with all configuration options.",
    )

    # Optional CLI overrides
    parser.add_argument("--add_time", "--add-time", dest="add_time", default=None)
    parser.add_argument("--time_mode", "--time-mode", dest="time_mode", default=None)
    parser.add_argument("--fourier_m", "--fourier-m", dest="fourier_m", type=int, default=None)
    parser.add_argument("--conv_type", "--conv-type", dest="conv_type", default=None)
    parser.add_argument("--gat_heads", "--gat-heads", dest="gat_heads", type=int, default=None)
    parser.add_argument("--attn_dropout", "--attn-dropout", dest="attn_dropout", type=float, default=None)

    args, _ = parser.parse_known_args()

    clear_memory()

    # Load YAML -> ROMConfig (your dataclass/namespace)
    cfg = load_config(args.config_path)

    # Overlay CLI overrides (only those provided)
    overrides = {}
    if args.add_time is not None:
        overrides["add_time"] = _truthy(args.add_time)
    if args.time_mode is not None:
        overrides["time_mode"] = args.time_mode
    if args.fourier_m is not None:
        overrides["fourier_m"] = args.fourier_m
    if args.conv_type is not None:
        overrides["conv_type"] = args.conv_type
    if args.gat_heads is not None:
        overrides["gat_heads"] = args.gat_heads
    if args.attn_dropout is not None:
        overrides["attn_dropout"] = args.attn_dropout

    if overrides:
        cfg = overlay_cli(cfg, **overrides)

    # Prepend base_data_dir to rev_dirs if relative
    if getattr(cfg, "base_data_dir", None) and getattr(cfg, "rev_dirs", None):
        cfg.rev_dirs = [
            d if os.path.isabs(d) else os.path.join(cfg.base_data_dir, d)
            for d in cfg.rev_dirs
        ]

    print(
        "[MAIN] Effective:",
        "add_time=", getattr(cfg, "add_time", None),
        "time_mode=", getattr(cfg, "time_mode", None),
        "fourier_m=", getattr(cfg, "fourier_m", None),
        "conv_type=", getattr(cfg, "conv_type", None),
    )

    # -------------------------
    # LAZY IMPORT + ROUTING
    # -------------------------
    rom_type = str(getattr(cfg, "rom_type", "")).strip()
    field_type = str(getattr(cfg, "type_of_field", "")).strip()

    if rom_type == "ML" and field_type == "Eulerian":
        from cpfd_rom.ml_rom.rom_eulerian_ml.pipeline import run_ml_rom_pipeline
        run_ml_rom_pipeline(cfg, log_time)

    elif rom_type == "ML" and field_type == "Lagrangian":
        from cpfd_rom.ml_rom.rom_lagrangian_ml.pipeline import run_lagrangian_ml_pipeline
        run_lagrangian_ml_pipeline(cfg, log_time)

    else:
        raise ValueError(f"Unsupported ROM type / field type: rom_type={rom_type}, type_of_field={field_type}")

    clear_memory()


if __name__ == "__main__":
    main()
