# cpfd_rom/rom_cli.py
from __future__ import annotations

import argparse

from cpfd_rom.main_entry import run_from_config_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ROM pipeline from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        "--config_yaml",
        dest="config_path",
        required=True,
        help="Path to YAML file with all configuration options.",
    )

    args = parser.parse_args()
    return int(run_from_config_path(args.config_path))


if __name__ == "__main__":
    raise SystemExit(main())