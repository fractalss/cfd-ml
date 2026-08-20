# cpfd_rom/rom_cli.py
from __future__ import annotations

import argparse

from cpfd_rom.main_entry import run_from_config_path
from cpfd_rom.util.logging_config import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ROM pipeline from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        "--config_yaml",
        dest="config_path",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help=(
            "Run inference using existing trained artifacts and "
            "skip model training."
        ),
    )

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase terminal output. Use -v for details or "
            "-vv for debug diagnostics."
        ),
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all terminal output except errors.",
    )

    args = parser.parse_args()

    configure_logging(
        verbosity=args.verbose,
        quiet=args.quiet,
    )

    return int(
        run_from_config_path(
            args.config_path,
            infer_only=args.infer_only,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())