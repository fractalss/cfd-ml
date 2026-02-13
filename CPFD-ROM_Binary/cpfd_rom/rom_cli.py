# cpfd_rom/rom_cli.py
from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def main() -> int:
    # Keep your existing UX; just add one optional JSON override flag.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", "--config_yaml", dest="config_path", required=True)
    parser.add_argument("--overrides_json", "--overrides-json", dest="overrides_json", default=None)

    # Pass-through existing flags (dont validate here; main_entry handles them today)
    parser.add_argument("--add_time", "--add-time", dest="add_time", default=None)
    parser.add_argument("--time_mode", "--time-mode", dest="time_mode", default=None)
    parser.add_argument("--fourier_m", "--fourier-m", dest="fourier_m", type=int, default=None)
    parser.add_argument("--conv_type", "--conv-type", dest="conv_type", default=None)
    parser.add_argument("--gat_heads", "--gat-heads", dest="gat_heads", type=int, default=None)
    parser.add_argument("--attn_dropout", "--attn-dropout", dest="attn_dropout", type=float, default=None)

    args, _ = parser.parse_known_args()

    # Build overrides dict from either JSON, or legacy CLI overrides, or both.
    overrides: Dict[str, Any] = {}

    if args.overrides_json:
        parsed = json.loads(args.overrides_json)
        if not isinstance(parsed, dict):
            raise SystemExit("ERROR: --overrides_json must be a JSON object (dict).")
        overrides.update(parsed)

    # Keep backward compat: CLI flags still override too (if provided)
    if args.add_time is not None:
        overrides["add_time"] = args.add_time
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

    from cpfd_rom.api import run
    return run(args.config_path, overrides=overrides or None)


if __name__ == "__main__":
    raise SystemExit(main())
