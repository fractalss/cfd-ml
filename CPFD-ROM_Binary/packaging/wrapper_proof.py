#!/usr/bin/env python3
"""
wrapper_proof.py

Simulates the C++ licensed wrapper:
- accepts a config YAML path
- accepts overrides as a JSON dict (string or file)
- calls cpfd_rom.api.run(config_path, overrides_dict)
- exits with the same return code that a wrapper would use

Examples:

1) Minimal:
  python wrapper_proof.py --config rom_inputs.yaml

2) JSON string overrides:
  python wrapper_proof.py --config rom_inputs.yaml \
    --overrides-json '{"conv_type":"GAT","gat_heads":4,"attn_dropout":0.1,"add_time":true,"time_mode":"fourier","fourier_m":8}'

3) JSON file overrides:
  python wrapper_proof.py --config rom_inputs.yaml --overrides-file overrides.json

4) Dry-run (if you implement cfg.dry_run in your pipeline / config):
  python wrapper_proof.py --config rom_inputs.yaml --overrides-json '{"dry_run": true}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _load_overrides(overrides_json: Optional[str], overrides_file: Optional[str]) -> Dict[str, Any]:
    if overrides_json and overrides_file:
        raise SystemExit("ERROR: Provide only one of --overrides-json or --overrides-file")

    if overrides_file:
        p = Path(overrides_file)
        if not p.is_file():
            raise SystemExit(f"ERROR: overrides file not found: {p}")
        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
    elif overrides_json:
        data = json.loads(overrides_json)
    else:
        return {}

    if not isinstance(data, dict):
        raise SystemExit("ERROR: overrides must be a JSON object (dict).")

    return data


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Proof harness that mimics the C++ wrapper calling cpfd_rom.api.run().")
    ap.add_argument("--config", "--config_yaml", dest="config_path", required=True, help="Path to rom_inputs.yaml")
    ap.add_argument("--overrides-json", dest="overrides_json", default=None, help="JSON dict string of overrides")
    ap.add_argument("--overrides-file", dest="overrides_file", default=None, help="Path to JSON file of overrides")
    ap.add_argument("--print-env", action="store_true", help="Print python executable and sys.path (debug)")

    args = ap.parse_args(argv)

    if args.print_env:
        print("[ENV] python:", sys.executable)
        print("[ENV] sys.path[0:5]:", sys.path[:5])

    overrides = _load_overrides(args.overrides_json, args.overrides_file)

    from cpfd_rom.api import run
    rc = run(args.config_path, overrides=overrides or None)

    print(f"[WRAPPER_PROOF] rc={rc}")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
