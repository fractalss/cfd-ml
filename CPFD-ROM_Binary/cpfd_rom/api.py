# cpfd_rom/api.py
from __future__ import annotations

import json
from typing import Any, Mapping, Optional


def run(config_path: str, overrides: Optional[Mapping[str, Any]] = None) -> int:
    """
    Stable programmatic entrypoint for wrappers (C++ license manager, etc).

    Protected boundary:
      cpfd_rom.api.run(config_path, overrides) -> int
    """
    from cpfd_rom.main_entry import run_from_config_path
    return int(run_from_config_path(config_path, overrides=overrides))


def run_from_json(config_path: str, overrides_json: str | None) -> int:
    overrides = None
    if overrides_json:
        overrides = json.loads(overrides_json)
        if not isinstance(overrides, dict):
            raise TypeError("overrides_json must decode to a JSON object (dict).")
    return run(config_path, overrides=overrides)
