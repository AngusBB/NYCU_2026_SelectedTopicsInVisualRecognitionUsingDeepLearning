"""Configuration helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """Save a YAML config file."""
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def deep_copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy so CLI overrides do not mutate the original."""
    return copy.deepcopy(config)


def set_nested(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a nested config value from a dotted key path."""
    keys = key_path.split(".")
    cursor = config
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value

