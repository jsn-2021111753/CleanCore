"""Configuration loading and merging helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from common.paths import CONFIG_ROOT


def load_yaml_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {} if data is None else dict(data)


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_default_config(config_root: Path = CONFIG_ROOT) -> Dict[str, Any]:
    return load_yaml_config(Path(config_root) / "default.yaml")
