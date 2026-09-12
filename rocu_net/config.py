from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .compat import normalize_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, optionally inheriting from a relative ``base_config``."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    config = normalize_config(config)
    base_reference = config.pop("base_config", None)
    if base_reference is not None:
        base_path = Path(str(base_reference)).expanduser()
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base = load_config(base_path)
        base.pop("_config_path", None)
        config = _deep_merge(base, config)
    config["_config_path"] = str(config_path)
    return config


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in normalize_config(config).items() if not key.startswith("_")}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def apply_common_overrides(
    config: dict[str, Any],
    *,
    data_root: str | None = None,
    name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    config = deepcopy(config)
    if data_root is not None:
        config["data"]["root"] = data_root
    if name is not None:
        config["experiment"]["name"] = name
    if seed is not None:
        config["experiment"]["seed"] = int(seed)
    return normalize_config(config)
