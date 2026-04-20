"""Load and merge settings.yaml and sar_settings.yaml with env-var overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def load_settings(config_path: str | None = None) -> dict[str, Any]:
    """Return the merged settings dictionary (env-var overrides applied)."""
    cfg_path = Path(config_path) if config_path else _REPO_ROOT / "config" / "settings.yaml"
    cfg = _load_yaml(cfg_path)

    # Apply environment-variable overrides
    env_map = {
        "LOG_LEVEL":          ("logging", "level"),
        "CACHE_MAX_AGE_DAYS": ("cache", "max_age_days"),
        "REGION":             ("region", "name"),
    }
    for env_key, (sec, key) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            cfg.setdefault(sec, {})[key] = val

    return cfg


@lru_cache(maxsize=1)
def load_sar_settings(config_path: str | None = None) -> dict[str, Any]:
    """Return the SAR settings dictionary with env-var SAR_ENABLED override."""
    cfg_path = (
        Path(config_path) if config_path else _REPO_ROOT / "config" / "sar_settings.yaml"
    )
    cfg = _load_yaml(cfg_path)

    env_enabled = os.environ.get("SAR_ENABLED", "").lower()
    if env_enabled in ("true", "1", "yes"):
        cfg.setdefault("sar", {})["enabled"] = True
    elif env_enabled in ("false", "0", "no"):
        cfg.setdefault("sar", {})["enabled"] = False

    return cfg


def is_sar_enabled() -> bool:
    """Convenience helper — check SAR_ENABLED flag."""
    return bool(load_sar_settings().get("sar", {}).get("enabled", False))
