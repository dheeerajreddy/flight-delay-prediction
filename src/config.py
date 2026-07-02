"""Load the YAML config once and expose it as a plain dict.

Usage:
    from src.config import load_config
    cfg = load_config()               # default: config/config.yaml
    cfg = load_config("config/other.yaml")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/config.py -> project/)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML config and resolve all relative paths against the root."""
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(relative_path: str) -> Path:
    """Turn a config-relative path into an absolute one anchored at the root."""
    return ROOT / relative_path
