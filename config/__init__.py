"""Configuration loading.

All tunable values live in ``config/config.yaml``. Relative paths in the file
are resolved against the repository root so scripts can be run from anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    config_file = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_file.resolve())
    return cfg


def resolve_path(path: str) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def output_dir(cfg: Dict[str, Any], *parts: str) -> Path:
    """Return (and create) a directory under ``paths.output_dir``."""
    d = resolve_path(cfg["paths"]["output_dir"]).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def backbone_config(cfg: Dict[str, Any], key: Optional[str] = None) -> Dict[str, Any]:
    """Return the model settings for backbone ``key`` (default: the primary one)."""
    bb = cfg["backbones"]
    key = key or bb["primary"]
    if key not in bb["models"]:
        raise KeyError(f"Unknown backbone '{key}'. Configured: {sorted(bb['models'])}")
    return {
        "key": key,
        "name_or_path": bb["models"][key]["name_or_path"],
        "hf_token_env": bb["hf_token_env"],
        "dtype": bb["dtype"],
        "device_map": bb["device_map"],
        "max_input_tokens": bb["max_input_tokens"],
        "generation": bb["generation"],
    }
