"""Load the YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section in ("market", "data", "battery", "env", "splits", "baselines"):
        if section not in cfg:
            raise ValueError(f"config {path} is missing section {section!r}")
    return cfg


def resolve_path(p: str | Path, root: Path = REPO_ROOT) -> Path:
    """Relative paths in the config are relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else root / p
