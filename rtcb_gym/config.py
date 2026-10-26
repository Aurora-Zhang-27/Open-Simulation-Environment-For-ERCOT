"""Load the YAML configuration.

A config file can start from another one and change only a few keys:

    extends: default.yaml          # path relative to this file
    market:
      location: HB_HOUSTON         # everything else comes from default.yaml

Nested sections are merged key by key; lists and scalars are replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
REQUIRED_SECTIONS = ("market", "data", "battery", "env", "splits", "baselines")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ValueError(f"config extends itself: {chain}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping")
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    return _merge(_load(path.parent / parent, (*seen, path)), cfg)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    cfg = _load(path, ())
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise ValueError(f"config {path} is missing section {section!r}")
    return cfg


def resolve_path(p: str | Path, root: Path = REPO_ROOT) -> Path:
    """Relative paths in the config are relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else root / p
