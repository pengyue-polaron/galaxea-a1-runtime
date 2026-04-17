"""Shared utility helpers used across the a1 package."""

from __future__ import annotations

from typing import Any


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a config object regardless of its concrete type.

    Supports:
    - ``None``            → returns *default*
    - ``dict``            → standard ``dict.get``
    - Hydra / OmegaConf  → ``cfg.get`` (duck-typed)
    - dataclass / object  → ``getattr`` fallback
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
