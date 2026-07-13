"""ACT joint-state deployment helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["ActConfig", "default_config_path", "load_act_config"]

if TYPE_CHECKING:
    from .config import ActConfig


def __getattr__(name: str):
    if name in __all__:
        from . import config

        return getattr(config, name)
    raise AttributeError(name)
