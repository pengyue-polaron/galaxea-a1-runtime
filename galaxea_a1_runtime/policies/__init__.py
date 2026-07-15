"""Policy-facing action adapters."""

from __future__ import annotations

from .actions import RuntimeAction, normalize_action

__all__ = [
    "RuntimeAction",
    "normalize_action",
]
