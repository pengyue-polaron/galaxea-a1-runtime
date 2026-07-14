"""Shared configuration primitives for system and app deployment contracts."""

from .system import SystemConfig, load_system_config

__all__ = ["SystemConfig", "load_system_config"]
