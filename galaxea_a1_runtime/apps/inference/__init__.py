"""Registered inference target discovery and selection."""

from galaxea_a1_runtime.apps.inference.select import add_arguments, main, run
from galaxea_a1_runtime.apps.inference.targets import (
    InferenceTarget,
    discover_targets,
)

__all__ = [
    "InferenceTarget",
    "add_arguments",
    "discover_targets",
    "main",
    "run",
]
