"""Reusable local operator-panel core.

Repository-specific behavior belongs in an adapter implementing
``PanelAdapter``. This package intentionally has no Galaxea, ROS, camera, or
model imports so it can later live in a standalone Git submodule.
"""

from .contracts import InputAction, PanelAdapter, WorkflowLaunch
from .server import serve_operator_panel

__all__ = [
    "InputAction",
    "PanelAdapter",
    "WorkflowLaunch",
    "serve_operator_panel",
]
