"""Compose the reusable panel core with A1-specific capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_ops.operator_panel import (
    PanelCapabilities,
    WorkflowLaunch,
    fetch_camera_health,
)

from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config

from .catalog import build_a1_catalog
from .workflows import build_a1_workflow_launch


class A1OperatorPanelAdapter:
    """Keep every A1-specific path, loader, and workflow out of the Web core."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        system = load_system_config(
            self.repo_root / SYSTEM_CONFIG, repo_root=self.repo_root
        )
        self.panel_bind = system.operator_panel.bind
        self.panel_port = system.operator_panel.port
        self._camera_web_port = system.web_preview.port
        self.capabilities = PanelCapabilities(camera=self)

    def catalog(self) -> dict[str, Any]:
        return build_a1_catalog(self.repo_root)

    def camera_health(self) -> dict[str, Any]:
        return fetch_camera_health(self._camera_web_port)

    def build_launch(self, workflow: str, values: dict[str, Any]) -> WorkflowLaunch:
        return build_a1_workflow_launch(self.repo_root, workflow, values)
