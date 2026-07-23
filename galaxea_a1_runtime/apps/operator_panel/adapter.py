"""Compose the reusable panel core with A1-specific capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodied_ops import load_task_catalog, register_task_prompt
from embodied_ops.operator_panel import (
    PanelCapabilities,
    RepositoryDocumentStore,
    WorkflowLaunch,
    fetch_camera_health,
)

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config

from .catalog import build_a1_catalog
from .configuration import build_a1_document_store
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
        self._document_store: RepositoryDocumentStore = build_a1_document_store(
            self.repo_root
        )
        self.capabilities = PanelCapabilities(
            camera=self,
            configuration=self,
            registration=self,
        )

    def catalog(self) -> dict[str, Any]:
        return build_a1_catalog(self.repo_root, self._document_store)

    def camera_health(self) -> dict[str, Any]:
        return fetch_camera_health(self._camera_web_port)

    def build_launch(self, workflow: str, values: dict[str, Any]) -> WorkflowLaunch:
        return build_a1_workflow_launch(self.repo_root, workflow, values)

    def config_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._document_store.template(
            _payload_text(payload, "kind"), _payload_text(payload, "source")
        )

    def validate_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._document_store.validate(
            _payload_text(payload, "kind"),
            _payload_text(payload, "filename"),
            _payload_content(payload),
        )

    def create_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._document_store.create(
            _payload_text(payload, "kind"),
            _payload_text(payload, "filename"),
            _payload_content(payload),
        )

    def register(self, registration: str, values: dict[str, Any]) -> dict[str, Any]:
        if registration != "prompt":
            raise ValueError(f"unknown registration: {registration!r}")
        required = {"catalog", "task_id", "prompt", "distribution"}
        if set(values) != required:
            raise ValueError(
                "prompt registration requires exactly catalog, task_id, prompt, "
                "and distribution"
            )
        catalog_path = self._task_catalog_path(_payload_text(values, "catalog"))
        target = register_task_prompt(
            catalog_path,
            task_id=_payload_text(values, "task_id"),
            prompt=_payload_text(values, "prompt"),
            distribution=_payload_text(values, "distribution"),
            repo_root=self.repo_root,
        )
        deployment_reference: str | None = None
        for path in sorted(
            (self.repo_root / "configs/deployments/lingbot").glob("*.toml")
        ):
            deployment = load_lingbot_config(path, repo_root=self.repo_root)
            if deployment.task_catalog.path == catalog_path:
                deployment_reference = path.relative_to(self.repo_root).as_posix()
                break
        activation_values = {}
        if deployment_reference is not None:
            activation_values["config"] = deployment_reference
        activation_values["task"] = _payload_text(values, "task_id")
        return {
            "created": target.relative_to(self.repo_root).as_posix(),
            "activate": {"panel": "evaluate", "values": activation_values},
        }

    def _task_catalog_path(self, value: str) -> Path:
        candidate = (self.repo_root / value).resolve()
        allowed = (self.repo_root / "configs/tasks").resolve()
        if (
            not candidate.is_relative_to(allowed)
            or candidate.name != "catalog.json"
            or not candidate.is_file()
        ):
            raise ValueError(
                "prompt catalog must be a repository catalog.json under configs/tasks"
            )
        load_task_catalog(candidate, repo_root=self.repo_root)
        return candidate


def _payload_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{key} must not have surrounding whitespace")
    return value


def _payload_content(payload: dict[str, Any]) -> str:
    value = payload.get("content")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("content must be non-empty text")
    return value
