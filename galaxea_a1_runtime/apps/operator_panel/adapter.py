"""Compose the reusable panel core with A1-specific capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from operator_panel.config_store import RepositoryConfigStore
from operator_panel.contracts import WorkflowLaunch

from .catalog import build_a1_catalog
from .configuration import build_a1_config_store
from .workflows import build_a1_workflow_launch


PANEL_BIND = "127.0.0.1"
PANEL_PORT = 8765


class A1OperatorPanelAdapter:
    """Keep every A1-specific path, loader, and workflow out of the Web core."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self._config_store: RepositoryConfigStore = build_a1_config_store(
            self.repo_root
        )

    def catalog(self) -> dict[str, Any]:
        return build_a1_catalog(self.repo_root, self._config_store)

    def build_launch(self, workflow: str, values: dict[str, Any]) -> WorkflowLaunch:
        return build_a1_workflow_launch(self.repo_root, workflow, values)

    def config_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._config_store.template(
            _payload_text(payload, "kind"), _payload_text(payload, "source")
        )

    def validate_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._config_store.validate(
            _payload_text(payload, "kind"),
            _payload_text(payload, "filename"),
            _payload_content(payload),
        )

    def create_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self._config_store.create(
            _payload_text(payload, "kind"),
            _payload_text(payload, "filename"),
            _payload_content(payload),
        )
        result["catalog"] = self.catalog()
        return result


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
