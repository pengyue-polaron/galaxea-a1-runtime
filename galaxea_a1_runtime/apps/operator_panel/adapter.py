"""Compose the reusable panel core with A1-specific capabilities."""

from __future__ import annotations

import json
import math
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

from embodied_ops.operator_panel import (
    PanelCapabilities,
    RepositoryDocumentStore,
    WorkflowLaunch,
)

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.configuration.tasks import (
    load_task_catalog,
    register_task_prompt,
)

from .catalog import build_a1_catalog
from .configuration import build_a1_document_store
from .workflows import build_a1_workflow_launch


CAMERA_HEALTH_TIMEOUT_S = 0.4
CAMERA_HEALTH_MAX_BYTES = 64 * 1024


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
        connection = HTTPConnection(
            "127.0.0.1",
            self._camera_web_port,
            timeout=CAMERA_HEALTH_TIMEOUT_S,
        )
        try:
            connection.request("GET", "/healthz", headers={"Cache-Control": "no-store"})
            response = connection.getresponse()
            body = response.read(CAMERA_HEALTH_MAX_BYTES + 1)
        except (OSError, TimeoutError):
            return _camera_health_unavailable("Camera monitor is not running.")
        finally:
            connection.close()
        if len(body) > CAMERA_HEALTH_MAX_BYTES:
            return _camera_health_unavailable(
                "Camera monitor health response is too large."
            )
        if response.status not in {200, 503}:
            return _camera_health_unavailable("Camera monitor health request failed.")
        try:
            payload = json.loads(body)
            return _normalize_camera_health(payload)
        except (TypeError, ValueError):
            return _camera_health_unavailable(
                "Camera monitor returned invalid health data."
            )

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


def _normalize_camera_health(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
        raise ValueError("camera health must contain a boolean ok value")
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, dict):
        raise ValueError("camera health streams must be an object")
    streams: dict[str, dict[str, Any]] = {}
    for stream_id, raw in raw_streams.items():
        if not isinstance(stream_id, str) or not isinstance(raw, dict):
            raise ValueError("camera health stream is invalid")
        ready = raw.get("ready")
        fresh = raw.get("fresh")
        error = raw.get("error")
        if not isinstance(ready, bool) or not isinstance(fresh, bool):
            raise ValueError("camera health readiness values must be booleans")
        if error is not None and not isinstance(error, str):
            raise ValueError("camera health error must be text or null")
        streams[stream_id] = {
            "ready": ready,
            "fresh": fresh,
            "preview_fps": _optional_nonnegative_number(
                raw.get("preview_fps"), label="preview_fps"
            ),
            "age_s": _optional_nonnegative_number(raw.get("age_s"), label="age_s"),
            "error": error,
        }
    return {"available": True, "ok": payload["ok"], "streams": streams}


def _optional_nonnegative_number(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"camera health {label} must be numeric or null")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"camera health {label} must be finite and non-negative")
    return number


def _camera_health_unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "ok": False, "streams": {}, "reason": reason}
