"""Compose the reusable panel core with A1-specific capabilities."""

from __future__ import annotations

import json
import math
from http.client import HTTPConnection
from pathlib import Path
from typing import Any

from operator_panel.config_store import RepositoryConfigStore
from operator_panel.contracts import WorkflowLaunch

from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config

from .catalog import build_a1_catalog
from .configuration import build_a1_config_store
from .workflows import build_a1_workflow_launch


PANEL_BIND = "127.0.0.1"
PANEL_PORT = 8765
CAMERA_HEALTH_TIMEOUT_S = 0.4
CAMERA_HEALTH_MAX_BYTES = 64 * 1024


class A1OperatorPanelAdapter:
    """Keep every A1-specific path, loader, and workflow out of the Web core."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        system = load_system_config(
            self.repo_root / SYSTEM_CONFIG, repo_root=self.repo_root
        )
        self._camera_web_port = system.web_preview.port
        self._config_store: RepositoryConfigStore = build_a1_config_store(
            self.repo_root
        )

    def catalog(self) -> dict[str, Any]:
        return build_a1_catalog(self.repo_root, self._config_store)

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
