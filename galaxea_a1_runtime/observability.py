"""Pure read-only telemetry mappings shared by the ROS observability adapter."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from embodied_ops.operator_panel import WORKFLOW_STATUS_SCHEMA_VERSION

from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.constants import IDLE_TIMEOUT_CODE


DIAGNOSTIC_OK = 0
DIAGNOSTIC_WARN = 1
DIAGNOSTIC_ERROR = 2
NO_MATCH_ALLOWLIST = ("^$",)
READ_ONLY_FOXGLOVE_CAPABILITIES = ("connectionGraph", "assets")
OPS_TELEMETRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DiagnosticFinding:
    name: str
    level: int
    message: str
    values: tuple[tuple[str, str], ...] = ()
    hardware_id: str = "galaxea-a1"


def foxglove_topic_whitelist(system: SystemConfig) -> tuple[str, ...]:
    """Return exact subscription-only topic regexes derived from tracked config."""

    topics = {
        *system.topics.__dict__.values(),
        *system.observability.topics.__dict__.values(),
        "/rosout",
        "/tf",
        "/tf_static",
    }
    return tuple(f"^{re.escape(topic)}$" for topic in sorted(topics))


def foxglove_asset_uri_allowlist(system: SystemConfig) -> tuple[str, ...]:
    """Allow only the configured URDF and the assets it references."""

    return tuple(f"^{re.escape(uri)}$" for uri in foxglove_asset_uris(system))


def foxglove_asset_uris(system: SystemConfig) -> tuple[str, ...]:
    """Return the exact configured URDF and mesh package URIs."""

    urdf_path = system.eef_ik.urdf.resolve()
    uris = {_package_uri(urdf_path)}
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"configured URDF is not valid XML: {urdf_path}") from exc
    for element in root.iter():
        filename = element.attrib.get("filename")
        if filename is None:
            continue
        if not filename.startswith("package://") or ".." in filename:
            raise ValueError(f"URDF asset must be a safe package URI: {filename!r}")
        uris.add(filename)
    return tuple(sorted(uris))


def relay_diagnostic(
    payload: str,
    *,
    age_s: float,
    max_age_s: float,
) -> DiagnosticFinding:
    """Convert the relay's existing JSON status into a typed diagnostic finding."""

    if not isfinite(age_s) or age_s < 0 or not isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("relay diagnostic ages must be finite and valid")
    if age_s > max_age_s:
        return DiagnosticFinding(
            name="A1/Relay",
            level=DIAGNOSTIC_ERROR,
            message=f"relay status stale ({age_s:.3f}s)",
            values=(("age_s", f"{age_s:.3f}"),),
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict):
        return DiagnosticFinding(
            name="A1/Relay",
            level=DIAGNOSTIC_ERROR,
            message="relay status is not a JSON object",
        )
    state = value.get("state")
    reason = value.get("reason", "")
    if not isinstance(state, str) or not isinstance(reason, str):
        return DiagnosticFinding(
            name="A1/Relay",
            level=DIAGNOSTIC_ERROR,
            message="relay status state/reason is invalid",
        )
    levels = {
        "ACTIVE": DIAGNOSTIC_OK,
        "LOCKED": DIAGNOSTIC_OK,
        "ARMING": DIAGNOSTIC_WARN,
        "FAULT": DIAGNOSTIC_ERROR,
    }
    level = levels.get(state, DIAGNOSTIC_ERROR)
    message = state if not reason else f"{state}: {reason}"
    values = tuple(
        (str(key), _diagnostic_value(item))
        for key, item in sorted(value.items())
        if key not in {"state", "reason"}
    )
    return DiagnosticFinding(
        name="A1/Relay",
        level=level,
        message=message,
        values=(("state", state), ("reason", reason), *values),
    )


def motor_diagnostic(
    error_codes: tuple[int, ...],
    *,
    arm_joints: int,
    gripper_ignored_error_mask: int,
) -> DiagnosticFinding:
    """Summarize every arm and gripper motor status without hiding raw codes."""

    required = arm_joints + 1
    values = tuple(
        (f"motor_{index + 1}_error", str(code))
        for index, code in enumerate(error_codes)
    )
    if len(error_codes) < required:
        return DiagnosticFinding(
            name="A1/Motors",
            level=DIAGNOSTIC_ERROR,
            message=f"motor status has {len(error_codes)} entries, need {required}",
            values=values,
        )
    bad_arm = [
        (index + 1, code)
        for index, code in enumerate(error_codes[:arm_joints])
        if code not in (0, IDLE_TIMEOUT_CODE)
    ]
    gripper_code = error_codes[arm_joints]
    gripper_remaining = gripper_code & ~(IDLE_TIMEOUT_CODE | gripper_ignored_error_mask)
    messages = [f"J{index}={code}" for index, code in bad_arm]
    if gripper_remaining:
        messages.append(f"gripper={gripper_code}")
    return DiagnosticFinding(
        name="A1/Motors",
        level=DIAGNOSTIC_ERROR if messages else DIAGNOSTIC_OK,
        message="motor errors: " + ", ".join(messages)
        if messages
        else "motor status healthy",
        values=values,
    )


def camera_diagnostic(
    *,
    connected: bool,
    front_age_s: float | None,
    wrist_age_s: float | None,
    pair_skew_s: float | None,
    max_age_s: float,
    max_pair_skew_s: float,
    error: str = "",
) -> DiagnosticFinding:
    """Summarize the read-only Camera Bridge consumer state."""

    values = tuple(
        (key, "unavailable" if value is None else f"{value:.3f}")
        for key, value in (
            ("front_age_s", front_age_s),
            ("wrist_age_s", wrist_age_s),
            ("pair_skew_s", pair_skew_s),
        )
    )
    if not connected:
        return DiagnosticFinding(
            name="A1/Cameras",
            level=DIAGNOSTIC_WARN,
            message=error or "Camera Bridge unavailable",
            values=values,
        )
    stale = (
        front_age_s is None
        or wrist_age_s is None
        or front_age_s > max_age_s
        or wrist_age_s > max_age_s
    )
    skewed = pair_skew_s is None or pair_skew_s > max_pair_skew_s
    if stale or skewed:
        return DiagnosticFinding(
            name="A1/Cameras",
            level=DIAGNOSTIC_WARN,
            message="camera pair stale or outside configured skew",
            values=values,
        )
    return DiagnosticFinding(
        name="A1/Cameras",
        level=DIAGNOSTIC_OK,
        message="camera pair fresh",
        values=values,
    )


def operator_panel_telemetry(
    snapshot: object | None,
    *,
    error: str = "",
) -> dict[str, object]:
    """Normalize the generic panel status into the stable A1 telemetry envelope."""

    if snapshot is None:
        return {
            "schema_version": OPS_TELEMETRY_SCHEMA_VERSION,
            "available": False,
            "error": error or "Operator Panel unavailable",
        }
    if not isinstance(snapshot, dict):
        raise ValueError("Operator Panel status must be a JSON object")
    if snapshot.get("schema_version") != WORKFLOW_STATUS_SCHEMA_VERSION:
        raise ValueError("Operator Panel status schema version mismatch")
    revision = snapshot.get("revision")
    active = snapshot.get("active")
    exit_code = snapshot.get("exit_code")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Operator Panel status revision must be non-negative")
    if not isinstance(active, bool):
        raise ValueError("Operator Panel status active must be boolean")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise ValueError("Operator Panel status exit_code must be integer or null")
    text_fields = ("run_id", "state", "workflow", "name", "started_at", "finished_at")
    if any(not isinstance(snapshot.get(field), str) for field in text_fields):
        raise ValueError("Operator Panel status identity fields must be strings")
    state = snapshot["state"]
    if state not in {
        "idle",
        "running",
        "waiting_for_input",
        "stopping",
        "stopped",
        "succeeded",
        "failed",
    }:
        raise ValueError(f"Operator Panel status state is unsupported: {state!r}")
    status_line = snapshot.get("status_line")
    progress = snapshot.get("progress")
    input_actions = snapshot.get("input_actions")
    if not isinstance(status_line, str):
        raise ValueError("Operator Panel status_line must be a string")
    if not isinstance(progress, list) or not all(
        isinstance(item, dict) for item in progress
    ):
        raise ValueError("Operator Panel progress must be a list of objects")
    if not isinstance(input_actions, list) or not all(
        isinstance(item, dict) for item in input_actions
    ):
        raise ValueError("Operator Panel input_actions must be a list of objects")
    return {
        "schema_version": OPS_TELEMETRY_SCHEMA_VERSION,
        "source_schema_version": WORKFLOW_STATUS_SCHEMA_VERSION,
        "available": True,
        "revision": revision,
        "run_id": snapshot["run_id"],
        "state": state,
        "active": active,
        "workflow": snapshot["workflow"],
        "name": snapshot["name"],
        "started_at": snapshot["started_at"],
        "finished_at": snapshot["finished_at"],
        "exit_code": exit_code,
        "progress": progress,
        "status_line": status_line,
        "input_actions": input_actions,
    }


def operator_panel_diagnostic(
    telemetry: dict[str, object],
) -> DiagnosticFinding:
    """Summarize the versioned Operator Panel telemetry without granting control."""

    if telemetry.get("schema_version") != OPS_TELEMETRY_SCHEMA_VERSION:
        raise ValueError("operator telemetry schema version mismatch")
    if telemetry.get("available") is False:
        error = telemetry.get("error")
        if not isinstance(error, str):
            raise ValueError("unavailable operator telemetry requires an error")
        return DiagnosticFinding(
            name="A1/Operator Panel",
            level=DIAGNOSTIC_WARN,
            message=error,
        )
    state = telemetry.get("state")
    if not isinstance(state, str):
        raise ValueError("operator telemetry state must be a string")
    level = DIAGNOSTIC_ERROR if state == "failed" else DIAGNOSTIC_OK
    if state == "stopping":
        level = DIAGNOSTIC_WARN
    values = tuple(
        (key, _diagnostic_value(telemetry.get(key)))
        for key in ("run_id", "workflow", "revision", "exit_code")
    )
    return DiagnosticFinding(
        name="A1/Operator Panel",
        level=level,
        message=state,
        values=values,
    )


def _package_uri(path: Path) -> str:
    parts = path.parts
    share_indexes = [index for index, part in enumerate(parts) if part == "share"]
    if not share_indexes:
        raise ValueError(f"configured URDF is not below a ROS share directory: {path}")
    index = share_indexes[-1]
    if len(parts) <= index + 2:
        raise ValueError(f"configured URDF has no ROS package-relative path: {path}")
    package = parts[index + 1]
    relative = "/".join(parts[index + 2 :])
    return f"package://{package}/{relative}"


def _diagnostic_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
