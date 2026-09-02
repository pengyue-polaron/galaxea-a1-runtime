"""Pure telemetry and scoped-operator mappings for the ROS observability adapter."""

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
BASE_FOXGLOVE_CAPABILITIES = ("connectionGraph", "assets")
OPS_TELEMETRY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DiagnosticFinding:
    name: str
    level: int
    message: str
    values: tuple[tuple[str, str], ...] = ()
    hardware_id: str = "galaxea-a1"


@dataclass(frozen=True)
class OperatorActionRequest:
    action_id: str
    run_id: str
    input_revision: int


@dataclass(frozen=True)
class CollectionServiceBinding:
    service_name: str
    action_id: str
    expected_phase: str


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


def foxglove_service_whitelist(system: SystemConfig) -> tuple[str, ...]:
    """Return the exact service patterns granted to the Foxglove connection."""

    if not system.operator_panel.control_enabled:
        return NO_MATCH_ALLOWLIST
    return tuple(
        f"^{re.escape(service)}$"
        for service in sorted(system.operator_panel.services.__dict__.values())
    )


def foxglove_capabilities(system: SystemConfig) -> tuple[str, ...]:
    """Return bridge capabilities without granting client topic publication."""

    if system.operator_panel.control_enabled:
        return (*BASE_FOXGLOVE_CAPABILITIES, "services")
    return BASE_FOXGLOVE_CAPABILITIES


def collection_action_service_bindings(
    system: SystemConfig,
) -> tuple[CollectionServiceBinding, ...]:
    """Map each configured episode service to its semantic one-shot input gate."""

    services = system.operator_panel.services
    return (
        CollectionServiceBinding(services.start, "start", "ready"),
        CollectionServiceBinding(services.save, "save", "recording"),
        CollectionServiceBinding(services.discard, "discard", "recording"),
        CollectionServiceBinding(services.reset, "reset", "ready"),
    )


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
    input_revision = snapshot.get("input_revision")
    active = snapshot.get("active")
    exit_code = snapshot.get("exit_code")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Operator Panel status revision must be non-negative")
    if (
        isinstance(input_revision, bool)
        or not isinstance(input_revision, int)
        or input_revision < 0
    ):
        raise ValueError("Operator Panel input revision must be non-negative")
    if not isinstance(active, bool):
        raise ValueError("Operator Panel status active must be boolean")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise ValueError("Operator Panel status exit_code must be integer or null")
    text_fields = (
        "run_id",
        "state",
        "workflow",
        "name",
        "started_at",
        "finished_at",
        "input_phase",
        "input_detail",
    )
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
    progress_ids: list[str] = []
    for item in progress:
        if set(item) != {"id", "label", "current", "total", "phase", "detail"}:
            raise ValueError("Operator Panel progress entry is invalid")
        progress_id = item["id"]
        if (
            not isinstance(progress_id, str)
            or not progress_id
            or any(
                not isinstance(item[field], str)
                for field in ("label", "phase", "detail")
            )
        ):
            raise ValueError("Operator Panel progress identity is invalid")
        current = item["current"]
        total = item["total"]
        if (
            isinstance(current, bool)
            or not isinstance(current, (int, float))
            or not isfinite(current)
            or current < 0
            or total is not None
            and (
                isinstance(total, bool)
                or not isinstance(total, (int, float))
                or not isfinite(total)
                or total <= 0
                or current > total
            )
        ):
            raise ValueError("Operator Panel progress values are invalid")
        progress_ids.append(progress_id)
    if len(set(progress_ids)) != len(progress_ids):
        raise ValueError("Operator Panel progress ids must not contain duplicates")
    if not isinstance(input_actions, list) or not all(
        isinstance(item, dict) for item in input_actions
    ):
        raise ValueError("Operator Panel input_actions must be a list of objects")
    for action in input_actions:
        if set(action) != {"id", "label", "tone"} or not all(
            isinstance(action[key], str) for key in ("id", "label", "tone")
        ):
            raise ValueError("Operator Panel input action is invalid")
        if (
            not action["id"]
            or not action["label"]
            or action["tone"] not in {"default", "primary", "danger", "quiet"}
        ):
            raise ValueError("Operator Panel input action values are invalid")
    action_ids = [action["id"] for action in input_actions]
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("Operator Panel input action ids must not contain duplicates")
    return {
        "schema_version": OPS_TELEMETRY_SCHEMA_VERSION,
        "source_schema_version": WORKFLOW_STATUS_SCHEMA_VERSION,
        "available": True,
        "revision": revision,
        "input_revision": input_revision,
        "input_phase": snapshot["input_phase"],
        "input_detail": snapshot["input_detail"],
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


def prepare_collection_action(
    snapshot: object,
    *,
    action_id: str,
    expected_phase: str,
) -> OperatorActionRequest:
    """Validate a Foxglove action against one exact active collection input gate."""

    telemetry = operator_panel_telemetry(snapshot)
    if not telemetry["active"] or telemetry["workflow"] != "collect":
        raise ValueError("no active collection session")
    if telemetry["state"] != "waiting_for_input":
        raise ValueError(f"collection is not accepting input ({telemetry['state']})")
    if telemetry["input_phase"] != expected_phase:
        raise ValueError(
            f"collection phase is {telemetry['input_phase']!r}, expected {expected_phase!r}"
        )
    action_ids = {action["id"] for action in telemetry["input_actions"]}
    if action_id not in action_ids:
        raise ValueError(f"action {action_id!r} is not currently available")
    run_id = telemetry["run_id"]
    input_revision = telemetry["input_revision"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("collection run id is unavailable")
    if not isinstance(input_revision, int):
        raise ValueError("collection input revision is unavailable")
    return OperatorActionRequest(
        action_id=action_id,
        run_id=run_id,
        input_revision=input_revision,
    )


def prepare_collection_stop(snapshot: object) -> str:
    """Validate the explicitly scoped stop service against the active collect run."""

    telemetry = operator_panel_telemetry(snapshot)
    if not telemetry["active"] or telemetry["workflow"] != "collect":
        raise ValueError("no active collection session")
    if telemetry["state"] == "stopping":
        raise ValueError("collection session is already stopping")
    run_id = telemetry["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("collection run id is unavailable")
    return run_id


def operator_panel_diagnostic(
    telemetry: dict[str, object],
) -> DiagnosticFinding:
    """Summarize the versioned Operator Panel telemetry."""

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
