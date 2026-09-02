import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
from embodied_ops.operator_panel import WORKFLOW_STATUS_SCHEMA_VERSION

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.foxglove_layout import (
    COLLECTION_CONSOLE_PANEL_TYPE,
    build_foxglove_layout,
    render_foxglove_extension_config,
    render_foxglove_layout,
)
from galaxea_a1_runtime.observability import (
    BASE_FOXGLOVE_CAPABILITIES,
    DIAGNOSTIC_ERROR,
    DIAGNOSTIC_OK,
    DIAGNOSTIC_WARN,
    NO_MATCH_ALLOWLIST,
    camera_diagnostic,
    collection_action_service_bindings,
    foxglove_asset_uri_allowlist,
    foxglove_capabilities,
    foxglove_service_whitelist,
    foxglove_topic_whitelist,
    motor_diagnostic,
    operator_panel_diagnostic,
    operator_panel_telemetry,
    prepare_collection_action,
    prepare_collection_stop,
    relay_diagnostic,
)


REPO = Path(__file__).resolve().parents[1]
SYSTEM = REPO / "configs/system/a1.toml"
LAYOUT = REPO / "foxglove/layouts/a1_observability.json"
EXTENSION_CONFIG = (
    REPO / "foxglove/extensions/galaxea-a1-collection-console/src/a1Config.ts"
)
FOXGLOVE_LAUNCH = REPO / "scripts/runtime/foxglove_bridge_scoped.launch"


def test_foxglove_contract_grants_only_exact_topics_assets_and_operator_services() -> (
    None
):
    system = load_system_config(SYSTEM, repo_root=REPO)
    topic_patterns = foxglove_topic_whitelist(system)
    asset_patterns = foxglove_asset_uri_allowlist(system)

    service_patterns = foxglove_service_whitelist(system)

    assert BASE_FOXGLOVE_CAPABILITIES == ("connectionGraph", "assets")
    assert foxglove_capabilities(system) == ("connectionGraph", "assets", "services")
    assert NO_MATCH_ALLOWLIST == ("^$",)
    assert all(
        pattern.startswith("^") and pattern.endswith("$") for pattern in topic_patterns
    )
    assert any(
        re.fullmatch(pattern, system.topics.joint_states) for pattern in topic_patterns
    )
    assert any(
        re.fullmatch(pattern, system.observability.topics.front_image)
        for pattern in topic_patterns
    )
    assert not any(
        re.fullmatch(pattern, f"{system.topics.joint_target}/unexpected")
        for pattern in topic_patterns
    )
    assert any("A1_URDF_0607_0028" in pattern for pattern in asset_patterns)
    assert all(
        ".*" not in pattern and ".." not in pattern for pattern in asset_patterns
    )
    assert len(service_patterns) == 5
    assert all(
        any(re.fullmatch(pattern, service) for pattern in service_patterns)
        for service in system.operator_panel.services.__dict__.values()
    )
    assert not any(
        re.fullmatch(pattern, "/rosout/get_loggers") for pattern in service_patterns
    )
    assert [
        (binding.service_name, binding.action_id, binding.expected_phase)
        for binding in collection_action_service_bindings(system)
    ] == [
        (system.operator_panel.services.start, "start", "ready"),
        (system.operator_panel.services.save, "save", "recording"),
        (system.operator_panel.services.discard, "discard", "recording"),
        (system.operator_panel.services.reset, "reset", "ready"),
    ]

    disabled = replace(
        system,
        operator_panel=replace(system.operator_panel, control_enabled=False),
    )
    assert foxglove_service_whitelist(disabled) == NO_MATCH_ALLOWLIST
    assert foxglove_capabilities(disabled) == BASE_FOXGLOVE_CAPABILITIES


def test_tracked_foxglove_workspace_is_current_and_contains_scoped_console() -> None:
    system = load_system_config(SYSTEM, repo_root=REPO)
    layout = build_foxglove_layout(system)

    assert LAYOUT.read_text(encoding="utf-8") == render_foxglove_layout(system)
    assert EXTENSION_CONFIG.read_text(encoding="utf-8") == (
        render_foxglove_extension_config(system)
    )
    assert all(not panel_id.startswith("Publish!") for panel_id in layout["configById"])
    assert any(
        panel_id.startswith(f"{COLLECTION_CONSOLE_PANEL_TYPE}!")
        for panel_id in layout["configById"]
    )
    assert '"publish"' not in LAYOUT.read_text(encoding="utf-8")
    assert system.observability.topics.front_image in LAYOUT.read_text(encoding="utf-8")
    assert system.observability.topics.diagnostics in LAYOUT.read_text(encoding="utf-8")


def test_foxglove_launch_keeps_publish_and_parameters_closed() -> None:
    root = ET.parse(FOXGLOVE_LAUNCH).getroot()
    declared = {item.attrib["name"]: item.attrib for item in root.findall("arg")}
    include = root.find("include")
    assert include is not None
    forwarded = {
        item.attrib["name"]: item.attrib["value"] for item in include.findall("arg")
    }

    assert "default" not in declared["service_whitelist"]
    assert forwarded["service_whitelist"] == "$(arg service_whitelist)"
    assert forwarded["param_whitelist"] == "$(arg no_match_allowlist)"
    assert forwarded["client_topic_whitelist"] == "$(arg no_match_allowlist)"


def test_observability_diagnostics_preserve_safety_state_and_raw_codes() -> None:
    active = relay_diagnostic(
        '{"state":"ACTIVE","reason":"","motor_error_codes":[0,0,0,0,0,0,0]}',
        age_s=0.1,
        max_age_s=1.0,
    )
    fault = relay_diagnostic(
        '{"state":"FAULT","reason":"joint feedback stale"}',
        age_s=0.1,
        max_age_s=1.0,
    )
    stale = relay_diagnostic(
        '{"state":"ACTIVE","reason":""}',
        age_s=1.1,
        max_age_s=1.0,
    )
    motors = motor_diagnostic(
        (0, 0, 68, 0, 0, 0, 72),
        arm_joints=6,
        gripper_ignored_error_mask=8,
    )

    assert active.level == DIAGNOSTIC_OK
    assert fault.level == DIAGNOSTIC_ERROR
    assert stale.level == DIAGNOSTIC_ERROR
    assert motors.level == DIAGNOSTIC_ERROR
    assert "J3=68" in motors.message
    assert ("motor_7_error", "72") in motors.values


def test_camera_diagnostic_uses_tracked_freshness_and_pair_skew() -> None:
    healthy = camera_diagnostic(
        connected=True,
        front_age_s=0.1,
        wrist_age_s=0.2,
        pair_skew_s=0.02,
        max_age_s=0.5,
        max_pair_skew_s=0.1,
    )
    stale = camera_diagnostic(
        connected=True,
        front_age_s=0.6,
        wrist_age_s=0.2,
        pair_skew_s=0.02,
        max_age_s=0.5,
        max_pair_skew_s=0.1,
    )
    disconnected = camera_diagnostic(
        connected=False,
        front_age_s=None,
        wrist_age_s=None,
        pair_skew_s=None,
        max_age_s=0.5,
        max_pair_skew_s=0.1,
    )

    assert healthy.level == DIAGNOSTIC_OK
    assert stale.level == DIAGNOSTIC_WARN
    assert disconnected.level == DIAGNOSTIC_WARN


def test_operator_panel_status_is_versioned_and_mirrored_without_commands_or_logs() -> (
    None
):
    telemetry = operator_panel_telemetry(
        {
            "schema_version": WORKFLOW_STATUS_SCHEMA_VERSION,
            "revision": 7,
            "input_revision": 3,
            "run_id": "run-1",
            "state": "waiting_for_input",
            "active": True,
            "workflow": "collect",
            "name": "Collect",
            "command": ["dangerous-command-is-not-telemetry"],
            "started_at": "2026-09-01T00:00:00+00:00",
            "finished_at": "",
            "exit_code": None,
            "progress": [
                {
                    "id": "episode",
                    "label": "Episode",
                    "current": 1,
                    "total": 3,
                    "phase": "ready",
                    "detail": "Episode 1",
                }
            ],
            "status_line": "waiting",
            "input_actions": [{"id": "enter", "label": "Start", "tone": "primary"}],
            "input_phase": "ready",
            "input_detail": "Episode 1",
            "logs": ["private child log"],
        }
    )
    finding = operator_panel_diagnostic(telemetry)

    assert telemetry["schema_version"] == 2
    assert telemetry["source_schema_version"] == WORKFLOW_STATUS_SCHEMA_VERSION
    assert telemetry["state"] == "waiting_for_input"
    assert "command" not in telemetry
    assert "logs" not in telemetry
    assert finding.level == DIAGNOSTIC_OK
    assert finding.message == "waiting_for_input"

    unavailable = operator_panel_telemetry(None, error="panel not running")
    assert operator_panel_diagnostic(unavailable).level == DIAGNOSTIC_WARN


def test_collection_control_requires_the_exact_active_input_gate() -> None:
    snapshot = {
        "schema_version": WORKFLOW_STATUS_SCHEMA_VERSION,
        "revision": 7,
        "input_revision": 3,
        "run_id": "run-1",
        "state": "waiting_for_input",
        "active": True,
        "workflow": "collect",
        "name": "Collect",
        "started_at": "2026-09-01T00:00:00+00:00",
        "finished_at": "",
        "exit_code": None,
        "progress": [],
        "status_line": "waiting",
        "input_actions": [
            {"id": "start", "label": "Start recording", "tone": "primary"}
        ],
        "input_phase": "ready",
        "input_detail": "Episode 1",
    }

    action = prepare_collection_action(
        snapshot, action_id="start", expected_phase="ready"
    )

    assert action.run_id == "run-1"
    assert action.input_revision == 3
    assert prepare_collection_stop(snapshot) == "run-1"
    with pytest.raises(ValueError, match="expected 'recording'"):
        prepare_collection_action(
            snapshot, action_id="start", expected_phase="recording"
        )
    with pytest.raises(ValueError, match="not currently available"):
        prepare_collection_action(snapshot, action_id="discard", expected_phase="ready")
    with pytest.raises(ValueError, match="already stopping"):
        prepare_collection_stop({**snapshot, "state": "stopping"})
