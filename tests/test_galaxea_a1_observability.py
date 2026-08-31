import re
from pathlib import Path

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.foxglove_layout import (
    build_foxglove_layout,
    render_foxglove_layout,
)
from galaxea_a1_runtime.observability import (
    DIAGNOSTIC_ERROR,
    DIAGNOSTIC_OK,
    DIAGNOSTIC_WARN,
    NO_MATCH_ALLOWLIST,
    READ_ONLY_FOXGLOVE_CAPABILITIES,
    camera_diagnostic,
    foxglove_asset_uri_allowlist,
    foxglove_topic_whitelist,
    motor_diagnostic,
    operator_panel_diagnostic,
    operator_panel_telemetry,
    relay_diagnostic,
)


REPO = Path(__file__).resolve().parents[1]
SYSTEM = REPO / "configs/system/a1.toml"
LAYOUT = REPO / "foxglove/layouts/a1_observability.json"


def test_foxglove_contract_is_exact_and_read_only() -> None:
    system = load_system_config(SYSTEM, repo_root=REPO)
    topic_patterns = foxglove_topic_whitelist(system)
    asset_patterns = foxglove_asset_uri_allowlist(system)

    assert READ_ONLY_FOXGLOVE_CAPABILITIES == ("connectionGraph", "assets")
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


def test_tracked_foxglove_layout_is_current_and_contains_no_publish_panel() -> None:
    system = load_system_config(SYSTEM, repo_root=REPO)
    layout = build_foxglove_layout(system)

    assert LAYOUT.read_text(encoding="utf-8") == render_foxglove_layout(system)
    assert all(not panel_id.startswith("Publish!") for panel_id in layout["configById"])
    assert '"publish"' not in LAYOUT.read_text(encoding="utf-8")
    assert system.observability.topics.front_image in LAYOUT.read_text(encoding="utf-8")
    assert system.observability.topics.diagnostics in LAYOUT.read_text(encoding="utf-8")


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
            "schema_version": 1,
            "revision": 7,
            "run_id": "run-1",
            "state": "waiting_for_input",
            "active": True,
            "workflow": "collect",
            "name": "Collect",
            "command": ["dangerous-command-is-not-telemetry"],
            "started_at": "2026-09-01T00:00:00+00:00",
            "finished_at": "",
            "exit_code": None,
            "progress": [{"id": "episode", "current": 1}],
            "status_line": "waiting",
            "input_actions": [{"id": "enter", "label": "Start"}],
            "logs": ["private child log"],
        }
    )
    finding = operator_panel_diagnostic(telemetry)

    assert telemetry["schema_version"] == 1
    assert telemetry["source_schema_version"] == 1
    assert telemetry["state"] == "waiting_for_input"
    assert "command" not in telemetry
    assert "logs" not in telemetry
    assert finding.level == DIAGNOSTIC_OK
    assert finding.message == "waiting_for_input"

    unavailable = operator_panel_telemetry(None, error="panel not running")
    assert operator_panel_diagnostic(unavailable).level == DIAGNOSTIC_WARN
