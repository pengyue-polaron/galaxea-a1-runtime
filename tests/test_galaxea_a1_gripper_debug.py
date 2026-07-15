import pytest

from galaxea_a1_runtime.apps.teleop.gripper_debug import (
    build_gripper_debug_reading,
    format_gripper_debug_reading,
)


def test_gripper_debug_reports_unified_action_and_measured_feedback():
    reading = build_gripper_debug_reading(
        target_mm=104.0,
        feedback_mm=103.833,
        stroke_min_mm=0.0,
        stroke_max_mm=104.0,
        source_min=0.0,
        source_max=53.16,
        invert=False,
    )

    assert reading.leader_position == pytest.approx(53.16)
    assert reading.action_normalized == pytest.approx(1.0)
    assert reading.state_normalized == pytest.approx(103.833 / 104.0)
    assert reading.error_mm == pytest.approx(-0.167)
    text = format_gripper_debug_reading(reading, relay_summary="ACTIVE:  (fresh)")
    assert "leader= 53.160" in text
    assert "action= 1.000" in text
    assert "target=104.000 mm" in text
    assert "A1=103.833 mm" in text
    assert "state= 0.998" in text


def test_gripper_debug_rejects_target_outside_tracked_range():
    with pytest.raises(ValueError, match="outside"):
        build_gripper_debug_reading(
            target_mm=104.1,
            feedback_mm=103.5,
            stroke_min_mm=0.0,
            stroke_max_mm=104.0,
            source_min=0.0,
            source_max=53.16,
            invert=False,
        )
