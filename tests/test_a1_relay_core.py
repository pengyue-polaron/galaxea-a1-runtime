import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "runtime"))

from a1_relay_core import RelayInputs, check_initial_alignment, validate_inputs


def healthy_inputs(**overrides):
    values = dict(
        enabled=True,
        joint_age=0.01,
        source_age=0.01,
        status_age=0.01,
        joint_count=7,
        source_count=6,
        motor_error_codes=(0, 0, 0, 0, 0, 0, 0),
    )
    values.update(overrides)
    return RelayInputs(**values)


def test_validate_inputs_accepts_idle_timeout_code():
    assert (
        validate_inputs(
            healthy_inputs(motor_error_codes=(64, 64, 64, 64, 64, 64, 64)),
            arm_joints=6,
            max_age=0.25,
        )
        is None
    )


def test_validate_inputs_rejects_motor_error_with_extra_bits():
    reason = validate_inputs(
        healthy_inputs(motor_error_codes=(0, 0, 68, 0, 0, 0, 0)),
        arm_joints=6,
        max_age=0.25,
    )
    assert reason == "motor errors: J3=68"


def test_validate_inputs_rejects_empty_joint_feedback():
    reason = validate_inputs(healthy_inputs(joint_count=0), arm_joints=6, max_age=0.25)
    assert "joint feedback has 0" in reason


def test_validate_inputs_locked_is_not_a_fault():
    assert validate_inputs(healthy_inputs(enabled=False), arm_joints=6, max_age=0.25) == "locked"


def test_initial_alignment_check_accepts_small_start_error():
    current = [0.1, -0.2, 0.3]
    raw = [0.08, -0.22, 0.31]

    assert check_initial_alignment(current, raw, max_abs_error=0.05) is None


def test_initial_alignment_check_rejects_large_start_error():
    try:
        check_initial_alignment([0.0], [0.1], max_abs_error=0.05)
    except ValueError as exc:
        assert "initial command error exceeds" in str(exc)
    else:
        raise AssertionError("expected large initial command error to be rejected")
