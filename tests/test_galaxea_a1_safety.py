import pytest

from galaxea_a1_runtime.constants import IDLE_TIMEOUT_CODE
from galaxea_a1_runtime.safety import (
    RelayInputs,
    validate_relay_inputs,
    validate_initial_alignment,
)


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


def test_relay_accepts_idle_timeout_code_for_arm_joints():
    decision = validate_relay_inputs(
        healthy_inputs(motor_error_codes=(IDLE_TIMEOUT_CODE,) * 7),
        arm_joints=6,
        max_input_age=0.25,
        max_status_age=1.0,
    )

    assert decision.allowed is True
    assert decision.reason is None


def test_relay_blocks_extra_motor_error_bits():
    decision = validate_relay_inputs(
        healthy_inputs(motor_error_codes=(0, 0, 68, 0, 0, 0, 0)),
        arm_joints=6,
        max_input_age=0.25,
        max_status_age=1.0,
    )

    assert decision.allowed is False
    assert decision.reason == "motor errors: J3=68"


def test_relay_locked_is_fail_closed_state():
    decision = validate_relay_inputs(
        healthy_inputs(enabled=False), max_input_age=0.25, max_status_age=1.0
    )

    assert decision.allowed is False
    assert decision.reason == "locked"


def test_initial_alignment_accepts_small_first_error_without_modifying_command():
    current = [0.1, -0.2, 0.3]
    raw = [0.08, -0.22, 0.31]

    assert validate_initial_alignment(current, raw, max_abs_error=0.05) is None


def test_initial_alignment_rejects_large_first_error():
    with pytest.raises(ValueError, match="initial command error exceeds"):
        validate_initial_alignment([0.0], [0.1], max_abs_error=0.05)
