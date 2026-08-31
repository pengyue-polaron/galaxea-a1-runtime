import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from galaxea_a1_runtime.safety import (
    RelayInputs,
    actuator_error_block_reason,
    gripper_stroke_block_reason,
    relay_block_reason,
    require_finite_vector,
    validate_arm_control_command,
    validate_initial_alignment,
)


REPO = Path(__file__).resolve().parents[1]


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
        relay_block_reason(
            healthy_inputs(motor_error_codes=(64, 64, 64, 64, 64, 64, 64)),
            arm_joints=6,
            max_input_age=0.25,
            max_status_age=1.0,
        )
        is None
    )


def test_validate_inputs_rejects_motor_error_with_extra_bits():
    reason = relay_block_reason(
        healthy_inputs(motor_error_codes=(0, 0, 68, 0, 0, 0, 0)),
        arm_joints=6,
        max_input_age=0.25,
        max_status_age=1.0,
    )
    assert reason == "motor errors: J3=68"


def test_validate_inputs_rejects_empty_joint_feedback():
    reason = relay_block_reason(
        healthy_inputs(joint_count=0),
        arm_joints=6,
        max_input_age=0.25,
        max_status_age=1.0,
    )
    assert "joint feedback has 0" in reason


def test_validate_inputs_locked_is_not_a_fault():
    assert (
        relay_block_reason(
            healthy_inputs(enabled=False),
            arm_joints=6,
            max_input_age=0.25,
            max_status_age=1.0,
        )
        == "locked"
    )


def test_relay_uses_independent_input_and_status_freshness_limits():
    assert (
        relay_block_reason(
            healthy_inputs(status_age=0.8),
            arm_joints=6,
            max_input_age=0.25,
            max_status_age=1.0,
        )
        is None
    )
    assert "motor status stale" in relay_block_reason(
        healthy_inputs(status_age=1.1),
        arm_joints=6,
        max_input_age=0.25,
        max_status_age=1.0,
    )


def test_initial_alignment_check_accepts_small_start_error():
    current = [0.1, -0.2, 0.3]
    raw = [0.08, -0.22, 0.31]

    assert validate_initial_alignment(current, raw, max_abs_error=0.05) is None


def test_initial_alignment_check_rejects_large_start_error():
    try:
        validate_initial_alignment([0.0], [0.1], max_abs_error=0.05)
    except ValueError as exc:
        assert "initial command error exceeds" in str(exc)
    else:
        raise AssertionError("expected large initial command error to be rejected")


def test_gripper_relay_accepts_only_finite_targets_in_system_range():
    assert gripper_stroke_block_reason(100.0, minimum_mm=0.0, maximum_mm=100.0) is None
    assert "outside" in gripper_stroke_block_reason(
        100.1,
        minimum_mm=0.0,
        maximum_mm=100.0,
    )
    assert "not finite" in gripper_stroke_block_reason(
        float("nan"),
        minimum_mm=0.0,
        maximum_mm=100.0,
    )


def test_gripper_relay_accepts_idle_status_but_rejects_extra_error_bits():
    assert (
        actuator_error_block_reason((0, 0, 0, 0, 0, 0, 64), index=6, label="gripper")
        is None
    )
    assert (
        actuator_error_block_reason(
            (0, 0, 0, 0, 0, 0, 72),
            index=6,
            label="gripper",
            ignored_mask=8,
        )
        is None
    )
    assert (
        actuator_error_block_reason(
            (0, 0, 0, 0, 0, 0, 76),
            index=6,
            label="gripper",
            ignored_mask=8,
        )
        == "gripper motor error: 76"
    )


def test_relay_vector_validation_rejects_short_and_non_finite_inputs():
    assert require_finite_vector([1, 2, 3], count=2, label="command") == (1.0, 2.0)
    try:
        require_finite_vector([1], count=2, label="command")
    except ValueError as exc:
        assert "need 2" in str(exc)
    else:
        raise AssertionError("short relay command must fail")
    try:
        require_finite_vector([1, float("nan")], count=2, label="command")
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite relay command must fail")
    assert (
        actuator_error_block_reason((0, 0, 0, 0, 0, 0, 68), index=6, label="gripper")
        == "gripper motor error: 68"
    )


def test_relay_callback_reorders_named_joint_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rospy = ModuleType("rospy")
    rospy.Publisher = lambda *_args, **_kwargs: SimpleNamespace(
        publish=lambda _msg: None
    )
    rospy.Subscriber = lambda *_args, **_kwargs: None
    rospy.logerr = lambda *_args: None
    message_modules = {
        "sensor_msgs.msg": {"JointState": type("JointState", (), {})},
        "signal_arm.msg": {
            "arm_control": type("arm_control", (), {}),
            "gripper_position_control": type("gripper_position_control", (), {}),
            "status_stamped": type("status_stamped", (), {}),
        },
        "std_msgs.msg": {
            "Bool": type("Bool", (), {}),
            "String": type("String", (), {}),
        },
    }
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    for name, attributes in message_modules.items():
        package_name = name.partition(".")[0]
        package = ModuleType(package_name)
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        setattr(package, "msg", module)
        monkeypatch.setitem(sys.modules, package_name, package)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "galaxea_a1_runtime.runtime.ros1_env.configure_ros1_python",
        lambda _root: None,
    )
    namespace = runpy.run_path(str(REPO / "scripts/runtime/safe_arm_command_relay.py"))
    config = namespace["RelayRuntimeConfig"](
        input_topic="/staged",
        output_topic="/host",
        joint_topic="/joints",
        motor_status_topic="/status",
        enable_topic="/enable",
        relay_status_topic="/relay",
        gripper_input_topic="/gripper_target",
        gripper_output_topic="/gripper_host",
        gripper_min_stroke_mm=0.0,
        gripper_max_stroke_mm=100.0,
        joint_names=("j1", "j2"),
        rate=100.0,
        status_rate=5.0,
        max_input_age=0.25,
        max_status_age=1.0,
        arming_timeout=1.0,
        max_initial_error=0.05,
        gripper_ignored_error_mask=0,
        allowed_control_modes=(0,),
    )
    relay = namespace["SafeArmCommandRelay"](config)

    relay._joint_cb(SimpleNamespace(name=["j2", "j1"], position=[2.0, 1.0]))
    assert relay.joints == [1.0, 2.0]

    relay._joint_cb(SimpleNamespace(name=["j1", "j1"], position=[1.0, 2.0]))
    assert relay.joints == []
    assert "duplicate" in relay.fault_reason


def _validate_arm_command(**overrides):
    values = {
        "p_des": [0.0] * 6,
        "v_des": [0.0] * 6,
        "kp": [20.0] * 6,
        "kd": [1.0] * 6,
        "t_ff": [0.0] * 6,
        "mode": 0,
        "arm_joints": 6,
        "allowed_modes": (0,),
    }
    values.update(overrides)
    return validate_arm_control_command(**values)


def test_arm_command_validator_checks_every_driver_field():
    assert _validate_arm_command() is None

    for override in (
        {"v_des": [0.0] * 5},
        {"t_ff": [0.0] * 5 + [float("nan")]},
        {"kp": [20.0] * 5 + [-1.0]},
        {"mode": 1},
    ):
        try:
            _validate_arm_command(**override)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe arm control field must be rejected")
