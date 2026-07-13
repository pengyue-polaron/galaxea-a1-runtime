import numpy as np
import pytest

from galaxea_a1_runtime.apps.lingbot.actions import (
    LingBotActionConfig,
    gripper_norm_from_stroke,
    gripper_stroke_from_norm,
    normalize_condition_action,
    prepare_policy_action,
    sanitize_policy_action,
    tracker_command_action,
)


def test_condition_action_preserves_feedback_xyz_outside_workspace():
    cfg = LingBotActionConfig(xyz_min=(0.0, 0.0, 0.0), xyz_max=(1.0, 1.0, 1.0))

    action = normalize_condition_action([2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 2.0, 1.2], cfg)

    assert action[:3] == pytest.approx((2.0, -1.0, 0.5))
    assert action[3:7] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert action[7] == pytest.approx(1.0)


def test_policy_action_applies_workspace_without_per_step_delta_clamp():
    cfg = LingBotActionConfig(
        xyz_min=(0.0, 0.0, 0.0),
        xyz_max=(1.0, 1.0, 1.0),
    )

    action = sanitize_policy_action(
        [0.3, -0.2, 2.0, 0.0, 0.0, 0.0, 1.0, -0.5],
        cfg,
        current_xyz=(0.1, 0.1, 0.1),
    )

    assert action[:3] == pytest.approx((0.3, 0.0, 1.0))
    assert action[7] == pytest.approx(0.0)


def test_hold_current_orientation_replaces_model_quaternion():
    cfg = LingBotActionConfig(orientation_mode="hold-current")

    action = prepare_policy_action(
        [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.5],
        cfg,
        current_xyz=None,
        current_quat=(0.0, 1.0, 0.0, 0.0),
        require_current_orientation=True,
    )

    assert action[3:7] == pytest.approx((0.0, 1.0, 0.0, 0.0))


def test_servo_compensation_is_off_by_default_and_explicit_when_enabled():
    policy_action = np.array([0.20, 0.10, 0.10, 0.0, 0.0, 0.0, 1.0, 0.5])

    off = tracker_command_action(policy_action, LingBotActionConfig(), current_xyz=(0.10, 0.10, 0.10))
    on = tracker_command_action(
        policy_action,
        LingBotActionConfig(eef_servo_gain=1.5, eef_servo_max_extra=1.0),
        current_xyz=(0.10, 0.10, 0.10),
    )

    assert off[:3] == pytest.approx((0.20, 0.10, 0.10))
    assert on[:3] == pytest.approx((0.25, 0.10, 0.10))


def test_gripper_mapping_is_binary():
    cfg = LingBotActionConfig()

    assert gripper_stroke_from_norm(0.49, cfg) == pytest.approx(0.0)
    assert gripper_stroke_from_norm(0.50, cfg) == pytest.approx(200.0)
    assert gripper_norm_from_stroke(29.9, cfg) == pytest.approx(0.0)
    assert gripper_norm_from_stroke(30.0, cfg) == pytest.approx(1.0)
