import numpy as np
import pytest

from galaxea_a1_runtime.apps.lingbot.actions import (
    LingBotActionConfig,
    absolute_action_to_relative,
    gripper_norm_from_stroke,
    gripper_stroke_from_norm,
    normalize_condition_action,
    prepare_policy_action,
    relative_action_to_absolute,
    sanitize_policy_action,
    tracker_command_action,
)


def action_config(**overrides) -> LingBotActionConfig:
    values = {
        "xyz_min": (0.06, -0.27, 0.06),
        "xyz_max": (0.44, 0.14, 0.50),
        "min_quat_norm": 0.25,
        "orientation_mode": "hold-current",
        "gripper_stroke_min": 0.0,
        "gripper_stroke_max": 100.0,
    }
    values.update(overrides)
    return LingBotActionConfig(**values)


def test_condition_action_preserves_feedback_xyz_outside_workspace():
    cfg = action_config(xyz_min=(0.0, 0.0, 0.0), xyz_max=(1.0, 1.0, 1.0))

    action = normalize_condition_action([2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 2.0, 1.2], cfg)

    assert action[:3] == pytest.approx((2.0, -1.0, 0.5))
    assert action[3:7] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert action[7] == pytest.approx(1.0)


def test_policy_action_applies_workspace_without_per_step_delta_clamp():
    cfg = action_config(
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
    cfg = action_config(orientation_mode="hold-current")

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

    off = tracker_command_action(policy_action, action_config(), current_xyz=(0.10, 0.10, 0.10))
    on = tracker_command_action(
        policy_action,
        action_config(eef_servo_gain=1.5, eef_servo_max_extra=1.0),
        current_xyz=(0.10, 0.10, 0.10),
    )

    assert off[:3] == pytest.approx((0.20, 0.10, 0.10))
    assert on[:3] == pytest.approx((0.25, 0.10, 0.10))


def test_gripper_mapping_is_continuous_across_shared_stroke():
    cfg = action_config()

    assert gripper_stroke_from_norm(0.25, cfg) == pytest.approx(25.0)
    assert gripper_stroke_from_norm(0.50, cfg) == pytest.approx(50.0)
    assert gripper_norm_from_stroke(50.0, cfg) == pytest.approx(0.50)
    assert gripper_norm_from_stroke(100.0, cfg) == pytest.approx(1.0)


def test_gripper_mapping_clips_to_configured_stroke():
    cfg = action_config(
        gripper_stroke_min=0.0,
        gripper_stroke_max=80.0,
    )

    action = sanitize_policy_action(
        [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.25],
        cfg,
        current_xyz=None,
    )

    assert action[7] == pytest.approx(0.25)
    assert gripper_stroke_from_norm(action[7], cfg) == pytest.approx(20.0)
    assert gripper_stroke_from_norm(2.0, cfg) == pytest.approx(80.0)


def test_episode_relative_action_roundtrips_xyz_and_quaternion():
    origin = [0.2, -0.1, 0.3, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5), 1.0]
    relative = [0.1, 0.02, -0.03, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5), 0.25]

    absolute = relative_action_to_absolute(relative, origin, min_quat_norm=0.25)
    recovered = absolute_action_to_relative(absolute, origin, min_quat_norm=0.25)

    assert absolute[:3] == pytest.approx((0.3, -0.08, 0.27))
    assert abs(float(np.dot(recovered[3:7], relative[3:7]))) == pytest.approx(1.0)
    assert recovered[:3] == pytest.approx(relative[:3])
    assert recovered[7] == pytest.approx(0.25)
