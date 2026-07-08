import pytest

from galaxea_a1_runtime.teleop import (
    JointMappingConfig,
    detect_leader_joint_keys,
    map_leader_joints_to_a1,
    parse_csv_floats,
)


def test_detect_leader_joint_keys_supports_current_so_leader_names():
    action = {f"joint{i}.pos": float(i) for i in range(6)}

    assert detect_leader_joint_keys(action, 6) == tuple(f"joint{i}.pos" for i in range(6))


def test_detect_leader_joint_keys_supports_legacy_names():
    action = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }

    assert detect_leader_joint_keys(action, 6) == (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )


def test_relative_mapping_starts_at_current_a1_pose_and_applies_signs():
    config = JointMappingConfig(
        sign=(-1.0, 1.0),
        scale=(1.0, 2.0),
        bias_rad=(0.0, 0.0),
        lower_limits=(-10.0, -10.0),
        upper_limits=(10.0, 10.0),
    )

    target = map_leader_joints_to_a1(
        leader_now=(10.0, 20.0),
        leader_start=(0.0, 10.0),
        a1_start=(1.0, 2.0),
        config=config,
    )

    assert target[0] == pytest.approx(1.0 - 0.1745329)
    assert target[1] == pytest.approx(2.0 + 2.0 * 0.1745329)


def test_parse_csv_floats_validates_length():
    with pytest.raises(ValueError, match="expects 3"):
        parse_csv_floats("1,2", 3, "values")
