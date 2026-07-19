from pathlib import Path

import pytest

from galaxea_a1_runtime.lerobot.hardware import make_a1_teleop_processors
from galaxea_a1_runtime.teleop import (
    JointMappingConfig,
    detect_leader_joint_keys,
    map_leader_joints_to_a1,
)
from galaxea_a1_runtime.teleop.config import load_teleop_config


REPO = Path(__file__).resolve().parents[1]


def test_detect_leader_joint_keys_supports_current_so_leader_names():
    action = {f"joint{i}.pos": float(i) for i in range(6)}

    assert detect_leader_joint_keys(action, 6) == tuple(
        f"joint{i}.pos" for i in range(6)
    )


def test_detect_leader_joint_keys_rejects_upstream_so_names():
    action = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }

    with pytest.raises(RuntimeError, match="joint0.pos"):
        detect_leader_joint_keys(action, 6)


def test_detect_leader_joint_keys_rejects_unknown_order_instead_of_sorting():
    action = {f"axis_{index}.pos": float(index) for index in range(6)}

    with pytest.raises(RuntimeError, match="joint0.pos"):
        detect_leader_joint_keys(action, 6)


def test_relative_mapping_starts_at_current_a1_pose_and_applies_signs():
    config = JointMappingConfig(
        relative=True,
        input_degrees=True,
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


def test_joint_mapping_rejects_non_finite_hardware_input():
    config = JointMappingConfig(
        relative=True,
        input_degrees=True,
        sign=(1.0,),
        scale=(1.0,),
        bias_rad=(0.0,),
        lower_limits=(-1.0,),
        upper_limits=(1.0,),
    )

    with pytest.raises(ValueError, match="leader_now"):
        map_leader_joints_to_a1(
            leader_now=(float("nan"),),
            leader_start=(0.0,),
            a1_start=(0.0,),
            config=config,
        )


def test_plugin_processor_mapping_is_derived_from_the_tracked_teleop_config():
    config = load_teleop_config(
        REPO / "configs/teleop/a1_so100.toml",
        repo_root=REPO,
    )

    teleop_processor, _, _ = make_a1_teleop_processors(config)
    step = teleop_processor.steps[0]

    assert step.mapping.sign == config.bridge.mapping.sign
    assert step.mapping.lower_limits_rad == config.system.joint_safety.lower_limits
    assert step.mapping.upper_limits_rad == config.system.joint_safety.upper_limits
    assert step.mapping.gripper_source_min == config.gripper.source_min
    assert step.mapping.gripper_source_max == config.gripper.source_max
