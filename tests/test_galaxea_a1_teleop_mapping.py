from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.teleop.pairing import GalaxeaA1TeleopMapping
from galaxea_a1_runtime.apps.teleop.processors import make_a1_teleop_processors
from galaxea_a1_runtime.teleop.config import load_teleop_config

REPO = Path(__file__).resolve().parents[1]


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
    assert (
        step.mapping.max_joint_action_step_rad
        == config.bridge.max_joint_action_step_rad
    )
    assert step.mapping.gripper_source_min == config.gripper.source_min
    assert step.mapping.gripper_source_max == config.gripper.source_max


def test_first_pairing_frame_is_exact_robot_hold_including_gripper() -> None:
    mapping = GalaxeaA1TeleopMapping(
        sign=(1.0,) * 6,
        scale=(1.0,) * 6,
        bias_rad=(0.1,) * 6,
        lower_limits_rad=(-3.0,) * 6,
        upper_limits_rad=(3.0,) * 6,
        max_joint_action_step_rad=0.35,
    )
    from galaxea_a1_runtime.apps.teleop.pairing import make_galaxea_a1_processors

    processor, _, _ = make_galaxea_a1_processors(mapping)
    observation = {
        **{f"joint_{index}_rad": index / 10 for index in range(1, 7)},
        "gripper_normalized": 0.8,
    }
    leader = {
        **{f"joint{index}.pos": 0.0 for index in range(6)},
        "gripper.pos": 0.0,
    }

    assert processor((leader, observation)) == observation
    second = processor((leader, observation))
    assert second["joint_1_rad"] == pytest.approx(observation["joint_1_rad"] + 0.1)
    assert second["gripper_normalized"] == 0.0
