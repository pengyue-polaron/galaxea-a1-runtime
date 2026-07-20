from pathlib import Path

from galaxea_a1_runtime.lerobot.hardware import make_a1_teleop_processors
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
    assert step.mapping.gripper_source_min == config.gripper.source_min
    assert step.mapping.gripper_source_max == config.gripper.source_max
