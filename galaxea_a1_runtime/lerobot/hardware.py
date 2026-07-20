"""Application composition for the out-of-tree A1 LeRobot plugins."""

from __future__ import annotations

from lerobot_robot_galaxea_a1 import (
    GalaxeaA1TeleopMapping,
    make_galaxea_a1_processors,
)

from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def make_a1_teleop_processors(config: TeleopConfig):
    """Build pair-specific processors exclusively from the tracked Teleop config."""

    mapping = config.bridge.mapping
    gripper = config.gripper
    return make_galaxea_a1_processors(
        GalaxeaA1TeleopMapping(
            sign=mapping.sign,
            scale=mapping.scale,
            bias_rad=mapping.bias_rad,
            lower_limits_rad=mapping.lower_limits,
            upper_limits_rad=mapping.upper_limits,
            gripper_source_min=gripper.source_min,
            gripper_source_max=gripper.source_max,
            gripper_invert=gripper.invert,
            gripper_saturate=gripper.saturate_out_of_range,
        )
    )
