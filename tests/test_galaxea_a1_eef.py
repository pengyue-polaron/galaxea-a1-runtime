from math import cos, sin

import pytest

from galaxea_a1_runtime.hardware.eef import EefPose, action_to_eef_target
from galaxea_a1_runtime.policies.actions import RuntimeAction
from galaxea_a1_runtime.schema import ActionMode


def test_translation_action_adds_delta_and_holds_orientation():
    current = EefPose((0.2, -0.1, 0.3), (0.0, 0.0, 0.0, 1.0), "base_link")
    action = RuntimeAction(
        mode=ActionMode.EEF_TRANSLATION,
        values=(0.01, -0.02, 0.03, 0.5),
        names=("delta_x", "delta_y", "delta_z", "gripper"),
    )

    target = action_to_eef_target(current, action)

    assert target is not None
    assert target.xyz == pytest.approx((0.21, -0.12, 0.33))
    assert target.quat_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert target.frame_id == "base_link"


def test_gripper_only_action_has_no_eef_target():
    current = EefPose((0.2, -0.1, 0.3), (0.0, 0.0, 0.0, 1.0))
    action = RuntimeAction(
        mode=ActionMode.EEF_TRANSLATION,
        values=(0.0, 0.0, 0.0, 0.5),
        names=("delta_x", "delta_y", "delta_z", "gripper"),
    )

    assert action_to_eef_target(current, action) is None


def test_eef_delta_composes_rotation():
    yaw = 0.2
    current = EefPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    action = RuntimeAction(
        mode=ActionMode.EEF_DELTA,
        values=(0.0, 0.0, 0.0, 0.0, 0.0, yaw, 0.0),
        names=(
            "delta_x",
            "delta_y",
            "delta_z",
            "delta_roll",
            "delta_pitch",
            "delta_yaw",
            "gripper",
        ),
    )

    target = action_to_eef_target(current, action)

    assert target is not None
    assert target.xyz == pytest.approx((0.0, 0.0, 0.0))
    assert target.quat_xyzw == pytest.approx((0.0, 0.0, sin(yaw / 2.0), cos(yaw / 2.0)))


def test_joint_action_is_rejected_by_safe_eef_converter():
    current = EefPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    action = RuntimeAction(
        mode=ActionMode.JOINT_ABSOLUTE,
        values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        names=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"),
    )

    with pytest.raises(ValueError, match="does not support"):
        action_to_eef_target(current, action)
