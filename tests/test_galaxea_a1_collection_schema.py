from galaxea_a1_runtime.apps.teleop.collection_task import normalize_collection_task
from galaxea_a1_runtime.collection import (
    find_joint_action_step_violation,
)


def test_collection_tasks_are_normalized_per_episode():
    assert normalize_collection_task("  pick fruit  ") == "pick fruit"
    assert normalize_collection_task("place fruit") == "place fruit"


def test_joint_action_quality_check_rejects_discontinuity():
    violation = find_joint_action_step_violation(
        [(0.0, 0.0), (0.1, 0.2), (0.15, 1.0)],
        action_names=("joint_1", "joint_2"),
        max_step_rad=0.35,
    )

    assert violation is not None
    assert violation.frame_index == 2
    assert violation.joint_name == "joint_2"
    assert violation.step_rad == 0.8


def test_joint_action_quality_check_accepts_continuous_actions():
    assert (
        find_joint_action_step_violation(
            [(0.0, 0.0, 0.0), (0.1, -0.1, 1.0), (0.2, -0.2, 0.0)],
            action_names=("joint_1", "joint_2", "gripper"),
            max_step_rad=0.35,
        )
        is None
    )
