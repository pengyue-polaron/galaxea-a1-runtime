from types import SimpleNamespace

import pytest

from galaxea_a1_runtime.runtime.ros_feedback import ordered_joint_positions


JOINTS = ("j1", "j2", "j3")


def message(names, positions):
    return SimpleNamespace(name=names, position=positions)


def test_named_joint_feedback_is_reordered_to_the_system_contract():
    msg = message(["j3", "j1", "extra", "j2"], [3.0, 1.0, 99.0, 2.0])

    assert ordered_joint_positions(msg, JOINTS, label="feedback") == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    ("msg", "error"),
    [
        (message(["j1", "j2"], [1.0, 2.0]), "need 3"),
        (message(["j1", "j1", "j3"], [1.0, 2.0, 3.0]), "duplicate"),
        (message(["j1", "j2", "other"], [1.0, 2.0, 3.0]), "missing"),
        (message(["j1", "j2", "j3"], [1.0, float("nan"), 3.0]), "non-finite"),
    ],
)
def test_named_joint_feedback_fails_closed(msg, error):
    with pytest.raises(ValueError, match=error):
        ordered_joint_positions(msg, JOINTS, label="feedback")


def test_unnamed_compatibility_is_explicit():
    msg = message([], [1.0, 2.0, 3.0])

    assert ordered_joint_positions(msg, JOINTS, label="feedback") == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="must include joint names"):
        ordered_joint_positions(msg, JOINTS, label="target", allow_unnamed=False)
