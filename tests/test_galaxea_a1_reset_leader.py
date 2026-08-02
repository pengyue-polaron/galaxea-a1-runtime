from types import SimpleNamespace

import pytest

from galaxea_a1_runtime.apps.teleop import reset_leader
from galaxea_a1_runtime.apps.teleop.reset_config import LeaderMotion


TARGET = {
    **{f"joint{index}.pos": 0.0 for index in range(6)},
    "gripper.pos": 0.0,
}
MOTION = LeaderMotion(
    hz=30.0,
    min_duration_s=2.0,
    max_velocity_units_s=60.0,
    hold_s=1.0,
    max_goal_attempts=3,
    goal_tolerance_units=2.0,
    gripper_goal_tolerance_units=5.0,
)


class FakeLeader:
    def __init__(self, readings):
        self.readings = iter(readings)

    def get_action(self):
        return next(self.readings)


def test_leader_goal_uses_bounded_correction_without_relaxing_tolerance(monkeypatch):
    first = {**TARGET, "joint4.pos": 2.286}
    leader = FakeLeader([first, TARGET])
    moves = []
    monkeypatch.setattr(
        reset_leader,
        "move_leader_smooth",
        lambda _leader, start, target, _motion, _progress: moves.append(
            (start, target)
        ),
    )

    reset_leader.move_leader_to_goal(
        leader,
        {**TARGET, "joint0.pos": 20.0},
        TARGET,
        MOTION,
        SimpleNamespace(update=lambda *_args: None),
    )

    assert len(moves) == 2
    assert moves[1][0] == first


def test_leader_goal_reports_joint_after_bounded_attempts(monkeypatch):
    outside = {**TARGET, "joint4.pos": 2.286}
    leader = FakeLeader([outside, outside, outside])
    monkeypatch.setattr(reset_leader, "move_leader_smooth", lambda *_args: None)

    with pytest.raises(
        RuntimeError,
        match=(
            r"did not converge after 3 attempts "
            r"\(joint4.pos error 2\.286 > 2\.000\)"
        ),
    ):
        reset_leader.move_leader_to_goal(
            leader,
            TARGET,
            TARGET,
            MOTION,
            SimpleNamespace(update=lambda *_args: None),
        )
