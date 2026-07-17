from types import SimpleNamespace

import pytest

from galaxea_a1_runtime.runtime.ros_feedback import (
    ordered_joint_positions,
    wait_for_staged_joint_alignment,
)


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


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeStagedMonitor:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = []

    def max_error(self, target, dof, *, max_age_s):
        self.calls.append((tuple(target), dof, max_age_s))
        if len(self.errors) > 1:
            return self.errors.pop(0)
        return self.errors[0]


def test_initial_staged_alignment_waits_for_complete_fresh_hold():
    clock = FakeClock()
    monitor = FakeStagedMonitor([None, 0.08, 0.01])

    error = wait_for_staged_joint_alignment(
        monitor,
        (0.1, 0.2),
        dof=2,
        timeout_s=1.0,
        max_age_s=0.25,
        tolerance_rad=0.05,
        is_shutdown=lambda: False,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert error == pytest.approx(0.01)
    assert monitor.calls[-1] == ((0.1, 0.2), 2, 0.25)


def test_initial_staged_alignment_fails_closed_on_timeout():
    clock = FakeClock()
    monitor = FakeStagedMonitor([None])

    with pytest.raises(RuntimeError, match="no complete fresh staged command"):
        wait_for_staged_joint_alignment(
            monitor,
            (0.1, 0.2),
            dof=2,
            timeout_s=0.05,
            max_age_s=0.25,
            tolerance_rad=0.05,
            is_shutdown=lambda: False,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
