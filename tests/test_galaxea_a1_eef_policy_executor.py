from __future__ import annotations

import numpy as np
import pytest

from galaxea_a1_runtime.apps.eef_policy_executor import (
    EefPolicyExecutor,
    close_policy_resources,
)


ACTION = np.asarray([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5])


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


class FakeCommander:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def publish_action(self, action, *, publish_gripper: bool) -> None:
        self.events.append(("action", publish_gripper, np.asarray(action).copy()))

    def publish_motion_enable(self, enabled: bool) -> None:
        self.events.append(("enable", enabled))

    def publish_gripper(self, gripper_norm: float) -> None:
        self.events.append(("gripper", gripper_norm))

    def publish_active_target(self) -> None:
        self.events.append("keepalive")


class FakeRelay:
    def __init__(self, *, activates: bool = True) -> None:
        self.active = False
        self.activates = activates

    def status(self):
        if self.activates:
            self.active = True
        return None, 0.0

    def is_active(self) -> bool:
        return self.active

    def summary(self) -> str:
        return "ACTIVE" if self.active else "LOCKED"


class FakeState:
    def __init__(self) -> None:
        self.pose_fresh = True
        self.gripper_fresh = True

    def pose_is_fresh(self) -> bool:
        return self.pose_fresh

    def gripper_is_fresh(self) -> bool:
        return self.gripper_fresh

    def current_xyz(self) -> np.ndarray:
        return ACTION[:3].copy()


def _executor(
    *,
    events: list[object],
    state: FakeState | None = None,
    relay: FakeRelay | None = None,
    settle_s: float = 0.1,
    corrections: int = 2,
) -> EefPolicyExecutor:
    clock = FakeClock()
    return EefPolicyExecutor(
        state=state or FakeState(),
        relay=relay or FakeRelay(),
        commander=FakeCommander(events),
        relay_enable_timeout_s=0.2,
        settle_s=settle_s,
        tolerance_m=0.005,
        corrections=corrections,
        is_shutdown=lambda: False,
        policy_label="test policy",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def test_executor_stages_target_before_unlock_and_delays_gripper_until_active():
    events: list[object] = []
    executor = _executor(events=events)

    command = executor.publish(ACTION)

    assert command == pytest.approx(ACTION)
    assert [(item[0], item[1]) for item in events if isinstance(item, tuple)] == [
        ("action", False),
        ("enable", True),
        ("gripper", pytest.approx(ACTION[7])),
    ]
    assert executor.motion_enabled is True


@pytest.mark.parametrize(
    ("pose_fresh", "gripper_fresh", "expected"),
    [
        (False, True, "EEF pose"),
        (True, False, "gripper"),
        (False, False, "EEF pose, gripper"),
    ],
)
def test_executor_identifies_stale_feedback_before_any_publication(
    pose_fresh: bool, gripper_fresh: bool, expected: str
):
    events: list[object] = []
    state = FakeState()
    state.pose_fresh = pose_fresh
    state.gripper_fresh = gripper_fresh
    executor = _executor(events=events, state=state)

    with pytest.raises(RuntimeError, match=rf"missing or stale: {expected}"):
        executor.publish(ACTION)

    assert events == []


def test_executor_locks_after_relay_enable_timeout():
    events: list[object] = []
    executor = _executor(events=events, relay=FakeRelay(activates=False))

    with pytest.raises(RuntimeError, match="did not become ACTIVE"):
        executor.publish(ACTION)

    assert [(item[0], item[1]) for item in events if isinstance(item, tuple)] == [
        ("action", False),
        ("enable", True),
        ("enable", False),
    ]
    assert executor.motion_enabled is False


def test_executor_stops_immediately_if_an_active_relay_loses_confirmation():
    events: list[object] = []
    relay = FakeRelay()
    executor = _executor(events=events, relay=relay)
    executor.motion_enabled = True

    with pytest.raises(RuntimeError, match="no longer confirmed ACTIVE"):
        executor.publish(ACTION)

    assert events[-1] == ("enable", False)
    assert executor.motion_enabled is False


def test_executor_republishes_only_configured_tracking_corrections():
    events: list[object] = []
    state = FakeState()
    state.current_xyz = lambda: ACTION[:3] + [0.02, 0.0, 0.0]
    executor = _executor(events=events, state=state, settle_s=0.1, corrections=2)

    command = executor.publish(ACTION)

    assert command == pytest.approx(ACTION)
    pose_events = [
        item for item in events if isinstance(item, tuple) and item[0] == "action"
    ]
    assert [(item[1], item[2][0]) for item in pose_events] == [
        (False, pytest.approx(ACTION[0])),
        (False, pytest.approx(ACTION[0])),
        (False, pytest.approx(ACTION[0])),
    ]


def test_cleanup_locks_first_and_continues_after_an_error():
    events: list[object] = []
    executor = _executor(events=events)

    class Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def _close(self) -> None:
            events.append(self.name)
            if self.fail:
                raise RuntimeError(self.name)

        shutdown = _close
        close = _close

    with pytest.raises(BaseExceptionGroup) as caught:
        close_policy_resources(
            policy_label="test policy",
            executor=executor,
            timer=Resource("timer", fail=True),
            cameras=Resource("cameras"),
            client=Resource("client"),
        )

    assert events == [("enable", False), "timer", "cameras", "client"]
    assert len(caught.value.exceptions) == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("relay_enable_timeout_s", 0.0),
        ("settle_s", -0.1),
        ("tolerance_m", 0.0),
        ("corrections", -1),
    ],
)
def test_executor_rejects_invalid_safety_settings(name: str, value: float):
    settings = {
        "relay_enable_timeout_s": 0.2,
        "settle_s": 0.1,
        "tolerance_m": 0.005,
        "corrections": 2,
    }
    settings[name] = value

    with pytest.raises(ValueError, match="invalid|non-negative"):
        EefPolicyExecutor(
            state=FakeState(),
            relay=FakeRelay(),
            commander=FakeCommander([]),
            is_shutdown=lambda: False,
            policy_label="test policy",
            **settings,
        )


def test_executor_rejects_corrections_without_positive_settle_time():
    with pytest.raises(ValueError, match="corrections require a positive settle"):
        EefPolicyExecutor(
            state=FakeState(),
            relay=FakeRelay(),
            commander=FakeCommander([]),
            relay_enable_timeout_s=0.2,
            settle_s=0.0,
            tolerance_m=0.005,
            corrections=1,
            is_shutdown=lambda: False,
            policy_label="test policy",
        )
