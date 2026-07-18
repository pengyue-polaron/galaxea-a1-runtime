from __future__ import annotations

import numpy as np
import pytest

from galaxea_a1_runtime.apps.eef_policy_executor import (
    EefPolicyExecutor,
    close_policy_resources,
)


ACTION = np.asarray([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 0.5])
HOLD = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


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

    def hold_current_target(self) -> tuple[float, ...]:
        self.events.append("hold")
        return HOLD


class FakeStagedMonitor:
    def __init__(self, events: list[object], errors: list[float | None]) -> None:
        self.events = events
        self.errors = errors

    def max_error(self, target, dof: int, *, max_age_s: float):
        self.events.append(("staged", tuple(target), dof, max_age_s))
        if len(self.errors) > 1:
            return self.errors.pop(0)
        return self.errors[0]


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


def _executor(
    *,
    events: list[object],
    relay: FakeRelay | None = None,
    staged_errors: list[float | None] | None = None,
) -> EefPolicyExecutor:
    clock = FakeClock()
    return EefPolicyExecutor(
        relay=relay or FakeRelay(),
        commander=FakeCommander(events),
        staged_monitor=FakeStagedMonitor(events, staged_errors or [0.0]),
        relay_enable_timeout_s=0.2,
        staged_wait_timeout_s=0.1,
        staged_max_age_s=0.25,
        staged_alignment_tolerance_rad=0.05,
        is_shutdown=lambda: False,
        policy_label="test policy",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def test_executor_activates_current_hold_before_publishing_policy_action():
    events: list[object] = []
    executor = _executor(events=events)

    executor.activate_current_hold()
    command = executor.publish(ACTION)

    assert command == pytest.approx(ACTION)
    assert [
        (item[0], item[1])
        for item in events
        if isinstance(item, tuple) and item[0] != "staged"
    ] == [
        ("enable", True),
        ("action", False),
        ("gripper", pytest.approx(ACTION[7])),
    ]
    assert events[:4] == [
        "hold",
        "keepalive",
        ("staged", HOLD, len(HOLD), 0.25),
        ("enable", True),
    ]
    assert executor.motion_enabled is True


def test_executor_refuses_policy_action_before_current_hold_activation():
    events: list[object] = []
    executor = _executor(events=events)

    with pytest.raises(RuntimeError, match="current-joint hold is not ACTIVE"):
        executor.publish(ACTION)

    assert events == []


def test_executor_locks_after_relay_enable_timeout():
    events: list[object] = []
    executor = _executor(events=events, relay=FakeRelay(activates=False))

    with pytest.raises(RuntimeError, match="did not become ACTIVE"):
        executor.activate_current_hold()

    assert [
        (item[0], item[1])
        for item in events
        if isinstance(item, tuple) and item[0] != "staged"
    ] == [
        ("enable", True),
        ("enable", False),
    ]
    assert events[:4] == [
        "hold",
        "keepalive",
        ("staged", HOLD, len(HOLD), 0.25),
        ("enable", True),
    ]
    assert executor.motion_enabled is False


def test_executor_keeps_relay_locked_until_staged_hold_is_complete_and_aligned():
    events: list[object] = []
    executor = _executor(events=events, staged_errors=[None, 0.08, 0.01])

    executor.activate_current_hold()

    staged_events = [
        item for item in events if isinstance(item, tuple) and item[0] == "staged"
    ]
    assert len(staged_events) == 3
    assert events[:5] == ["hold", "keepalive", *staged_events]
    assert events[5] == ("enable", True)


def test_executor_never_enables_relay_when_staged_hold_is_missing():
    events: list[object] = []
    executor = _executor(events=events, staged_errors=[None])

    with pytest.raises(RuntimeError, match="no complete fresh staged command"):
        executor.activate_current_hold()

    assert ("enable", True) not in events
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


@pytest.mark.parametrize("value", [0.0, -0.1, float("inf")])
def test_executor_rejects_invalid_relay_timeout(value: float):
    with pytest.raises(ValueError, match="finite and positive"):
        EefPolicyExecutor(
            relay=FakeRelay(),
            commander=FakeCommander([]),
            staged_monitor=FakeStagedMonitor([], [0.0]),
            relay_enable_timeout_s=value,
            staged_wait_timeout_s=0.1,
            staged_max_age_s=0.25,
            staged_alignment_tolerance_rad=0.05,
            is_shutdown=lambda: False,
            policy_label="test policy",
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("staged_wait_timeout_s", 0.0),
        ("staged_max_age_s", 0.0),
        ("staged_alignment_tolerance_rad", -0.1),
        ("staged_wait_timeout_s", float("inf")),
    ],
)
def test_executor_rejects_invalid_staged_hold_settings(name: str, value: float):
    settings = {
        "staged_wait_timeout_s": 0.1,
        "staged_max_age_s": 0.25,
        "staged_alignment_tolerance_rad": 0.05,
    }
    settings[name] = value

    with pytest.raises(ValueError, match="staged-hold"):
        EefPolicyExecutor(
            relay=FakeRelay(),
            commander=FakeCommander([]),
            staged_monitor=FakeStagedMonitor([], [0.0]),
            relay_enable_timeout_s=0.2,
            is_shutdown=lambda: False,
            policy_label="test policy",
            **settings,
        )
