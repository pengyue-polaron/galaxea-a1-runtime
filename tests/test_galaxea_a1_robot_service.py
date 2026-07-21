from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Mapping

import pytest
from lerobot_robot_galaxea_a1.runtime.contracts import (
    HealthReport,
    HealthStatus,
    RuntimeContractError as ContractError,
    RuntimeLifecycleError as LifecycleError,
)

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.apps.robot_service.server import build_server
from galaxea_a1_runtime.apps.robot_service.device import (
    A1RuntimeDevice,
    GRIPPER_FEATURE_KEY,
    JOINT_FEATURE_KEYS,
    _RosA1Session,
)

REPO = Path(__file__).resolve().parents[1]
SYSTEM_CONFIG = REPO / "configs/system/a1.toml"


class FakeSession:
    def __init__(self, observation: dict[str, float]) -> None:
        self.observation = observation
        self.is_connected = False
        self.commands: list[dict[str, object]] = []
        self.disconnect_count = 0
        self.command_acquire_count = 0
        self.command_release_count = 0
        self.command_lease_active = False

    def connect(self) -> None:
        self.is_connected = True

    def observe(self) -> Mapping[str, object]:
        return self.observation

    def acquire_command_lease(self) -> None:
        if self.command_lease_active:
            raise LifecycleError("fake command lease is already active")
        self.command_lease_active = True
        self.command_acquire_count += 1

    def release_command_lease(self) -> None:
        self.command_lease_active = False
        self.command_release_count += 1

    def command(self, action: Mapping[str, object]) -> Mapping[str, object]:
        if not self.command_lease_active:
            raise LifecycleError("fake command lease is not active")
        self.commands.append(dict(action))
        return action

    def health(self) -> HealthReport:
        return HealthReport(HealthStatus.HEALTHY, "fake runtime ready")

    def disconnect(self) -> None:
        self.is_connected = False
        self.disconnect_count += 1


@dataclass
class FakeRosResource:
    topic: str
    unregistered: bool = False

    def unregister(self) -> None:
        self.unregistered = True


@dataclass
class FakeRosSubscriber(FakeRosResource):
    callback: object = None


@dataclass
class FakeRosPublisher(FakeRosResource):
    messages: list[object] = field(default_factory=list)

    def publish(self, message: object) -> None:
        self.messages.append(message)


@dataclass
class FakeRosTimer:
    shutdown_called: bool = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeJointState:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None)
        self.name: list[str] = []
        self.position: list[float] = []


class FakeString:
    def __init__(self, *, data: str = "") -> None:
        self.data = data


class FakeBool:
    def __init__(self, *, data: bool = False) -> None:
        self.data = data


class FakeGripperCommand:
    def __init__(self) -> None:
        self.header = SimpleNamespace(stamp=None)
        self.gripper_stroke = 0.0


def _install_fake_ros(monkeypatch: pytest.MonkeyPatch, system) -> SimpleNamespace:
    state = SimpleNamespace(subscribers=[], publishers=[], timers=[])
    rospy = ModuleType("rospy")
    rospy.core = SimpleNamespace(is_initialized=lambda: True)
    rospy.is_shutdown = lambda: False
    rospy.Duration = lambda seconds: seconds
    rospy.Time = SimpleNamespace(now=lambda: 1.0)

    def subscriber(topic, _message_type, callback, *, queue_size):
        assert queue_size == 10
        resource = FakeRosSubscriber(topic=topic, callback=callback)
        state.subscribers.append(resource)
        if topic == system.topics.joint_states:
            message = FakeJointState()
            message.name = list(system.joint_safety.names)
            message.position = [0.0] * len(message.name)
            callback(message)
        elif topic == system.topics.gripper_feedback:
            message = FakeJointState()
            message.position = [0.0]
            callback(message)
        elif topic == system.topics.relay_status:
            callback(FakeString(data=json.dumps({"state": "ACTIVE"})))
        return resource

    def publisher(topic, _message_type, **_kwargs):
        resource = FakeRosPublisher(topic=topic)
        state.publishers.append(resource)
        return resource

    def timer(_duration, _callback):
        resource = FakeRosTimer()
        state.timers.append(resource)
        return resource

    rospy.Subscriber = subscriber
    rospy.Publisher = publisher
    rospy.Timer = timer

    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.JointState = FakeJointState
    sensor_msgs.msg = sensor_msgs_msg
    std_msgs = ModuleType("std_msgs")
    std_msgs_msg = ModuleType("std_msgs.msg")
    std_msgs_msg.Bool = FakeBool
    std_msgs_msg.String = FakeString
    std_msgs.msg = std_msgs_msg
    signal_arm = ModuleType("signal_arm")
    signal_arm_msg = ModuleType("signal_arm.msg")
    signal_arm_msg.arm_control = type("FakeArmControl", (), {})
    signal_arm_msg.gripper_position_control = FakeGripperCommand
    signal_arm.msg = signal_arm_msg
    for name, module in {
        "rospy": rospy,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "signal_arm": signal_arm,
        "signal_arm.msg": signal_arm_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "galaxea_a1_runtime.runtime.ros1_env.configure_ros1_python",
        lambda _root: None,
    )
    return state


def test_device_construction_is_static_and_session_uses_canonical_contract() -> None:
    sessions: list[FakeSession] = []

    def session_factory(system, timeout_s):
        assert system.path == SYSTEM_CONFIG
        assert timeout_s == 30.0
        observation = {
            **{name: 0.0 for name in JOINT_FEATURE_KEYS},
            GRIPPER_FEATURE_KEY: 0.25,
        }
        session = FakeSession(observation)
        sessions.append(session)
        return session

    system = load_system_config(SYSTEM_CONFIG, repo_root=REPO)
    device = A1RuntimeDevice(
        system=system,
        session_factory=session_factory,
    )

    assert sessions == []
    assert tuple(feature.name for feature in device.manifest.action_features) == (
        *JOINT_FEATURE_KEYS,
        GRIPPER_FEATURE_KEY,
    )
    device.connect()
    assert device.observe()[GRIPPER_FEATURE_KEY] == 0.25

    action = {
        **{
            name: (lower + upper) / 2
            for name, lower, upper in zip(
                JOINT_FEATURE_KEYS,
                device.system.joint_safety.lower_limits,
                device.system.joint_safety.upper_limits,
                strict=True,
            )
        },
        GRIPPER_FEATURE_KEY: 0.5,
    }
    with pytest.raises(LifecycleError, match="command lease is not active"):
        device.command(action)
    device.acquire_command_lease()
    assert device.command(action) == action
    assert sessions[0].commands == [action]
    device.release_command_lease()
    assert device.observe()[GRIPPER_FEATURE_KEY] == 0.25
    assert sessions[0].command_acquire_count == 1
    assert sessions[0].command_release_count == 1
    device.disconnect()
    assert sessions[0].disconnect_count == 1


def test_device_actions_fail_closed() -> None:
    session = FakeSession(
        {
            **{name: 0.0 for name in JOINT_FEATURE_KEYS},
            GRIPPER_FEATURE_KEY: 0.0,
        }
    )
    device = A1RuntimeDevice(
        system=load_system_config(SYSTEM_CONFIG, repo_root=REPO),
        session_factory=lambda _system, _timeout: session,
    )
    device.connect()
    valid = {
        **{name: 0.0 for name in JOINT_FEATURE_KEYS},
        GRIPPER_FEATURE_KEY: 0.0,
    }
    first = device.manifest.action_features[0]
    assert first.maximum is not None
    device.acquire_command_lease()
    with pytest.raises(ContractError, match="above maximum"):
        device.command({**valid, first.name: first.maximum + 0.01})
    assert session.commands == []
    device.disconnect()


def test_ros_session_separates_observation_and_command_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = load_system_config(SYSTEM_CONFIG, repo_root=REPO)
    ros = _install_fake_ros(monkeypatch, system)
    session = _RosA1Session(system, connect_timeout_s=0.1)

    session.connect()
    assert session.observe()[GRIPPER_FEATURE_KEY] == 0.0
    assert len(ros.subscribers) == 3
    assert ros.publishers == []
    with pytest.raises(LifecycleError, match="active/arming owner"):
        session.acquire_command_lease()
    assert ros.publishers == []

    relay_subscriber = next(
        subscriber
        for subscriber in ros.subscribers
        if subscriber.topic == system.topics.relay_status
    )
    relay_subscriber.callback(FakeString(data=json.dumps({"state": "LOCKED"})))
    session.acquire_command_lease()
    assert len(ros.subscribers) == 4
    assert len(ros.publishers) == 3
    assert len(ros.timers) == 1

    observation_resources = ros.subscribers[:3]
    command_resources = [ros.subscribers[3], *ros.publishers]
    session.release_command_lease()
    assert all(resource.unregistered for resource in command_resources)
    assert not any(resource.unregistered for resource in observation_resources)
    assert session.observe()[GRIPPER_FEATURE_KEY] == 0.0

    session.disconnect()
    assert all(resource.unregistered for resource in observation_resources)


def test_runtime_server_wiring_is_static_and_uses_system_transport_contract() -> None:
    system = load_system_config(SYSTEM_CONFIG, repo_root=REPO)
    captured: dict[str, object] = {}

    def server_factory(device, **kwargs):
        captured["device"] = device
        captured.update(kwargs)
        return object()

    server = build_server(
        system,
        session_factory=lambda _system, _timeout: pytest.fail(
            "server construction must not create a ROS session"
        ),
        server_factory=server_factory,
    )

    assert server is not None
    assert captured["endpoint"] == system.robot_service.endpoint
    assert captured["lease_timeout_s"] == system.robot_service.lease_timeout_s
    assert captured["command_timeout_s"] == system.robot_service.command_timeout_s
    assert captured["device"].is_connected is False
