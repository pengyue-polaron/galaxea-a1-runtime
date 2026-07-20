from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest
from embodied_ops import ContractError, HealthReport, HealthStatus

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.apps.embodied_ops.server import build_server
from galaxea_a1_runtime.embodied_ops_device import (
    A1RuntimeDevice,
    GRIPPER_FEATURE_KEY,
    JOINT_FEATURE_KEYS,
)

REPO = Path(__file__).resolve().parents[1]
SYSTEM_CONFIG = REPO / "configs/system/a1.toml"


class FakeSession:
    def __init__(self, observation: dict[str, float]) -> None:
        self.observation = observation
        self.is_connected = False
        self.commands: list[dict[str, object]] = []
        self.disconnect_count = 0

    def connect(self) -> None:
        self.is_connected = True

    def observe(self) -> Mapping[str, object]:
        return self.observation

    def command(self, action: Mapping[str, object]) -> Mapping[str, object]:
        self.commands.append(dict(action))
        return action

    def health(self) -> HealthReport:
        return HealthReport(HealthStatus.HEALTHY, "fake runtime ready")

    def disconnect(self) -> None:
        self.is_connected = False
        self.disconnect_count += 1


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
    assert device.command(action) == action
    assert sessions[0].commands == [action]
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
    with pytest.raises(ContractError, match="above maximum"):
        device.command({**valid, first.name: first.maximum + 0.01})
    assert session.commands == []
    device.disconnect()


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
    assert captured["endpoint"] == system.embodied_ops.endpoint
    assert captured["lease_timeout_s"] == system.embodied_ops.lease_timeout_s
    assert captured["device"].is_connected is False
