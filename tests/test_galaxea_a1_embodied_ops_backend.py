from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest
from embodied_ops import ContractError, HealthReport, HealthStatus

from galaxea_a1_runtime.embodied_ops_backend import (
    A1RuntimeBackend,
    GRIPPER_FEATURE_KEY,
    JOINT_FEATURE_KEYS,
    create_backend,
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


def test_backend_construction_is_static_and_session_uses_canonical_contract() -> None:
    sessions: list[FakeSession] = []

    def session_factory(system, timeout_s):
        assert system.path == SYSTEM_CONFIG
        assert timeout_s == 12.0
        observation = {
            **{name: 0.0 for name in JOINT_FEATURE_KEYS},
            GRIPPER_FEATURE_KEY: 0.25,
        }
        session = FakeSession(observation)
        sessions.append(session)
        return session

    backend = A1RuntimeBackend(
        system_config=SYSTEM_CONFIG,
        connect_timeout_s=12.0,
        session_factory=session_factory,
    )

    assert sessions == []
    assert tuple(feature.name for feature in backend.manifest.action_features) == (
        *JOINT_FEATURE_KEYS,
        GRIPPER_FEATURE_KEY,
    )
    backend.connect()
    assert backend.observe()[GRIPPER_FEATURE_KEY] == 0.25

    action = {
        **{
            name: (lower + upper) / 2
            for name, lower, upper in zip(
                JOINT_FEATURE_KEYS,
                backend.system.joint_safety.lower_limits,
                backend.system.joint_safety.upper_limits,
                strict=True,
            )
        },
        GRIPPER_FEATURE_KEY: 0.5,
    }
    assert backend.command(action) == action
    assert sessions[0].commands == [action]
    backend.disconnect()
    assert sessions[0].disconnect_count == 1


def test_backend_factory_and_actions_fail_closed() -> None:
    with pytest.raises(ContractError, match="unknown"):
        create_backend(
            {
                "system_config": str(SYSTEM_CONFIG),
                "connect_timeout_s": 10.0,
                "host_command_topic": "/unsafe",
            }
        )

    session = FakeSession(
        {
            **{name: 0.0 for name in JOINT_FEATURE_KEYS},
            GRIPPER_FEATURE_KEY: 0.0,
        }
    )
    backend = A1RuntimeBackend(
        system_config=SYSTEM_CONFIG,
        connect_timeout_s=10.0,
        session_factory=lambda _system, _timeout: session,
    )
    backend.connect()
    valid = {
        **{name: 0.0 for name in JOINT_FEATURE_KEYS},
        GRIPPER_FEATURE_KEY: 0.0,
    }
    first = backend.manifest.action_features[0]
    assert first.maximum is not None
    with pytest.raises(ContractError, match="above maximum"):
        backend.command({**valid, first.name: first.maximum + 0.01})
    assert session.commands == []
    backend.disconnect()
