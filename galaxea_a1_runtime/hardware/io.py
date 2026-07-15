"""Hardware IO protocol for Galaxea A1 runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from galaxea_a1_runtime.policies.actions import RuntimeAction
from galaxea_a1_runtime.schema import DEFAULT_STATE_NAMES


@dataclass(frozen=True)
class A1Observation:
    state: tuple[float, ...]
    images: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None

    def validate(self) -> None:
        if len(self.state) != len(DEFAULT_STATE_NAMES):
            raise ValueError(
                f"A1 state has {len(self.state)} values, need {len(DEFAULT_STATE_NAMES)}"
            )


class A1HardwareIO(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> A1Observation: ...

    def send_runtime_action(self, action: RuntimeAction) -> RuntimeAction: ...


class InMemoryA1HardwareIO:
    """In-memory IO adapter for static tests and dry runs."""

    def __init__(self, observation: A1Observation | None = None):
        self._connected = False
        self._observation = observation or A1Observation(
            state=(0.0,) * len(DEFAULT_STATE_NAMES)
        )
        self.last_action: RuntimeAction | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> A1Observation:
        if not self._connected:
            raise RuntimeError("A1 IO is not connected")
        self._observation.validate()
        return self._observation

    def send_runtime_action(self, action: RuntimeAction) -> RuntimeAction:
        if not self._connected:
            raise RuntimeError("A1 IO is not connected")
        self.last_action = action
        return action
