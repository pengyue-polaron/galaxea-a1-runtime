"""Hardware-agnostic LeRobotDataset episode recorder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from galaxea_a1_runtime.hardware.io import A1HardwareIO, A1Observation
from galaxea_a1_runtime.policies.actions import RuntimeAction

from .writer import LeRobotV3DatasetWriter, build_lerobot_frame


@dataclass(frozen=True)
class RecordedStep:
    observation: A1Observation
    action: RuntimeAction
    frame: dict[str, Any]
    executed: bool


class LeRobotEpisodeRecorder:
    """Compose hardware IO and a LeRobotDataset writer.

    The recorder is intentionally passive by default. `record_step` writes the
    current observation paired with a supplied normalized action. It only sends
    the action to hardware when `execute=True` is passed explicitly.
    """

    def __init__(
        self,
        *,
        io: A1HardwareIO,
        writer: LeRobotV3DatasetWriter,
        task: str,
        connect_io: bool = True,
    ):
        if not task:
            raise ValueError("task must not be empty")
        self.io = io
        self.writer = writer
        self.task = task
        self.connect_io = connect_io
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        if self._open:
            return
        if self.connect_io and not self.io.is_connected:
            self.io.connect()
        self.writer.open()
        self._open = True

    def record_step(
        self,
        action: RuntimeAction,
        *,
        timestamp: float | None = None,
        execute: bool = False,
    ) -> RecordedStep:
        if not self._open:
            raise RuntimeError("episode recorder is not open")
        observation = self.io.get_observation()
        build_lerobot_frame(
            observation=observation,
            action=action,
            task=self.task,
            contract=self.writer.contract,
            timestamp=timestamp,
        )
        sent_action = self.io.send_runtime_action(action) if execute else action
        frame = self.writer.add_frame(
            observation=observation,
            action=sent_action,
            task=self.task,
            timestamp=timestamp,
        )
        return RecordedStep(
            observation=observation,
            action=sent_action,
            frame=frame,
            executed=execute,
        )

    def save_episode(self) -> None:
        if not self._open:
            raise RuntimeError("episode recorder is not open")
        self.writer.save_episode()

    def close(self, *, disconnect_io: bool = False) -> None:
        if not self._open:
            return
        self.writer.finalize()
        if disconnect_io and self.io.is_connected:
            self.io.disconnect()
        self._open = False
