"""Pure quality checks for recorded teleoperation episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ActionStepViolation:
    frame_index: int
    joint_name: str
    previous: float
    current: float
    step_rad: float
    limit_rad: float

    def describe(self) -> str:
        return (
            f"frame {self.frame_index - 1}->{self.frame_index} {self.joint_name}: "
            f"{self.previous:.6f}->{self.current:.6f} rad "
            f"(step={self.step_rad:+.6f}, limit={self.limit_rad:.6f})"
        )


def find_joint_action_step_violation(
    samples: Iterable[Sequence[float]],
    *,
    action_names: Sequence[str],
    max_step_rad: float,
) -> ActionStepViolation | None:
    if max_step_rad <= 0:
        raise ValueError("max_step_rad must be positive")
    joint_indices = tuple(
        index for index, name in enumerate(action_names) if name.startswith("joint_")
    )
    if not joint_indices:
        raise ValueError("action_names must contain at least one joint_* action")

    previous: tuple[float, ...] | None = None
    for frame_index, sample in enumerate(samples):
        current = tuple(float(value) for value in sample)
        if len(current) != len(action_names):
            raise ValueError(
                f"action sample {frame_index} has {len(current)} values, expected {len(action_names)}"
            )
        if previous is not None:
            for index in joint_indices:
                step = current[index] - previous[index]
                if abs(step) > max_step_rad:
                    return ActionStepViolation(
                        frame_index=frame_index,
                        joint_name=action_names[index],
                        previous=previous[index],
                        current=current[index],
                        step_rad=step,
                        limit_rad=max_step_rad,
                    )
        previous = current
    return None
