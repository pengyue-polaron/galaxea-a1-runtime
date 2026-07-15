"""Normalize model-native actions into the Galaxea A1 runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from galaxea_a1_runtime.safety import clamp_eef_delta
from galaxea_a1_runtime.schema import ActionMode, action_names_for_mode


@dataclass(frozen=True)
class RuntimeAction:
    mode: ActionMode
    values: tuple[float, ...]
    names: tuple[str, ...]
    source: str = "unknown"

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


def normalize_action(
    action: Mapping[str, float] | Sequence[float],
    *,
    mode: ActionMode,
    source: str = "unknown",
    max_translation: float | None = None,
    max_rotation: float | None = None,
) -> RuntimeAction:
    """Convert model output to a typed runtime action.

    EEF deltas are forwarded unchanged unless explicit limits are supplied.
    """

    names = action_names_for_mode(mode)
    values = _ordered_values(action, names)
    if len(values) != len(names):
        raise ValueError(
            f"action length mismatch for {mode}: {len(values)} != {len(names)}"
        )
    if not all(isfinite(value) for value in values):
        raise ValueError("action contains non-finite values")

    if mode in (ActionMode.EEF_DELTA, ActionMode.EEF_TRANSLATION):
        limits_requested = max_translation is not None or max_rotation is not None
        if limits_requested:
            if max_translation is None:
                raise ValueError("max_translation is required when EEF limits are used")
            values = clamp_eef_delta(
                values,
                max_translation=max_translation,
                max_rotation=max_rotation,
            )
        elif not 0.0 <= values[-1] <= 1.0:
            raise ValueError("gripper action is outside [0, 1]")
    elif mode == ActionMode.JOINT_ABSOLUTE:
        values = tuple(float(v) for v in values)
        if not 0.0 <= values[-1] <= 1.0:
            raise ValueError("gripper action is outside [0, 1]")
    else:
        raise ValueError(f"unsupported action mode: {mode}")

    return RuntimeAction(mode=mode, values=values, names=names, source=source)


def _ordered_values(
    action: Mapping[str, float] | Sequence[float],
    names: tuple[str, ...],
) -> tuple[float, ...]:
    if isinstance(action, Mapping):
        missing = [name for name in names if name not in action]
        if missing:
            raise ValueError(f"action mapping missing keys: {missing}")
        return tuple(float(action[name]) for name in names)
    return tuple(float(v) for v in action)
