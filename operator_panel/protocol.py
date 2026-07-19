"""Machine-readable child-process input readiness protocol."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable


PROTOCOL_ENV = "OPERATOR_PANEL_PROTOCOL"
PROTOCOL_PREFIX = "@@OPERATOR_PANEL "


def announce_input(actions: Iterable[str]) -> None:
    """Tell a supervising panel which input actions are currently safe."""

    if os.environ.get(PROTOCOL_ENV) != "1":
        return
    normalized = _normalize_actions(actions)
    print(
        PROTOCOL_PREFIX + json.dumps({"input": normalized}, separators=(",", ":")),
        flush=True,
    )


def parse_input_event(line: str) -> tuple[str, ...] | None:
    if not line.startswith(PROTOCOL_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROTOCOL_PREFIX) :])
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, dict) or not isinstance(payload.get("input"), list):
        return ()
    try:
        return _normalize_actions(payload["input"])
    except ValueError:
        return ()


def _normalize_actions(actions: Iterable[str]) -> tuple[str, ...]:
    values = tuple(actions)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("operator-panel input actions must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("operator-panel input actions must not contain duplicates")
    return values
