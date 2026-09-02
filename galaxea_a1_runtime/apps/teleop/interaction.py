"""A1 collection actions shared by terminal, Web, and Foxglove surfaces."""

from __future__ import annotations

from enum import Enum

from embodied_ops.collection import CollectionInteraction
from embodied_ops.operator_panel import InputAction


class CollectionReadyAction(str, Enum):
    START = "start"
    RESET = "reset"
    QUIT = "quit"


A1_COLLECTION_INTERACTION = CollectionInteraction(
    input_actions=(
        InputAction("start", "Start recording", "\n", "primary"),
        InputAction("save", "Stop & save", "\n", "primary"),
        InputAction("discard", "Discard episode", "d\n", "danger"),
        InputAction("reset", "Reset position", "r\n", "danger"),
        InputAction("quit", "End session", "q\n", "quiet"),
    ),
    start_action_ids=("start", "reset", "quit"),
    recording_action_ids=("save", "discard", "quit"),
)


def normalize_collection_ready_action(text: str | None) -> CollectionReadyAction:
    """Accept only one explicit action at the pre-episode gate."""

    value = (text or "").strip().lower()
    if value in {"", "s", "start"}:
        return CollectionReadyAction.START
    if value in {"r", "reset"}:
        return CollectionReadyAction.RESET
    if value in {"q", "quit", "exit"}:
        return CollectionReadyAction.QUIT
    raise ValueError(f"unknown collection ready action: {text!r}")
