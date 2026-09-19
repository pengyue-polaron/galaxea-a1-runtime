"""A1 collection actions shared by terminal, Web, and Foxglove surfaces."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from embodied_ops.collection import (
    CollectionInteraction,
    CollectionResetPolicy,
    EpisodeDecision,
)
from embodied_ops.operator_panel import InputAction


class CollectionReadyAction(str, Enum):
    START = "start"
    RESET = "reset"
    ENABLE_RESET_AFTER_SAVE = "enable_reset_after_save"
    DISABLE_RESET_AFTER_SAVE = "disable_reset_after_save"
    QUIT = "quit"


class CollectionRecordingDecision(NamedTuple):
    decision: EpisodeDecision
    reset_required_override: bool | None = None


A1_COLLECTION_INTERACTION = CollectionInteraction(
    input_actions=(
        InputAction("start", "Start recording", "\n", "primary"),
        InputAction("save", "Stop & save", "\n", "primary"),
        InputAction(
            "save_without_reset",
            "Stop & save, keep position",
            "n\n",
            "primary",
        ),
        InputAction(
            "enable_reset_after_save",
            "Enable Reset after save",
            "o\n",
        ),
        InputAction(
            "disable_reset_after_save",
            "Disable Reset after save",
            "f\n",
        ),
        InputAction("discard", "Discard episode", "d\n", "danger"),
        InputAction("reset", "Reset position", "r\n", "danger"),
        InputAction("quit", "End session", "q\n", "quiet"),
    ),
    start_action_ids=(
        "start",
        "enable_reset_after_save",
        "disable_reset_after_save",
        "reset",
        "quit",
    ),
    recording_action_ids=("save", "save_without_reset", "discard", "quit"),
)


def normalize_collection_ready_action(text: str | None) -> CollectionReadyAction:
    """Accept only one explicit action at the pre-episode gate."""

    value = (text or "").strip().lower()
    if value in {"", "s", "start"}:
        return CollectionReadyAction.START
    if value in {"r", "reset"}:
        return CollectionReadyAction.RESET
    if value in {"o", "enable_reset_after_save"}:
        return CollectionReadyAction.ENABLE_RESET_AFTER_SAVE
    if value in {"f", "disable_reset_after_save"}:
        return CollectionReadyAction.DISABLE_RESET_AFTER_SAVE
    if value in {"q", "quit", "exit"}:
        return CollectionReadyAction.QUIT
    raise ValueError(f"unknown collection ready action: {text!r}")


def normalize_collection_recording_decision(
    text: str | None,
) -> CollectionRecordingDecision:
    """Map one episode decision and an explicit no-Reset Save override."""

    value = (text or "").strip().lower()
    if value in {"", "s", "save"}:
        return CollectionRecordingDecision(EpisodeDecision.SAVE)
    if value in {"y", "save_with_reset"}:
        return CollectionRecordingDecision(
            EpisodeDecision.SAVE,
            reset_required_override=True,
        )
    if value in {"n", "save_without_reset"}:
        return CollectionRecordingDecision(
            EpisodeDecision.SAVE,
            reset_required_override=False,
        )
    if value in {"d", "discard"}:
        return CollectionRecordingDecision(EpisodeDecision.DISCARD)
    if value in {"q", "quit", "exit"}:
        return CollectionRecordingDecision(EpisodeDecision.QUIT)
    raise ValueError(f"unknown collection recording decision: {text!r}")


def reset_required_after_recording(
    decision: EpisodeDecision,
    *,
    policy: CollectionResetPolicy,
    override: bool | None,
    reset_after_save: bool | None = None,
) -> bool:
    """Apply a Save-only operator override without changing discard policy."""

    if decision is EpisodeDecision.SAVE and override is not None:
        return override
    if decision is EpisodeDecision.SAVE and reset_after_save is not None:
        return reset_after_save
    return policy.required_after(decision)


def collection_ready_action_ids(*, reset_after_save: bool) -> tuple[str, ...]:
    toggle = (
        "disable_reset_after_save" if reset_after_save else "enable_reset_after_save"
    )
    return ("start", toggle, "reset", "quit")
