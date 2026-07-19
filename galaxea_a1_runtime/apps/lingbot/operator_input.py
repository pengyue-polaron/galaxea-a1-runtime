"""Operator-entered scene notes for attributable LingBot runs."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from galaxea_a1_runtime.console import ArgumentParser


MAX_SCENE_NOTE_LENGTH = 120


class SceneNoteCancelled(RuntimeError):
    pass


def validate_scene_note(value: str) -> str:
    note = value.strip()
    if not note:
        raise ValueError("scene note must not be empty")
    if "\n" in note or "\r" in note:
        raise ValueError("scene note must be a single line")
    if len(note) > MAX_SCENE_NOTE_LENGTH:
        raise ValueError(f"scene note exceeds {MAX_SCENE_NOTE_LENGTH} characters")
    return note


def prompt_scene_note(
    *,
    input_fn: Callable[[], str] = input,
    output: TextIO = sys.stderr,
) -> str:
    while True:
        output.write("Scene note (q=cancel) > ")
        output.flush()
        try:
            raw = input_fn()
        except EOFError as exc:
            raise SceneNoteCancelled("scene note received EOF") from exc
        if raw.strip().lower() in {"q", "quit", "exit"}:
            raise SceneNoteCancelled("scene note cancelled")
        try:
            return validate_scene_note(raw)
        except ValueError as exc:
            print(f"[FAIL] {exc}.", file=output)


def main(argv: list[str] | None = None) -> int:
    ArgumentParser(description=__doc__).parse_args(argv)
    try:
        note = prompt_scene_note()
    except SceneNoteCancelled as exc:
        print(f"[INFO] {exc}.", file=sys.stderr)
        return 2
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
