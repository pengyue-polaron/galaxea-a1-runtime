"""Shared terminal presentation for operator-facing A1 commands.

The module deliberately stays dependency-free so static doctors and hardware
entrypoints can use the same semantic colors without importing ROS or Rich.
"""

from __future__ import annotations

import argparse
import os
import sys
from enum import StrEnum
from typing import TextIO


class Tone(StrEnum):
    INFO = "1;34"
    STEP = "1;36"
    SUCCESS = "1;32"
    WARNING = "1;33"
    FAILURE = "1;31"
    MUTED = "2"


LEVEL_TONES = {
    "INFO": Tone.INFO,
    "STEP": Tone.STEP,
    "PASS": Tone.SUCCESS,
    "WARN": Tone.WARNING,
    "FAIL": Tone.FAILURE,
}


def color_enabled(stream: TextIO = sys.stdout) -> bool:
    """Honor the NO_COLOR convention and never color redirected output."""

    return not os.environ.get("NO_COLOR") and stream.isatty()


def style(text: str, tone: Tone, *, stream: TextIO = sys.stdout) -> str:
    if not color_enabled(stream):
        return text
    return f"\033[{tone.value}m{text}\033[0m"


def label(level: str, *, stream: TextIO = sys.stdout) -> str:
    normalized = level.upper()
    tone = LEVEL_TONES.get(normalized, Tone.INFO)
    return style(f"[{normalized}]", tone, stream=stream)


def padded_label(level: str, *, width: int = 6, stream: TextIO = sys.stdout) -> str:
    normalized = level.upper()
    tone = LEVEL_TONES.get(normalized, Tone.INFO)
    return style(f"[{normalized}]".ljust(width), tone, stream=stream)


def emit(
    level: str,
    message: str,
    *,
    stream: TextIO = sys.stdout,
    flush: bool = True,
) -> None:
    print(f"{label(level, stream=stream)} {message}", file=stream, flush=flush)


def info(message: str, *, flush: bool = True) -> None:
    emit("INFO", message, flush=flush)


def step(message: str, *, flush: bool = True) -> None:
    emit("STEP", message, flush=flush)


def success(message: str, *, flush: bool = True) -> None:
    emit("PASS", message, flush=flush)


def warning(message: str, *, flush: bool = True) -> None:
    emit("WARN", message, stream=sys.stderr, flush=flush)


def failure(message: str, *, flush: bool = True) -> None:
    emit("FAIL", message, stream=sys.stderr, flush=flush)


class ArgumentParser(argparse.ArgumentParser):
    """Argparse with the same failure presentation as the rest of the CLI."""

    def print_usage(self, file: TextIO | None = None) -> None:
        stream = file or sys.stdout
        print(_styled_help(self.format_usage(), stream), end="", file=stream)

    def print_help(self, file: TextIO | None = None) -> None:
        stream = file or sys.stdout
        print(_styled_help(self.format_help(), stream), end="", file=stream)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        failure(message)
        raise SystemExit(2)


def _styled_help(value: str, stream: TextIO) -> str:
    lines: list[str] = []
    for line in value.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        suffix = line[len(stripped) :]
        if stripped.startswith("usage:"):
            stripped = style("usage:", Tone.INFO, stream=stream) + stripped[6:]
        elif stripped and not stripped.startswith(" ") and stripped.endswith(":"):
            stripped = style(stripped, Tone.INFO, stream=stream)
        lines.append(stripped + suffix)
    return "".join(lines)
