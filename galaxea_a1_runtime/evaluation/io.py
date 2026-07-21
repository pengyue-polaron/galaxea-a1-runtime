"""Validated input and immutable output helpers for offline evaluations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from embodied_ops.artifacts import (
    create_only_output_file,
    write_json_once,
    write_text_once,
)

from galaxea_a1_runtime.evaluation.offline_config import OfflineEvalConfig


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def evaluation_run_dir(config: OfflineEvalConfig, run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"invalid offline evaluation run id: {run_id!r}")
    path = config.output_root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_object(path: Path, value: dict[str, Any]) -> None:
    write_json_once(path, value)


def write_text_new(path: Path, value: str) -> None:
    write_text_once(path, value)


def write_contact_sheet(path: Path, model: str, visuals) -> None:
    from PIL import Image, ImageDraw

    if not visuals:
        return
    row_height = 220
    canvas = Image.new("RGB", (1280, 55 + row_height * len(visuals)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 16),
        f"{model} real first-frame predictions vs demonstrations",
        fill="black",
    )
    for row, (task, front, wrist, prediction, target) in enumerate(visuals):
        y = 55 + row * row_height
        front_image = Image.fromarray(front).resize((200, 200))
        wrist_image = Image.fromarray(wrist).resize((267, 200))
        canvas.paste(front_image, (10, y))
        canvas.paste(wrist_image, (220, y))
        error_cm = float(np.linalg.norm(prediction[:3] - target[:3]) * 100.0)
        draw.multiline_text(
            (510, y + 10),
            "\n".join(
                [
                    task,
                    f"pred xyz={np.round(prediction[:3], 4).tolist()} grip={prediction[7]:.3f}",
                    f"demo xyz={np.round(target[:3], 4).tolist()} grip={target[7]:.3f}",
                    f"first-target xyz error={error_cm:.2f} cm",
                ]
            ),
            fill="black",
            spacing=8,
        )
    with create_only_output_file(path) as temporary:
        canvas.save(temporary, format="PNG")
