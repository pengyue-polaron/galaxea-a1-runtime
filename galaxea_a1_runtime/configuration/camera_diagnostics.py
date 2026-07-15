"""Tracked settings for the camera snapshot diagnostic command."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    floating,
    integer,
    repo_path,
    require_exact_keys,
    string,
)


@dataclass(frozen=True)
class CameraDiagnosticsConfig:
    output_root: Path
    frame_timeout_s: float
    rate_probe_s: float
    jpeg_quality: int


def parse_camera_diagnostics_config(
    data: dict[str, Any], *, repo_root: Path
) -> CameraDiagnosticsConfig:
    require_exact_keys(
        data,
        required={
            "output_root",
            "frame_timeout_s",
            "rate_probe_s",
            "jpeg_quality",
        },
        label="camera_diagnostics",
    )
    output_root = repo_path(repo_root, string(data, "output_root"))
    if not output_root.is_relative_to(repo_root):
        raise ValueError("camera_diagnostics.output_root must stay inside the repo")
    config = CameraDiagnosticsConfig(
        output_root=output_root,
        frame_timeout_s=floating(data, "frame_timeout_s"),
        rate_probe_s=floating(data, "rate_probe_s"),
        jpeg_quality=integer(data, "jpeg_quality"),
    )
    if config.frame_timeout_s <= 0:
        raise ValueError("camera_diagnostics.frame_timeout_s must be positive")
    if config.rate_probe_s < 0:
        raise ValueError("camera_diagnostics.rate_probe_s must be non-negative")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("camera_diagnostics.jpeg_quality must be in [1, 100]")
    return config
