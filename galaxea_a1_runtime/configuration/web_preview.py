"""Pure configuration schema for the read-only camera web preview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    boolean,
    floating,
    integer,
    require_exact_keys,
    string,
)


@dataclass(frozen=True)
class WebPreviewConfig:
    enabled: bool
    bind: str
    port: int
    fps: float
    jpeg_quality: int
    startup_timeout_s: float = 15.0
    shutdown_timeout_s: float = 5.0

    def validate(self) -> None:
        if not self.bind:
            raise ValueError("web_preview.bind must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("web_preview.port must be in [1, 65535]")
        if self.fps <= 0:
            raise ValueError("web_preview.fps must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("web_preview.jpeg_quality must be in [1, 100]")
        if self.startup_timeout_s < 1:
            raise ValueError(
                "web_preview.startup_timeout_s must be at least one second"
            )
        if self.shutdown_timeout_s < 1:
            raise ValueError(
                "web_preview.shutdown_timeout_s must be at least one second"
            )


def parse_web_preview_config(
    data: dict[str, Any], *, repo_root: Path
) -> WebPreviewConfig:
    del repo_root
    require_exact_keys(
        data,
        required={
            "enabled",
            "bind",
            "port",
            "fps",
            "jpeg_quality",
            "startup_timeout_s",
            "shutdown_timeout_s",
        },
        label="web_preview",
    )
    config = WebPreviewConfig(
        enabled=boolean(data, "enabled"),
        bind=string(data, "bind"),
        port=integer(data, "port"),
        fps=floating(data, "fps"),
        jpeg_quality=integer(data, "jpeg_quality"),
        startup_timeout_s=floating(data, "startup_timeout_s"),
        shutdown_timeout_s=floating(data, "shutdown_timeout_s"),
    )
    config.validate()
    return config
