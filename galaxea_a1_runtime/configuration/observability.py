"""Strict configuration for ROS telemetry and scoped Foxglove access."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import AddressValueError, IPv4Address
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.base import (
    boolean,
    floating,
    integer,
    require_exact_keys,
    required_table,
    string,
)


@dataclass(frozen=True)
class ObservabilityTopicsConfig:
    front_image: str
    wrist_image: str
    staged_joint_state: str
    host_joint_state: str
    gripper_feedback_state: str
    gripper_target_state: str
    gripper_command_state: str
    workflow_status: str
    diagnostics: str


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool
    bind: str
    port: int
    image_rate_hz: float
    jpeg_quality: int
    diagnostics_rate_hz: float
    operator_panel_poll_rate_hz: float
    operator_session_timeout_s: float
    camera_connect_timeout_s: float
    camera_retry_s: float
    startup_timeout_s: float
    shutdown_timeout_s: float
    graph_update_ms: int
    send_buffer_limit_bytes: int
    topics: ObservabilityTopicsConfig

    def validate(self) -> None:
        try:
            IPv4Address(self.bind)
        except AddressValueError as exc:
            raise ValueError("observability.bind must be an IPv4 address") from exc
        if not 1 <= self.port <= 65535:
            raise ValueError("observability.port must be in [1, 65535]")
        if (
            min(
                self.image_rate_hz,
                self.diagnostics_rate_hz,
                self.operator_panel_poll_rate_hz,
                self.operator_session_timeout_s,
                self.camera_connect_timeout_s,
                self.camera_retry_s,
                self.startup_timeout_s,
                self.shutdown_timeout_s,
            )
            <= 0
        ):
            raise ValueError("observability rates and timeouts must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("observability.jpeg_quality must be in [1, 100]")
        if self.graph_update_ms <= 0:
            raise ValueError("observability.graph_update_ms must be positive")
        if self.send_buffer_limit_bytes < 1_000_000:
            raise ValueError(
                "observability.send_buffer_limit_bytes must be at least 1000000"
            )


def parse_observability_config(
    data: dict[str, Any], *, repo_root: Path
) -> ObservabilityConfig:
    del repo_root
    require_exact_keys(
        data,
        required={
            "enabled",
            "bind",
            "port",
            "image_rate_hz",
            "jpeg_quality",
            "diagnostics_rate_hz",
            "operator_panel_poll_rate_hz",
            "operator_session_timeout_s",
            "camera_connect_timeout_s",
            "camera_retry_s",
            "startup_timeout_s",
            "shutdown_timeout_s",
            "graph_update_ms",
            "send_buffer_limit_bytes",
            "topics",
        },
        label="observability",
    )
    topics = required_table(data, "topics")
    require_exact_keys(
        topics,
        required=set(ObservabilityTopicsConfig.__annotations__),
        label="observability.topics",
    )
    config = ObservabilityConfig(
        enabled=boolean(data, "enabled"),
        bind=string(data, "bind"),
        port=integer(data, "port"),
        image_rate_hz=floating(data, "image_rate_hz"),
        jpeg_quality=integer(data, "jpeg_quality"),
        diagnostics_rate_hz=floating(data, "diagnostics_rate_hz"),
        operator_panel_poll_rate_hz=floating(data, "operator_panel_poll_rate_hz"),
        operator_session_timeout_s=floating(data, "operator_session_timeout_s"),
        camera_connect_timeout_s=floating(data, "camera_connect_timeout_s"),
        camera_retry_s=floating(data, "camera_retry_s"),
        startup_timeout_s=floating(data, "startup_timeout_s"),
        shutdown_timeout_s=floating(data, "shutdown_timeout_s"),
        graph_update_ms=integer(data, "graph_update_ms"),
        send_buffer_limit_bytes=integer(data, "send_buffer_limit_bytes"),
        topics=ObservabilityTopicsConfig(
            **{
                name: string(topics, name)
                for name in ObservabilityTopicsConfig.__annotations__
            }
        ),
    )
    config.validate()
    return config
