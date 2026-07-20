#!/usr/bin/env python3
"""Persistent config-driven camera owner for the shared LAN Web monitor."""

from __future__ import annotations

import signal
import threading
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from galaxea_a1_runtime.configuration.system import (
    SYSTEM_CONFIG,
    load_system_config,
)
from galaxea_a1_runtime.console import ArgumentParser, info, success
from galaxea_a1_runtime.hardware.camera_bridge import CameraBridgeServer
from galaxea_a1_runtime.hardware.cameras import (
    LatestCameraReader,
    RealSenseColorCamera,
    open_configured_camera,
    close_camera_resources,
)
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    color_from_bgr,
    color_from_frameset,
)

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Serve the tracked A1 agent/wrist cameras over LAN MJPEG."
    )
    parser.add_argument("--config", type=Path, default=ROOT / SYSTEM_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system = load_system_config(args.config, repo_root=ROOT)
    info(f"Config: {system.path}")
    front_config = system.cameras.front
    wrist_config = system.cameras.wrist
    front = None
    wrist = None
    front_reader = None
    wrist_reader = None
    bridge = None
    preview = None
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        opened_front = open_configured_camera(
            front_config,
            warmup_frames=system.cameras.warmup_frames,
            enable_depth=front_config.depth,
        )
        front = opened_front
        if not isinstance(front, RealSenseColorCamera):
            raise RuntimeError("Camera Bridge AgentView must be a RealSense camera")
        wrist = open_configured_camera(
            wrist_config,
            warmup_frames=system.cameras.warmup_frames,
            enable_depth=False,
        )
        front_reader = LatestCameraReader("front", front.read_frameset)
        wrist_reader = LatestCameraReader("wrist", wrist.read_bgr)
        front_reader.start()
        wrist_reader.start()
        bridge = CameraBridgeServer(
            system.cameras,
            front_reader=front_reader,
            wrist_reader=wrist_reader,
            front_source=front.label,
            wrist_source=wrist.label,
            front_usb_type=front.usb_type,
        )
        bridge.start()
        preview_config = (
            system.web_preview
            if system.web_preview.enabled
            else replace(system.web_preview, enabled=True)
        )
        preview = CameraWebPreview(
            preview_config,
            max_source_age_s=system.cameras.max_age_s,
        )
        preview.register_reader(
            "agent",
            front_reader,
            extract=color_from_frameset,
            source=front.label,
        )
        preview.register_reader(
            "wrist", wrist_reader, extract=color_from_bgr, source=wrist.label
        )
        preview.start()
        success("Persistent raw Camera Bridge and Web monitor are ready.")
        while not stop.wait(0.5):
            for reader in (front_reader, wrist_reader):
                error = reader.exception()
                if error is not None:
                    raise RuntimeError(f"{reader.name} camera reader failed") from error
            bridge_error = bridge.exception()
            if bridge_error is not None:
                raise RuntimeError("raw Camera Bridge failed") from bridge_error
    finally:
        cleanup_errors: list[BaseException] = []
        if preview is not None:
            try:
                preview.close()
            except BaseException as exc:  # Cleanup must continue to camera close.
                cleanup_errors.append(exc)
        if bridge is not None:
            try:
                bridge.close()
            except BaseException as exc:  # Cleanup must continue to camera close.
                cleanup_errors.append(exc)
        try:
            close_camera_resources(
                (wrist_reader, front_reader),
                (wrist, front),
            )
        except BaseException as exc:  # Report all cleanup failures together.
            cleanup_errors.append(exc)
        if cleanup_errors:
            raise BaseExceptionGroup("camera web cleanup failed", cleanup_errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
