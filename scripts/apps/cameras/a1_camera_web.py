#!/usr/bin/env python3
"""Standalone dual-RealSense owner for the shared LAN web preview."""

from __future__ import annotations

import argparse
import signal
import threading
from dataclasses import replace
from pathlib import Path

from galaxea_a1_runtime.hardware.cameras import LatestCameraReader, RealSenseColorCamera, open_color_camera
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    color_from_bgr,
    color_from_frameset,
)
from galaxea_a1_runtime.teleop.config import default_config_path, load_teleop_config


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the tracked A1 agent/wrist cameras over LAN MJPEG.")
    parser.add_argument("--config", type=Path, default=default_config_path(ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_teleop_config(args.config, repo_root=ROOT)
    front_config = config.system.cameras.front
    wrist_config = config.system.cameras.wrist
    front = None
    wrist = None
    front_reader = None
    wrist_reader = None
    preview = None
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        front = RealSenseColorCamera(
            front_config.serial,
            front_config.width,
            front_config.height,
            front_config.fps,
            warmup_frames=20,
            require_usb3=front_config.require_usb3,
        )
        wrist = open_color_camera(
            wrist_config.backend,
            serial=wrist_config.serial,
            device=wrist_config.device,
            width=wrist_config.width,
            height=wrist_config.height,
            fps=wrist_config.fps,
            pixel_format=wrist_config.pixel_format,
            warmup_frames=20,
        )
        front_reader = LatestCameraReader("front", front.read_frameset)
        wrist_reader = LatestCameraReader("wrist", wrist.read_bgr)
        front_reader.start()
        wrist_reader.start()
        preview_config = (
            config.system.web_preview
            if config.system.web_preview.enabled
            else replace(config.system.web_preview, enabled=True)
        )
        preview = CameraWebPreview(preview_config)
        preview.register_reader(
            "agent",
            front_reader,
            extract=color_from_frameset,
            source=front.label,
            overlay_roi=front_config.crop,
            overlay_label=(
                f"RECORDED {front_config.crop.width}x{front_config.crop.height}"
                if front_config.crop is not None
                else ""
            ),
        )
        preview.register_reader("wrist", wrist_reader, extract=color_from_bgr, source=wrist.label)
        preview.start()
        print("[Camera Web] standalone camera owner is ready; Ctrl+C to stop", flush=True)
        while not stop.wait(0.5):
            for reader in (front_reader, wrist_reader):
                error = reader.exception()
                if error is not None:
                    raise RuntimeError(f"{reader.name} camera reader failed") from error
    finally:
        if preview is not None:
            preview.close()
        for reader in (wrist_reader, front_reader):
            if reader is not None:
                reader.stop()
        for camera in (wrist, front):
            if camera is not None:
                camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
