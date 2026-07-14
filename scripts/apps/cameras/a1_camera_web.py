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
            config.front_camera.serial,
            config.front_camera.width,
            config.front_camera.height,
            config.front_camera.fps,
            warmup_frames=20,
            require_usb3=config.front_camera.require_usb3,
        )
        wrist = open_color_camera(
            config.wrist_camera.backend,
            serial=config.wrist_camera.serial,
            device=config.wrist_camera.device,
            width=config.wrist_camera.width,
            height=config.wrist_camera.height,
            fps=config.wrist_camera.fps,
            pixel_format=config.wrist_camera.pixel_format,
            warmup_frames=20,
        )
        front_reader = LatestCameraReader("front", front.read_frameset)
        wrist_reader = LatestCameraReader("wrist", wrist.read_bgr)
        front_reader.start()
        wrist_reader.start()
        preview_config = config.web_preview if config.web_preview.enabled else replace(config.web_preview, enabled=True)
        preview = CameraWebPreview(preview_config)
        preview.register_reader(
            "agent",
            front_reader,
            extract=color_from_frameset,
            source=front.label,
            overlay_roi=config.front_camera.crop,
            overlay_label=(
                f"RECORDED {config.front_camera.crop.width}x{config.front_camera.crop.height}"
                if config.front_camera.crop is not None
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
