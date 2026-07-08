#!/usr/bin/env python3
# ruff: noqa: E402
"""Capture front/wrist camera snapshots from the tracked teleop config."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture configured A1 teleop camera snapshots.")
    parser.add_argument("--config", type=Path, help="Teleop TOML config. Defaults to configs/teleop/a1_so100.toml")
    parser.add_argument("--out-dir", type=Path, help="Directory for captured images.")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parse_args()

import cv2
import numpy as np

from galaxea_a1_runtime.hardware.cameras import OpenCVColorCamera, RealSenseColorCamera
from galaxea_a1_runtime.teleop.config import default_config_path, load_teleop_config


def main() -> int:
    args = parse_args()
    config_path = args.config or default_config_path(ROOT_DIR)
    config = load_teleop_config(config_path, repo_root=ROOT_DIR)
    out_dir = args.out_dir or (
        ROOT_DIR
        / "data"
        / "diagnostics"
        / "camera_snapshots"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    front: RealSenseColorCamera | None = None
    wrist: OpenCVColorCamera | None = None
    try:
        front = RealSenseColorCamera(
            config.front_camera.serial,
            config.front_camera.width,
            config.front_camera.height,
            config.front_camera.fps,
            warmup_frames=args.warmup_frames,
        )
        wrist = OpenCVColorCamera(
            config.wrist_camera.device,
            config.wrist_camera.width,
            config.wrist_camera.height,
            config.wrist_camera.fps,
            warmup_frames=args.warmup_frames,
        )

        front_img = wait_frame(front, timeout_s=args.timeout_s, label="front")
        wrist_img = wait_frame(wrist, timeout_s=args.timeout_s, label="wrist")
        front_path = out_dir / "cam0_front.jpg"
        wrist_path = out_dir / "cam1_wrist.jpg"
        sheet_path = out_dir / "contact_sheet.jpg"
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
        cv2.imwrite(str(front_path), front_img, jpeg_params)
        cv2.imwrite(str(wrist_path), wrist_img, jpeg_params)
        cv2.imwrite(
            str(sheet_path),
            contact_sheet((("cam0 front", front_img), (f"cam1 wrist {wrist.label}", wrist_img))),
            jpeg_params,
        )
    finally:
        if wrist is not None:
            wrist.close()
        if front is not None:
            front.close()

    print(f"cam0_front={front_path}")
    print(f"cam1_wrist={wrist_path}")
    print(f"contact_sheet={sheet_path}")
    return 0


def wait_frame(camera, *, timeout_s: float, label: str) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = camera.read_bgr()
        if frame is not None:
            return frame
        time.sleep(0.03)
    raise RuntimeError(f"No frame from {label} camera within {timeout_s:.1f}s")


def contact_sheet(images: tuple[tuple[str, np.ndarray], ...]) -> np.ndarray:
    labelled = tuple(_label_image(label, image) for label, image in images)
    height = max(image.shape[0] for image in labelled)
    fitted = tuple(_fit_height(image, height) for image in labelled)
    return cv2.hconcat(fitted)


def _label_image(label: str, image: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    band_h = 34
    band = np.zeros((band_h, canvas.shape[1], 3), dtype=canvas.dtype)
    cv2.putText(
        band,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.vconcat((band, canvas))


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / float(image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    raise SystemExit(main())
