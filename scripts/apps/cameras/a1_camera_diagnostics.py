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
    parser.add_argument("--probe-s", type=float, default=2.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    parse_args()

import cv2
import numpy as np

from galaxea_a1_runtime.hardware.cameras import (
    ColorCamera,
    LatestCameraReader,
    RealSenseColorCamera,
    RealSenseFrameSet,
    open_color_camera,
)
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
    wrist: ColorCamera | None = None
    try:
        front = RealSenseColorCamera(
            config.front_camera.serial,
            config.front_camera.width,
            config.front_camera.height,
            config.front_camera.fps,
            enable_depth=config.front_camera.depth,
            depth_width=config.front_camera.depth_width,
            depth_height=config.front_camera.depth_height,
            align_depth_to_color=config.front_camera.align_depth_to_color,
            warmup_frames=args.warmup_frames,
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
            warmup_frames=args.warmup_frames,
        )

        rates = probe_camera_rates(
            front,
            wrist,
            duration_s=args.probe_s,
            front_target_fps=config.front_camera.fps,
            wrist_target_fps=config.wrist_camera.fps,
        )
        front_frameset = wait_realsense_frameset(front, timeout_s=args.timeout_s, label="front")
        front_img = front_frameset.color_bgr
        wrist_img = wait_frame(wrist, timeout_s=args.timeout_s, label="wrist")
        front_path = out_dir / "cam0_front.jpg"
        wrist_path = out_dir / "cam1_wrist.jpg"
        sheet_path = out_dir / "contact_sheet.jpg"
        depth_path: Path | None = None
        depth_preview_path: Path | None = None
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
        cv2.imwrite(str(front_path), front_img, jpeg_params)
        cv2.imwrite(str(wrist_path), wrist_img, jpeg_params)
        sheet_images: list[tuple[str, np.ndarray]] = [
            ("cam0 front", front_img),
            (f"cam1 wrist {wrist.label}", wrist_img),
        ]
        if config.front_camera.depth:
            if front_frameset.depth_mm is None:
                raise RuntimeError("RealSense depth is enabled but no depth frame was captured")
            depth_path = out_dir / "cam0_depth.png"
            depth_preview_path = out_dir / "cam0_depth_preview.jpg"
            preview = depth_preview(front_frameset.depth_mm)
            cv2.imwrite(str(depth_path), front_frameset.depth_mm)
            cv2.imwrite(str(depth_preview_path), preview, jpeg_params)
            sheet_images.append(("cam0 depth", preview))
        cv2.imwrite(
            str(sheet_path),
            contact_sheet(tuple(sheet_images)),
            jpeg_params,
        )
    finally:
        if wrist is not None:
            wrist.close()
        if front is not None:
            front.close()

    print(f"cam0_usb={front.usb_type if front is not None else 'unknown'}")
    if args.probe_s > 0:
        print(f"cam0_front_fps={rates['front']:.2f}")
        print(f"cam1_wrist_fps={rates['wrist']:.2f}")
    print(f"cam0_front={front_path}")
    if depth_path is not None:
        print(f"cam0_depth={depth_path}")
    if depth_preview_path is not None:
        print(f"cam0_depth_preview={depth_preview_path}")
    print(f"cam1_wrist={wrist_path}")
    print(f"contact_sheet={sheet_path}")
    return 0


def probe_camera_rates(
    front: RealSenseColorCamera,
    wrist: ColorCamera,
    *,
    duration_s: float,
    front_target_fps: float,
    wrist_target_fps: float,
) -> dict[str, float]:
    if duration_s <= 0:
        return {"front": 0.0, "wrist": 0.0}
    readers = (
        LatestCameraReader("front", front.read_frameset),
        LatestCameraReader("wrist", wrist.read_bgr),
    )
    for reader in readers:
        reader.start()
    start = time.perf_counter()
    time.sleep(duration_s)
    elapsed = max(time.perf_counter() - start, 1e-6)
    for reader in readers:
        reader.stop()
        exc = reader.exception()
        if exc is not None:
            raise RuntimeError(f"{reader.name} camera probe failed") from exc
    rates = {reader.name: reader.frame_count() / elapsed for reader in readers}
    too_slow = []
    for name, target_fps in (("front", front_target_fps), ("wrist", wrist_target_fps)):
        if rates[name] < 0.8 * target_fps:
            too_slow.append(f"{name}={rates[name]:.1f}fps target={target_fps:g}fps")
    if too_slow:
        raise RuntimeError("Camera rate probe failed: " + ", ".join(too_slow))
    return rates


def wait_realsense_frameset(camera: RealSenseColorCamera, *, timeout_s: float, label: str) -> RealSenseFrameSet:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frameset = camera.read_frameset()
        if frameset is not None:
            return frameset
        time.sleep(0.03)
    raise RuntimeError(f"No frame from {label} camera within {timeout_s:.1f}s")


def wait_frame(camera, *, timeout_s: float, label: str) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = camera.read_bgr()
        if frame is not None:
            return frame
        time.sleep(0.03)
    raise RuntimeError(f"No frame from {label} camera within {timeout_s:.1f}s")


def depth_preview(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((*depth_mm.shape[:2], 3), dtype=np.uint8)
    near, far = np.percentile(valid, (2, 98))
    if far <= near:
        far = near + 1
    normalized = np.clip((depth_mm.astype(np.float32) - near) * (255.0 / (far - near)), 0, 255)
    preview = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[depth_mm == 0] = (0, 0, 0)
    return preview


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
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
