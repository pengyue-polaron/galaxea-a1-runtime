"""Capture diagnostic snapshots from the tracked A1 system cameras."""

from __future__ import annotations

import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.console import ArgumentParser, failure

if TYPE_CHECKING:
    import numpy as np

    from galaxea_a1_runtime.configuration.system import SystemConfig
    from galaxea_a1_runtime.hardware.cameras import CameraSample, LatestCameraReader


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description="Capture snapshots from the tracked A1 system cameras."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / SYSTEM_CONFIG,
        help="Tracked physical system TOML config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from galaxea_a1_runtime.configuration.system import load_system_config

    system = load_system_config(args.config, repo_root=REPO_ROOT)
    return capture_diagnostics(system)


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except RuntimeError as exc:
        failure(str(exc))
        return 1


def capture_diagnostics(system: SystemConfig) -> int:
    import cv2
    import numpy as np

    from galaxea_a1_runtime.configuration.system import SystemRealSenseCameraConfig
    from galaxea_a1_runtime.hardware.cameras import (
        ColorCamera,
        LatestCameraReader,
        RealSenseColorCamera,
        RealSenseFrameSet,
        close_camera_resources,
        open_configured_camera,
    )

    front_config = system.cameras.front
    wrist_config = system.cameras.wrist
    diagnostic = system.camera_diagnostics
    out_dir = diagnostic.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    front: ColorCamera | None = None
    wrist: ColorCamera | None = None
    front_reader: LatestCameraReader | None = None
    wrist_reader: LatestCameraReader | None = None
    try:
        front = open_configured_camera(
            front_config,
            warmup_frames=system.cameras.warmup_frames,
            enable_depth=(
                front_config.depth
                if isinstance(front_config, SystemRealSenseCameraConfig)
                else False
            ),
        )
        wrist = open_configured_camera(
            wrist_config,
            warmup_frames=system.cameras.warmup_frames,
            enable_depth=False,
        )
        front_reader = LatestCameraReader(
            "front",
            front.read_frameset
            if isinstance(front, RealSenseColorCamera)
            else front.read_bgr,
        )
        wrist_reader = LatestCameraReader("wrist", wrist.read_bgr)
        front_reader.start()
        wrist_reader.start()
        readers = (front_reader, wrist_reader)
        _wait_for_latest_samples(readers, timeout_s=diagnostic.frame_timeout_s)
        rates = probe_camera_rates(
            readers,
            duration_s=diagnostic.rate_probe_s,
            front_target_fps=front_config.fps,
            wrist_target_fps=wrist_config.fps,
        )
        front_sample, wrist_sample = _latest_samples(readers)
        front_frameset = front_sample.value
        if isinstance(front_frameset, RealSenseFrameSet):
            front_img = front_frameset.color_bgr
        elif isinstance(front_frameset, np.ndarray):
            front_img = front_frameset
        else:
            raise RuntimeError("front camera returned an invalid image")
        wrist_img = wrist_sample.value
        if not isinstance(wrist_img, np.ndarray):
            raise RuntimeError("wrist camera returned an invalid image")
        front_path = out_dir / "cam0_front.jpg"
        wrist_path = out_dir / "cam1_wrist.jpg"
        sheet_path = out_dir / "contact_sheet.jpg"
        depth_path: Path | None = None
        depth_preview_path: Path | None = None
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), diagnostic.jpeg_quality]
        _write_image(front_path, front_img, jpeg_params)
        _write_image(wrist_path, wrist_img, jpeg_params)
        sheet_images: list[tuple[str, np.ndarray]] = [
            ("cam0 front", front_img),
            (f"cam1 wrist {wrist.label}", wrist_img),
        ]
        if isinstance(front_config, SystemRealSenseCameraConfig) and front_config.depth:
            if not isinstance(front_frameset, RealSenseFrameSet):
                raise RuntimeError("depth-enabled front camera returned no frameset")
            if front_frameset.depth_mm is None:
                raise RuntimeError(
                    "RealSense depth is enabled but no depth frame was captured"
                )
            depth_path = out_dir / "cam0_depth.png"
            depth_preview_path = out_dir / "cam0_depth_preview.jpg"
            preview = depth_preview(front_frameset.depth_mm)
            _write_image(depth_path, front_frameset.depth_mm)
            _write_image(depth_preview_path, preview, jpeg_params)
            sheet_images.append(("cam0 depth", preview))
        _write_image(
            sheet_path,
            contact_sheet(tuple(sheet_images)),
            jpeg_params,
        )
    finally:
        close_camera_resources(
            (wrist_reader, front_reader),
            (wrist, front),
        )

    print(
        f"cam0_usb={front.usb_type if isinstance(front, RealSenseColorCamera) else 'n/a'}"
    )
    print(f"config={system.path}")
    if diagnostic.rate_probe_s > 0:
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
    readers: tuple[LatestCameraReader, LatestCameraReader],
    *,
    duration_s: float,
    front_target_fps: float,
    wrist_target_fps: float,
) -> dict[str, float]:
    if duration_s <= 0:
        return {"front": 0.0, "wrist": 0.0}
    initial_counts = {reader.name: reader.frame_count() for reader in readers}
    start = time.perf_counter()
    time.sleep(duration_s)
    elapsed = max(time.perf_counter() - start, 1e-6)
    for reader in readers:
        exc = reader.exception()
        if exc is not None:
            raise RuntimeError(f"{reader.name} camera probe failed") from exc
    rates = {
        reader.name: (reader.frame_count() - initial_counts[reader.name]) / elapsed
        for reader in readers
    }
    too_slow = []
    for name, target_fps in (("front", front_target_fps), ("wrist", wrist_target_fps)):
        if rates[name] < 0.8 * target_fps:
            too_slow.append(f"{name}={rates[name]:.1f}fps target={target_fps:g}fps")
    if too_slow:
        raise RuntimeError("Camera rate probe failed: " + ", ".join(too_slow))
    return rates


def _wait_for_latest_samples(
    readers: tuple[LatestCameraReader, LatestCameraReader], *, timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for reader in readers:
            exc = reader.exception()
            if exc is not None:
                raise RuntimeError(f"{reader.name} camera reader failed") from exc
        if all(reader.latest() is not None for reader in readers):
            return
        time.sleep(0.01)
    details = ", ".join(
        f"{reader.name}:seq={reader.latest_seq()}" for reader in readers
    )
    raise RuntimeError(f"No frame from all cameras within {timeout_s:.1f}s ({details})")


def _latest_samples(
    readers: tuple[LatestCameraReader, LatestCameraReader],
) -> tuple[CameraSample, CameraSample]:
    samples = tuple(reader.latest() for reader in readers)
    if samples[0] is None or samples[1] is None:
        raise RuntimeError("camera samples disappeared after readiness")
    return samples


def _write_image(
    path: Path, image: np.ndarray, params: list[int] | None = None
) -> None:
    import cv2

    if not cv2.imwrite(str(path), image, params or []):
        raise RuntimeError(f"failed to write diagnostic image: {path}")


def depth_preview(depth_mm: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((*depth_mm.shape[:2], 3), dtype=np.uint8)
    near, far = np.percentile(valid, (2, 98))
    if far <= near:
        far = near + 1
    normalized = np.clip(
        (depth_mm.astype(np.float32) - near) * (255.0 / (far - near)), 0, 255
    )
    preview = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[depth_mm == 0] = (0, 0, 0)
    return preview


def contact_sheet(images: tuple[tuple[str, np.ndarray], ...]) -> np.ndarray:
    import cv2

    labelled = tuple(_label_image(label, image) for label, image in images)
    height = max(image.shape[0] for image in labelled)
    fitted = tuple(_fit_height(image, height) for image in labelled)
    return cv2.hconcat(fitted)


def _label_image(label: str, image: np.ndarray) -> np.ndarray:
    import cv2
    import numpy as np

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
    import cv2

    if image.shape[0] == height:
        return image
    scale = height / float(image.shape[0])
    width = max(1, round(image.shape[1] * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
