#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class OpenCVCamera:
    def __init__(self, *, device, width: int, height: int, fps: int, backend_api: str = "auto"):
        source = int(device) if str(device).isdigit() else str(device)
        if backend_api == "v4l2":
            self._cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open OpenCV camera device={device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        self._cap.release()


class RealSenseCamera:
    def __init__(self, *, serial: str | None, width: int, height: int, fps: int):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")

        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)

    def read(self):
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self):
        self._pipeline.stop()


def _build_camera_source(camera_cfg):
    cam_id = str(_cfg_get(camera_cfg, "id", "cam_0"))
    enabled = bool(_cfg_get(camera_cfg, "enabled", True))
    if not enabled:
        return cam_id, None

    backend = str(_cfg_get(camera_cfg, "backend", "opencv")).lower()
    width = int(_cfg_get(camera_cfg, "width", 640))
    height = int(_cfg_get(camera_cfg, "height", 480))
    fps = int(_cfg_get(camera_cfg, "fps", 30))

    if backend == "realsense":
        serial = _cfg_get(camera_cfg, "serial", None)
        source = RealSenseCamera(serial=serial, width=width, height=height, fps=fps)
    elif backend == "opencv":
        device = _cfg_get(camera_cfg, "device", 0)
        backend_api = str(_cfg_get(camera_cfg, "backend_api", "auto")).lower()
        source = OpenCVCamera(device=device, width=width, height=height, fps=fps, backend_api=backend_api)
    else:
        raise ValueError(f"Unsupported camera backend '{backend}' for camera '{cam_id}'")

    return cam_id, source


def _format_backend(camera_cfg):
    backend = str(_cfg_get(camera_cfg, "backend", "opencv")).lower()
    if backend == "realsense":
        serial = _cfg_get(camera_cfg, "serial", None)
        return f"realsense(serial={serial})"
    device = _cfg_get(camera_cfg, "device", 0)
    backend_api = _cfg_get(camera_cfg, "backend_api", "auto")
    return f"opencv(device={device}, backend_api={backend_api})"


def _save_probe_image(output_dir: Path, cam_id: str, frame):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{cam_id}_probe.jpg"
    cv2.imwrite(str(path), frame)
    return path


def test_one_camera(camera_cfg, timeout_s: float, output_dir: Path | None):
    cam_id = str(_cfg_get(camera_cfg, "id", "cam"))
    enabled = bool(_cfg_get(camera_cfg, "enabled", True))

    if not enabled:
        print(f"[SKIP] {cam_id}: disabled")
        return True

    desc = _format_backend(camera_cfg)
    source = None
    start = time.time()
    frame = None
    frame_count = 0

    try:
        source_id, source = _build_camera_source(camera_cfg)
        cam_id = source_id
        print(f"[TEST] {cam_id}: {desc}")

        while time.time() - start < timeout_s:
            frame = source.read()
            if frame is None:
                time.sleep(0.01)
                continue
            frame_count += 1
            if frame_count >= 3:
                break

        if frame is None:
            print(f"[FAIL] {cam_id}: no frame within {timeout_s:.1f}s")
            return False

        h, w = frame.shape[:2]
        print(f"[PASS] {cam_id}: frame={w}x{h}, captured_frames={frame_count}")
        if output_dir is not None:
            image_path = _save_probe_image(output_dir, cam_id, frame)
            print(f"       saved: {image_path}")
        return True
    except Exception as exc:
        print(f"[FAIL] {cam_id}: {exc}")
        return False
    finally:
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Test camera connections from DataCoach config (RealSense + OpenCV)."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "drag_replay.yaml"),
        help="YAML config path (expects camera_server.cameras).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=6.0,
        help="Per-camera timeout waiting for frames.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(ROOT_DIR / "outputs" / "camera_probe"),
        help="Where to save one probe frame per camera.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save probe images.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        return 2

    if OmegaConf is not None:
        cfg = OmegaConf.load(str(cfg_path))
    elif yaml is not None:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        print("[ERROR] Neither omegaconf nor pyyaml is available in this python env.")
        return 2

    camera_cfgs = _cfg_get(_cfg_get(cfg, "camera_server", {}), "cameras", None)
    if not camera_cfgs:
        print(f"[ERROR] No camera_server.cameras in {cfg_path}")
        return 2

    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Cameras: {len(camera_cfgs)}")

    output_dir = None if args.no_save else Path(args.save_dir).expanduser().resolve()
    ok = True
    for cam_cfg in camera_cfgs:
        ok = test_one_camera(cam_cfg, timeout_s=args.timeout_s, output_dir=output_dir) and ok

    if ok:
        print("[SUMMARY] Camera connection test passed.")
        return 0

    print("[SUMMARY] Camera connection test failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
