import time

import cv2
import numpy as np
import zmq

from datacoach.constants import ZMQ_CAM_PORT

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


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


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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


def main(cfg=None):
    bind = _cfg_get(cfg, "bind", f"tcp://*:{ZMQ_CAM_PORT}")
    jpeg_quality = int(_cfg_get(cfg, "jpeg_quality", 85))
    loop_sleep_s = float(_cfg_get(cfg, "loop_sleep_s", 0.001))

    camera_cfgs = _cfg_get(cfg, "cameras", None)
    if not camera_cfgs:
        camera_cfgs = [
            {
                "id": "cam_0",
                "backend": "realsense",
                "width": 640,
                "height": 480,
                "fps": 30,
                "serial": None,
            }
        ]

    sources = {}
    for camera_cfg in camera_cfgs:
        cam_id = str(_cfg_get(camera_cfg, "id", "cam_0"))
        try:
            built_cam_id, source = _build_camera_source(camera_cfg)
            if source is None:
                print(f"[Camera Server] Skip disabled camera: {built_cam_id}")
                continue
            sources[built_cam_id] = source
            print(f"[Camera Server] Camera ready: {built_cam_id}")
        except Exception as exc:
            print(f"[Camera Server] Failed to init {cam_id}: {exc}")

    if not sources:
        raise RuntimeError("No camera source is available. Check camera config and dependencies.")

    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind(bind)

    print(f"[Camera Server] Publishing on {bind}")
    print("[Camera Server] Format: [camera_id, timestamp, jpeg_bytes]")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    try:
        while True:
            published = 0
            for cam_id, source in sources.items():
                frame = source.read()
                if frame is None:
                    continue

                ok, img_bytes = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                timestamp = time.time_ns()
                pub.send_multipart(
                    [
                        cam_id.encode("utf-8"),
                        str(timestamp).encode("ascii"),
                        img_bytes.tobytes(),
                    ]
                )
                published += 1

            if published == 0:
                time.sleep(loop_sleep_s)
    except KeyboardInterrupt:
        print("\n[Camera Server] Stopped by user.")
    finally:
        for source in sources.values():
            source.close()
        pub.close(0)
        context.term()


if __name__ == "__main__":
    main()
