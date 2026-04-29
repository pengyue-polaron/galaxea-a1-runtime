import time
from pathlib import Path

import cv2
import numpy as np
import zmq

from a1.constants import ZMQ_CAM_PORT
from a1.utils import cfg_get as _cfg_get

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
    def __init__(self, *, serial: str | None, width: int, height: int, fps: int,
                 auto_exposure: bool = True, exposure: int | None = None,
                 gain: int | None = None, white_balance: int | None = None):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")

        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self._pipeline.start(config)

        # Configure exposure settings on the color sensor
        color_sensor = profile.get_device().query_sensors()[1]  # index 1 = RGB sensor
        if not auto_exposure and exposure is not None:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, exposure)
        if gain is not None:
            color_sensor.set_option(rs.option.gain, gain)
        if white_balance is not None:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.white_balance, white_balance)

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
        auto_exposure = bool(_cfg_get(camera_cfg, "auto_exposure", True))
        exposure = _cfg_get(camera_cfg, "exposure", None)
        if exposure is not None:
            exposure = int(exposure)
        gain = _cfg_get(camera_cfg, "gain", None)
        if gain is not None:
            gain = int(gain)
        white_balance = _cfg_get(camera_cfg, "white_balance", None)
        if white_balance is not None:
            white_balance = int(white_balance)
        source = RealSenseCamera(
            serial=serial, width=width, height=height, fps=fps,
            auto_exposure=auto_exposure, exposure=exposure, gain=gain,
            white_balance=white_balance,
        )
    elif backend == "opencv":
        device = _cfg_get(camera_cfg, "device", 0)
        backend_api = str(_cfg_get(camera_cfg, "backend_api", "auto")).lower()
        source = OpenCVCamera(device=device, width=width, height=height, fps=fps, backend_api=backend_api)
    else:
        raise ValueError(f"Unsupported camera backend '{backend}' for camera '{cam_id}'")

    return cam_id, source


def main(cfg=None, stop_event=None):
    bind = _cfg_get(cfg, "bind", f"tcp://*:{ZMQ_CAM_PORT}")
    jpeg_quality = int(_cfg_get(cfg, "jpeg_quality", 85))
    loop_sleep_s = float(_cfg_get(cfg, "loop_sleep_s", 0.001))
    startup_timeout_s = float(_cfg_get(cfg, "startup_timeout_s", 3.0))
    stall_timeout_s = float(_cfg_get(cfg, "stall_timeout_s", 2.0))
    debug_dump_enable = bool(_cfg_get(cfg, "debug_dump_enable", False))
    debug_dump_dir_raw = _cfg_get(cfg, "debug_dump_dir", None)
    debug_dump_every_n = int(_cfg_get(cfg, "debug_dump_every_n", 30))
    debug_dump_max_per_cam = int(_cfg_get(cfg, "debug_dump_max_per_cam", 0))

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
    camera_state = {}
    for camera_cfg in camera_cfgs:
        cam_id = str(_cfg_get(camera_cfg, "id", "cam_0"))
        try:
            built_cam_id, source = _build_camera_source(camera_cfg)
            if source is None:
                print(f"[Camera Server] Skip disabled camera: {built_cam_id}")
                continue
            sources[built_cam_id] = source
            print(f"[Camera Server] Camera ready: {built_cam_id}")
            now = time.monotonic()
            camera_state[built_cam_id] = {
                "startup_ts": now,
                "last_frame_ts": None,
                "warned_no_frame": False,
            }
        except Exception as exc:
            print(f"[Camera Server] Failed to init {cam_id}: {exc}")

    if not sources:
        raise RuntimeError("No camera source is available. Check camera config and dependencies.")

    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.bind(bind)

    print(f"[Camera Server] Publishing on {bind}")
    print("[Camera Server] Format: [camera_id, timestamp, jpeg_bytes]")

    debug_dump_dir = None
    dump_stats = {}
    dump_write_failed = set()
    if debug_dump_enable and debug_dump_dir_raw:
        debug_dump_dir = Path(str(debug_dump_dir_raw)).expanduser().resolve()
        debug_dump_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[Camera Server] Debug dump enabled: "
            f"dir={debug_dump_dir}, every_n={max(1, debug_dump_every_n)}, "
            f"max_per_cam={debug_dump_max_per_cam}"
        )

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    try:
        while stop_event is None or not stop_event.is_set():
            published = 0
            for cam_id, source in sources.items():
                state = camera_state[cam_id]
                try:
                    frame = source.read()
                except Exception as exc:
                    raise RuntimeError(f"Camera {cam_id} read failed: {exc}") from exc

                now = time.monotonic()
                if frame is None:
                    if state["last_frame_ts"] is None:
                        wait_s = now - state["startup_ts"]
                        if wait_s >= startup_timeout_s:
                            raise RuntimeError(
                                f"Camera {cam_id} produced no frames within {startup_timeout_s:.1f}s."
                            )
                        if wait_s >= 0.5 and not state["warned_no_frame"]:
                            print(
                                f"[Camera Server] Waiting for first frame from {cam_id} "
                                f"({wait_s:.1f}s elapsed) ..."
                            )
                            state["warned_no_frame"] = True
                    elif now - state["last_frame_ts"] >= stall_timeout_s:
                        raise RuntimeError(
                            f"Camera {cam_id} stopped producing frames for "
                            f"{stall_timeout_s:.1f}s."
                        )
                    continue

                ok, img_bytes = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                state["last_frame_ts"] = now
                timestamp = time.time_ns()
                encoded = img_bytes.tobytes()
                pub.send_multipart(
                    [
                        cam_id.encode("utf-8"),
                        str(timestamp).encode("ascii"),
                        encoded,
                    ]
                )
                published += 1

                if debug_dump_dir is not None:
                    cam_stats = dump_stats.setdefault(cam_id, {"seen": 0, "saved": 0})
                    cam_stats["seen"] += 1
                    every_n = max(1, debug_dump_every_n)
                    should_save = (cam_stats["seen"] % every_n) == 0
                    under_limit = (
                        debug_dump_max_per_cam <= 0 or cam_stats["saved"] < debug_dump_max_per_cam
                    )
                    if should_save and under_limit:
                        cam_dir = debug_dump_dir / cam_id
                        if cam_id not in dump_write_failed:
                            try:
                                cam_dir.mkdir(parents=True, exist_ok=True)
                                out_path = cam_dir / f"{timestamp}_{cam_stats['saved']:06d}.jpg"
                                out_path.write_bytes(encoded)
                                cam_stats["saved"] += 1
                            except Exception as exc:
                                dump_write_failed.add(cam_id)
                                print(
                                    f"[Camera Server] WARNING: failed to dump frames for {cam_id}: {exc}"
                                )

            if published == 0:
                time.sleep(loop_sleep_s)
    except KeyboardInterrupt:
        print("\n[Camera Server] Stopped by user.")
    finally:
        if debug_dump_dir is not None:
            print(f"[Camera Server] Debug dump summary @ {debug_dump_dir}")
            for cam_id in sorted(dump_stats):
                stats = dump_stats[cam_id]
                print(
                    f"  - {cam_id}: seen={stats['seen']}, saved={stats['saved']}"
                )
        for source in sources.values():
            source.close()
        pub.close(0)
        context.term()


if __name__ == "__main__":
    main()
