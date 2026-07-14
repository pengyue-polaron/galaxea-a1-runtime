"""Read-only LAN MJPEG preview for already-open camera readers."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import cv2
import numpy as np

from galaxea_a1_runtime.hardware.image_geometry import ImageRoi, draw_image_roi


@dataclass(frozen=True)
class WebPreviewConfig:
    enabled: bool = False
    bind: str = "0.0.0.0"
    port: int = 8088
    fps: float = 10.0
    jpeg_quality: int = 75

    def validate(self) -> None:
        if not self.bind:
            raise ValueError("web_preview.bind must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("web_preview.port must be in [1, 65535]")
        if self.fps <= 0:
            raise ValueError("web_preview.fps must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("web_preview.jpeg_quality must be in [1, 100]")


def parse_web_preview_config(data: dict[str, Any], *, repo_root: Path) -> WebPreviewConfig:
    del repo_root
    config = WebPreviewConfig(
        enabled=bool(data.get("enabled", False)),
        bind=str(data.get("bind", "0.0.0.0")),
        port=int(data.get("port", 8088)),
        fps=float(data.get("fps", 10.0)),
        jpeg_quality=int(data.get("jpeg_quality", 75)),
    )
    config.validate()
    return config


def web_preview_argv(config: WebPreviewConfig) -> list[str]:
    return [
        "--web-preview" if config.enabled else "--no-web-preview",
        "--web-preview-bind",
        config.bind,
        "--web-preview-port",
        str(config.port),
        "--web-preview-fps",
        f"{config.fps:g}",
        "--web-preview-jpeg-quality",
        str(config.jpeg_quality),
    ]


def add_web_preview_arguments(parser: Any) -> None:
    parser.add_argument("--web-preview", action=__import__("argparse").BooleanOptionalAction, default=False)
    parser.add_argument("--web-preview-bind", default="0.0.0.0")
    parser.add_argument("--web-preview-port", type=int, default=8088)
    parser.add_argument("--web-preview-fps", type=float, default=10.0)
    parser.add_argument("--web-preview-jpeg-quality", type=int, default=75)


def web_preview_config_from_args(args: Any) -> WebPreviewConfig:
    config = WebPreviewConfig(
        enabled=bool(args.web_preview),
        bind=str(args.web_preview_bind),
        port=int(args.web_preview_port),
        fps=float(args.web_preview_fps),
        jpeg_quality=int(args.web_preview_jpeg_quality),
    )
    config.validate()
    return config


@dataclass
class _StreamState:
    reader: Any
    extract: Callable[[Any], np.ndarray]
    source: str
    overlay_roi: ImageRoi | None = None
    overlay_label: str = ""
    last_source_seq: int = -1
    jpeg: bytes | None = None
    encoded_seq: int = -1
    source_monotonic_s: float | None = None
    encoded_monotonic_s: float | None = None
    encode_times: deque[float] = field(default_factory=lambda: deque(maxlen=30))


class CameraWebPreview:
    """Serve latest frames without opening cameras or touching ROS."""

    def __init__(self, config: WebPreviewConfig):
        config.validate()
        self.config = config
        self._streams: dict[str, _StreamState] = {}
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._encoder_thread: threading.Thread | None = None

    def register_reader(
        self,
        name: str,
        reader: Any,
        *,
        extract: Callable[[Any], np.ndarray],
        source: str,
        overlay_roi: ImageRoi | None = None,
        overlay_label: str = "",
    ) -> None:
        if self._server is not None:
            raise RuntimeError("cannot register camera streams after web preview starts")
        if name not in {"agent", "wrist"}:
            raise ValueError(f"unsupported preview stream {name!r}")
        self._streams[name] = _StreamState(
            reader=reader,
            extract=extract,
            source=source,
            overlay_roi=overlay_roi,
            overlay_label=overlay_label,
        )

    def start(self) -> None:
        if not self.config.enabled:
            return
        if set(self._streams) != {"agent", "wrist"}:
            raise RuntimeError("web preview requires registered 'agent' and 'wrist' readers")
        handler = self._handler_type()
        self._server = ThreadingHTTPServer((self.config.bind, self.config.port), handler)
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="camera-web-http",
            daemon=True,
        )
        self._encoder_thread = threading.Thread(
            target=self._encode_loop,
            name="camera-web-encoder",
            daemon=True,
        )
        self._encoder_thread.start()
        self._server_thread.start()
        print(
            "[Camera Web] listening on "
            f"http://{self.config.bind}:{self.config.port}",
            flush=True,
        )

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        if self._encoder_thread is not None:
            self._encoder_thread.join(timeout=2.0)
        self._server = None

    def _encode_loop(self) -> None:
        interval = 1.0 / self.config.fps
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        while not self._stop.is_set():
            started = time.perf_counter()
            updates: list[tuple[str, int, float, bytes, float]] = []
            for name, state in self._streams.items():
                sample = state.reader.latest()
                if sample is None or sample.seq == state.last_source_seq:
                    continue
                try:
                    image = state.extract(sample.value)
                    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
                        continue
                    if state.overlay_roi is not None:
                        image = draw_image_roi(
                            image,
                            state.overlay_roi,
                            label=state.overlay_label,
                        )
                    ok, encoded = cv2.imencode(".jpg", image, jpeg_params)
                except Exception:
                    continue
                if not ok:
                    continue
                now = time.perf_counter()
                updates.append((name, sample.seq, sample.monotonic_s, encoded.tobytes(), now))
            if updates:
                with self._condition:
                    for name, source_seq, source_time, jpeg, now in updates:
                        state = self._streams[name]
                        state.last_source_seq = source_seq
                        state.source_monotonic_s = source_time
                        state.jpeg = jpeg
                        state.encoded_seq += 1
                        state.encoded_monotonic_s = now
                        state.encode_times.append(now)
                    self._condition.notify_all()
            remaining = interval - (time.perf_counter() - started)
            self._stop.wait(max(0.001, remaining))

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        preview = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "A1CameraWeb/1"

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                path = urlsplit(self.path).path
                if path == "/":
                    preview._send_dashboard(self)
                elif path == "/healthz":
                    preview._send_health(self)
                elif path in {"/agent.mjpg", "/wrist.mjpg"}:
                    preview._send_mjpeg(self, path[1:].split(".", 1)[0])
                elif path in {"/snapshot/agent.jpg", "/snapshot/wrist.jpg"}:
                    preview._send_snapshot(self, Path(path).stem)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        return Handler

    def _send_dashboard(self, handler: BaseHTTPRequestHandler) -> None:
        body = _DASHBOARD_HTML.encode()
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("X-Frame-Options", "DENY")
        handler.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'",
        )
        handler.end_headers()
        handler.wfile.write(body)

    def _send_health(self, handler: BaseHTTPRequestHandler) -> None:
        now = time.perf_counter()
        streams: dict[str, Any] = {}
        with self._condition:
            for name, state in self._streams.items():
                age = None if state.source_monotonic_s is None else max(0.0, now - state.source_monotonic_s)
                times = list(state.encode_times)
                fps = 0.0
                if len(times) >= 2 and times[-1] > times[0]:
                    fps = (len(times) - 1) / (times[-1] - times[0])
                streams[name] = {
                    "source": state.source,
                    "ready": state.jpeg is not None,
                    "age_s": None if age is None else round(age, 3),
                    "preview_fps": round(fps, 2),
                    "source_seq": state.last_source_seq,
                    "overlay_roi_xywh": (
                        None if state.overlay_roi is None else list(state.overlay_roi.xywh)
                    ),
                }
        body = json.dumps({"ok": all(item["ready"] for item in streams.values()), "streams": streams}).encode()
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(body)

    def _send_snapshot(self, handler: BaseHTTPRequestHandler, name: str) -> None:
        with self._condition:
            state = self._streams.get(name)
            jpeg = None if state is None else state.jpeg
        if jpeg is None:
            handler.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "camera frame not ready")
            return
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "image/jpeg")
        handler.send_header("Content-Length", str(len(jpeg)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(jpeg)

    def _send_mjpeg(self, handler: BaseHTTPRequestHandler, name: str) -> None:
        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        handler.send_header("Pragma", "no-cache")
        handler.end_headers()
        last_seq = -1
        try:
            while not self._stop.is_set():
                with self._condition:
                    state = self._streams[name]
                    self._condition.wait_for(
                        lambda: self._stop.is_set() or (state.jpeg is not None and state.encoded_seq != last_seq),
                        timeout=2.0,
                    )
                    if self._stop.is_set() or state.jpeg is None or state.encoded_seq == last_seq:
                        continue
                    jpeg = state.jpeg
                    last_seq = state.encoded_seq
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return


def color_from_frameset(value: Any) -> np.ndarray:
    return value.color_bgr


def color_from_bgr(value: Any) -> np.ndarray:
    return value


_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Galaxea A1 Cameras</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#111;color:#eee}body{margin:0;padding:18px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:16px}h1{font-size:20px;margin:0 0 14px}
#status{font:13px ui-monospace,monospace;color:#9fd}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
figure{margin:0;background:#1c1c1c;border:1px solid #333;border-radius:10px;overflow:hidden}img{display:block;width:100%;height:auto;background:#000}
figcaption{padding:9px 12px;font-weight:600}.note{margin-top:12px;color:#aaa;font-size:12px}
</style></head><body><header><h1>Galaxea A1 · Live Cameras</h1><span id="status">connecting…</span></header>
<main class="grid"><figure><img src="/agent.mjpg" alt="Agent view"><figcaption>Agent view · D455 · red box = recorded area</figcaption></figure>
<figure><img src="/wrist.mjpg" alt="Wrist view"><figcaption>Wrist view · D405</figcaption></figure></main>
<p class="note">Read-only preview. This service has no robot-control endpoints.</p><script>
async function health(){try{const r=await fetch('/healthz',{cache:'no-store'}),d=await r.json();
const s=Object.entries(d.streams).map(([n,v])=>`${n}: ${v.preview_fps}fps age=${v.age_s ?? '-'}s`).join(' · ');
document.getElementById('status').textContent=s;}catch(e){document.getElementById('status').textContent='health unavailable';}}
health();setInterval(health,2000);</script></body></html>"""
