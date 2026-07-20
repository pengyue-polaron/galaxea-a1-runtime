"""Read-only LAN MJPEG preview for already-open camera readers."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

from galaxea_a1_runtime.configuration.image import ImageRoi
from galaxea_a1_runtime.configuration.web_preview import WebPreviewConfig
from galaxea_a1_runtime.console import info
from galaxea_a1_runtime.hardware.image_geometry import draw_image_roi


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
    last_error: str | None = None
    error_monotonic_s: float | None = None


class CameraWebPreview:
    """Serve latest frames without opening cameras or touching ROS."""

    def __init__(self, config: WebPreviewConfig, *, max_source_age_s: float):
        config.validate()
        if max_source_age_s <= 0:
            raise ValueError("max_source_age_s must be positive")
        self.config = config
        self.max_source_age_s = max_source_age_s
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
            raise RuntimeError(
                "cannot register camera streams after web preview starts"
            )
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
            raise RuntimeError(
                "web preview requires registered 'agent' and 'wrist' readers"
            )
        handler = self._handler_type()
        self._server = ThreadingHTTPServer(
            (self.config.bind, self.config.port), handler
        )
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
        info(f"Camera Web listening on http://{self.config.bind}:{self.config.port}")

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
        alive = [
            thread.name
            for thread in (self._server_thread, self._encoder_thread)
            if thread is not None and thread.is_alive()
        ]
        self._server = None
        self._server_thread = None
        self._encoder_thread = None
        if alive:
            raise RuntimeError(f"web preview threads did not stop: {alive}")

    def _encode_loop(self) -> None:
        interval = 1.0 / self.config.fps
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        while not self._stop.is_set():
            started = time.perf_counter()
            updates: list[tuple[str, int, float, bytes, float]] = []
            for name, state in self._streams.items():
                reader_error = _reader_exception(state.reader)
                if reader_error is not None:
                    self._set_stream_error(name, reader_error)
                    continue
                sample = state.reader.latest()
                if sample is None or sample.seq == state.last_source_seq:
                    continue
                try:
                    image = state.extract(sample.value)
                    if (
                        not isinstance(image, np.ndarray)
                        or image.ndim != 3
                        or image.shape[2] != 3
                    ):
                        raise ValueError(
                            "preview image must be a HxWx3 numpy array, got "
                            f"{type(image).__name__} shape={getattr(image, 'shape', None)}"
                        )
                    if state.overlay_roi is not None:
                        image = draw_image_roi(
                            image,
                            state.overlay_roi,
                            label=state.overlay_label,
                        )
                    ok, encoded = cv2.imencode(".jpg", image, jpeg_params)
                except Exception as exc:
                    self._set_stream_error(name, exc, source_seq=sample.seq)
                    continue
                if not ok:
                    self._set_stream_error(
                        name,
                        RuntimeError("OpenCV JPEG encoder returned failure"),
                        source_seq=sample.seq,
                    )
                    continue
                now = time.perf_counter()
                updates.append(
                    (name, sample.seq, sample.monotonic_s, encoded.tobytes(), now)
                )
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
                        state.last_error = None
                        state.error_monotonic_s = None
                    self._condition.notify_all()
            remaining = interval - (time.perf_counter() - started)
            self._stop.wait(max(0.001, remaining))

    def _set_stream_error(
        self, name: str, error: BaseException, *, source_seq: int | None = None
    ) -> None:
        with self._condition:
            state = self._streams[name]
            if source_seq is not None:
                state.last_source_seq = source_seq
            state.last_error = f"{type(error).__name__}: {error}"
            state.error_monotonic_s = time.perf_counter()
            self._condition.notify_all()

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
                age = (
                    None
                    if state.source_monotonic_s is None
                    else max(0.0, now - state.source_monotonic_s)
                )
                times = list(state.encode_times)
                fps = 0.0
                if len(times) >= 2 and times[-1] > times[0]:
                    fps = (len(times) - 1) / (times[-1] - times[0])
                streams[name] = {
                    "source": state.source,
                    "ready": state.jpeg is not None,
                    "fresh": age is not None and age <= self.max_source_age_s,
                    "age_s": None if age is None else round(age, 3),
                    "preview_fps": round(fps, 2),
                    "source_seq": state.last_source_seq,
                    "error": state.last_error,
                    "overlay_roi_xywh": (
                        None
                        if state.overlay_roi is None
                        else list(state.overlay_roi.xywh)
                    ),
                }
        ok = all(
            item["ready"] and item["fresh"] and item["error"] is None
            for item in streams.values()
        )
        body = json.dumps({"ok": ok, "streams": streams}).encode()
        handler.send_response(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE)
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
                        lambda state=state, last_seq=last_seq: (
                            self._stop.is_set()
                            or (
                                state.jpeg is not None and state.encoded_seq != last_seq
                            )
                        ),
                        timeout=2.0,
                    )
                    if (
                        self._stop.is_set()
                        or state.jpeg is None
                        or state.encoded_seq == last_seq
                    ):
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


def _reader_exception(reader: Any) -> BaseException | None:
    getter = getattr(reader, "exception", None)
    return getter() if callable(getter) else None


_DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A1 Cameras</title><style>
html,body{margin:0;min-height:100%;background:#000}main{min-height:100vh;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:2px}
img{display:block;width:100%;height:100vh;object-fit:contain;background:#000}@media(max-width:800px){main{grid-template-columns:1fr}img{height:50vh}}
</style></head><body><main><img data-stream="agent" alt=""><img data-stream="wrist" alt=""></main><script>
const images=[...document.querySelectorAll('img[data-stream]')];let retryTimer=null;
function connect(){retryTimer=null;for(const image of images){image.src=`/${image.dataset.stream}.mjpg?t=${Date.now()}`;}}
function retry(){if(retryTimer===null)retryTimer=setTimeout(connect,1000);}
for(const image of images)image.addEventListener('error',retry);
async function probe(){try{const response=await fetch('/healthz',{cache:'no-store'});if(!response.ok)throw new Error();}catch(error){retry();}}
connect();setInterval(probe,2000);</script></body></html>"""
