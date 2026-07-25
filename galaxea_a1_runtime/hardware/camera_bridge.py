"""Raw dual-camera bridge shared by persistent Web and application consumers."""

from __future__ import annotations

import hashlib
import json
import socket
import socketserver
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from galaxea_a1_runtime.configuration.cameras import (
    SystemCamerasConfig,
    SystemRealSenseCameraConfig,
)
from galaxea_a1_runtime.hardware.camera_reader import CameraReader, CameraSample
from galaxea_a1_runtime.hardware.cameras import RealSenseFrameSet
from galaxea_a1_runtime.runtime.local_ipc import (
    prepare_unix_socket,
    process_socket_path,
    receive_packet,
    send_packet,
)

_PROTOCOL_VERSION = 2
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 128 * 1024 * 1024
_REQUEST_WAIT_S = 0.25


@dataclass(frozen=True)
class CameraBridgeMetadata:
    contract_digest: str
    front_source: str
    wrist_source: str
    front_usb_type: str
    depth_enabled: bool
    front_color_shape: tuple[int, int, int]
    wrist_color_shape: tuple[int, int, int]


def camera_bridge_socket_path(*, state_root: Path | None = None) -> Path:
    """Return the per-user lifecycle socket shared by scripts and apps."""

    return process_socket_path("a1-camera-bridge.sock", state_root=state_root)


def camera_contract_digest(config: SystemCamerasConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class CameraBridgeServer:
    """Expose exact latest raw frames while physical readers remain sole owners."""

    def __init__(
        self,
        config: SystemCamerasConfig,
        *,
        front_reader: CameraReader,
        wrist_reader: CameraReader,
        front_source: str,
        wrist_source: str,
        front_usb_type: str,
        socket_path: Path | None = None,
    ) -> None:
        if front_reader.name != "front" or wrist_reader.name != "wrist":
            raise ValueError("camera bridge readers must be named front and wrist")
        self.config = config
        self.front_reader = front_reader
        self.wrist_reader = wrist_reader
        self.socket_path = socket_path or camera_bridge_socket_path()
        self.metadata = CameraBridgeMetadata(
            contract_digest=camera_contract_digest(config),
            front_source=front_source,
            wrist_source=wrist_source,
            front_usb_type=front_usb_type,
            depth_enabled=(
                isinstance(config.front, SystemRealSenseCameraConfig)
                and config.front.depth
            ),
            front_color_shape=(config.front.height, config.front.width, 3),
            wrist_color_shape=(config.wrist.height, config.wrist.width, 3),
        )
        self._server = _CameraUnixServer(self.socket_path, self)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("camera bridge server was already started")
        self._thread = threading.Thread(
            target=self._serve,
            name="camera-bridge-server",
            daemon=False,
        )
        self._thread.start()

    def close(self) -> None:
        thread = self._thread
        if thread is not None:
            self._server.shutdown()
        self._server.close_clients()
        self._server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("camera bridge server did not stop")
        self._thread = None
        try:
            if self.socket_path.is_socket():
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def exception(self) -> BaseException | None:
        return self._error

    def next_response(self, request: Any) -> dict[str, Any]:
        try:
            return self._next_response(request)
        except BaseException as exc:  # Client receives an explicit fail-closed error.
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _serve(self) -> None:
        try:
            self._server.serve_forever(poll_interval=0.1)
        except BaseException as exc:  # Surfaced to the persistent owner loop.
            self._error = exc

    def _next_response(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("camera bridge request must be a map")
        if request.get("version") != _PROTOCOL_VERSION:
            raise ValueError("camera bridge protocol version mismatch")
        operation = request.get("op")
        if operation == "describe":
            return {
                "ok": True,
                "version": _PROTOCOL_VERSION,
                "metadata": asdict(self.metadata),
            }
        if operation != "next_pair":
            raise ValueError("unsupported camera bridge operation")
        if request.get("contract_digest") != self.metadata.contract_digest:
            raise ValueError(
                "camera contract mismatch; restart the persistent camera bridge"
            )
        after = request.get("after")
        if not isinstance(after, dict):
            raise ValueError("camera bridge request after must be a map")
        front_after = _plain_int(after.get("front"), label="after.front")
        wrist_after = _plain_int(after.get("wrist"), label="after.wrist")
        deadline = time.perf_counter() + _REQUEST_WAIT_S
        while True:
            self._raise_reader_errors()
            front = self.front_reader.latest()
            wrist = self.wrist_reader.latest()
            if (
                front is not None
                and wrist is not None
                and front.seq > front_after
                and wrist.seq > wrist_after
            ):
                return {
                    "ok": True,
                    "version": _PROTOCOL_VERSION,
                    "metadata": asdict(self.metadata),
                    "changed": True,
                    "front": self._encode_front(front),
                    "wrist": self._encode_wrist(wrist),
                }
            if time.perf_counter() >= deadline:
                return {
                    "ok": True,
                    "version": _PROTOCOL_VERSION,
                    "metadata": asdict(self.metadata),
                    "changed": False,
                }
            time.sleep(0.002)

    def _raise_reader_errors(self) -> None:
        for reader in (self.front_reader, self.wrist_reader):
            error = reader.exception()
            if error is not None:
                raise RuntimeError(
                    f"{reader.name} physical camera reader failed"
                ) from error

    def _encode_front(self, sample: CameraSample) -> dict[str, Any]:
        if not isinstance(sample.value, RealSenseFrameSet):
            raise TypeError("front camera bridge value must be RealSenseFrameSet")
        front = self.config.front
        color = _encode_array(
            sample.value.color_bgr,
            shape=(front.height, front.width, 3),
            dtype=np.dtype(np.uint8),
            label="front color",
        )
        depth = None
        if self.metadata.depth_enabled:
            if sample.value.depth_mm is None:
                raise ValueError("front depth is enabled but the frame has no depth")
            depth = _encode_array(
                sample.value.depth_mm,
                shape=(front.height, front.width),
                dtype=np.dtype(np.uint16),
                label="front aligned depth",
            )
        return {
            "seq": sample.seq,
            "monotonic_s": sample.monotonic_s,
            "color_bgr": color,
            "depth_mm": depth,
        }

    def _encode_wrist(self, sample: CameraSample) -> dict[str, Any]:
        wrist = self.config.wrist
        return {
            "seq": sample.seq,
            "monotonic_s": sample.monotonic_s,
            "color_bgr": _encode_array(
                sample.value,
                shape=(wrist.height, wrist.width, 3),
                dtype=np.dtype(np.uint8),
                label="wrist color",
            ),
        }


class CameraBridgeReader:
    """Reader view backed by one atomic raw pair in CameraBridgeReaders."""

    def __init__(self, owner: CameraBridgeReaders, name: str) -> None:
        self._owner = owner
        self.name = name

    def latest(self) -> CameraSample | None:
        return self._owner._latest_sample(self.name)

    def latest_seq(self) -> int:
        sample = self.latest()
        return -1 if sample is None else sample.seq

    def frame_count(self) -> int:
        return self.latest_seq() + 1

    def exception(self) -> BaseException | None:
        return self._owner.exception()


class CameraBridgeReaders:
    """Receive exact raw pairs from the persistent camera owner in one thread."""

    def __init__(
        self,
        config: SystemCamerasConfig,
        *,
        socket_path: Path | None = None,
    ) -> None:
        self.config = config
        self.socket_path = socket_path or camera_bridge_socket_path()
        self.contract_digest = camera_contract_digest(config)
        self.front = CameraBridgeReader(self, "front")
        self.wrist = CameraBridgeReader(self, "wrist")
        self._lock = threading.Lock()
        self._latest: dict[str, CameraSample | None] = {
            "front": None,
            "wrist": None,
        }
        self._metadata: CameraBridgeMetadata | None = None
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    @property
    def metadata(self) -> CameraBridgeMetadata:
        with self._lock:
            metadata = self._metadata
        if metadata is None:
            raise RuntimeError("camera bridge metadata is not ready")
        return metadata

    def start(self, *, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("camera bridge startup timeout must be positive")
        if self._thread is not None:
            raise RuntimeError("camera bridge readers were already started")
        self._thread = threading.Thread(
            target=self._run,
            name="camera-bridge-client",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout=timeout_s):
            self.close()
            raise RuntimeError(
                f"camera bridge produced no raw pair within {timeout_s:.1f}s"
            )
        error = self.exception()
        if error is not None:
            self.close()
            raise RuntimeError(
                "persistent camera bridge is unavailable; run `just camera-web`"
            ) from error

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            active_socket = self._socket
        if active_socket is not None:
            with suppress(OSError):
                active_socket.shutdown(socket.SHUT_RDWR)
            active_socket.close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("camera bridge client did not stop")
        self._thread = None

    def exception(self) -> BaseException | None:
        with self._lock:
            return self._error

    def _latest_sample(self, name: str) -> CameraSample | None:
        with self._lock:
            return self._latest[name]

    def latest_pair(self) -> tuple[CameraSample, CameraSample] | None:
        """Return the latest front/wrist samples from one atomic bridge update."""

        with self._lock:
            front = self._latest["front"]
            wrist = self._latest["wrist"]
        if front is None or wrist is None:
            return None
        return front, wrist

    def _run(self) -> None:
        active_socket: socket.socket | None = None
        try:
            active_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            active_socket.settimeout(1.0)
            active_socket.connect(str(self.socket_path))
            with self._lock:
                self._socket = active_socket
            after = {"front": -1, "wrist": -1}
            while not self._stop.is_set():
                send_packet(
                    active_socket,
                    {
                        "version": _PROTOCOL_VERSION,
                        "op": "next_pair",
                        "contract_digest": self.contract_digest,
                        "after": after,
                    },
                )
                response = receive_packet(active_socket, max_bytes=_MAX_RESPONSE_BYTES)
                if response is None:
                    raise ConnectionError("camera bridge closed the raw stream")
                metadata = self._decode_metadata(response)
                if not response.get("changed"):
                    continue
                front = _decode_front(response.get("front"), self.config)
                wrist = _decode_wrist(response.get("wrist"), self.config)
                with self._lock:
                    self._metadata = metadata
                    self._latest["front"] = front
                    self._latest["wrist"] = wrist
                after = {"front": front.seq, "wrist": wrist.seq}
                self._ready.set()
        except BaseException as exc:  # Surfaced through both reader views.
            if not self._stop.is_set():
                with self._lock:
                    self._error = exc
        finally:
            self._ready.set()
            with self._lock:
                self._socket = None
            if active_socket is not None:
                active_socket.close()

    def _decode_metadata(self, response: Any) -> CameraBridgeMetadata:
        if not isinstance(response, dict):
            raise ValueError("camera bridge response must be a map")
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "camera bridge error")))
        if response.get("version") != _PROTOCOL_VERSION:
            raise ValueError("camera bridge response version mismatch")
        raw = response.get("metadata")
        if not isinstance(raw, dict) or set(raw) != set(
            CameraBridgeMetadata.__annotations__
        ):
            raise ValueError("camera bridge metadata is incomplete")
        metadata = CameraBridgeMetadata(
            contract_digest=str(raw["contract_digest"]),
            front_source=str(raw["front_source"]),
            wrist_source=str(raw["wrist_source"]),
            front_usb_type=str(raw["front_usb_type"]),
            depth_enabled=raw["depth_enabled"],
            front_color_shape=_shape3(raw["front_color_shape"], label="front color"),
            wrist_color_shape=_shape3(raw["wrist_color_shape"], label="wrist color"),
        )
        if metadata.contract_digest != self.contract_digest:
            raise ValueError("camera bridge metadata contract mismatch")
        expected_depth = (
            isinstance(self.config.front, SystemRealSenseCameraConfig)
            and self.config.front.depth
        )
        if (
            not isinstance(metadata.depth_enabled, bool)
            or metadata.depth_enabled != expected_depth
        ):
            raise ValueError("camera bridge depth contract mismatch")
        if metadata.front_color_shape != (
            self.config.front.height,
            self.config.front.width,
            3,
        ):
            raise ValueError("camera bridge front color contract mismatch")
        if metadata.wrist_color_shape != (
            self.config.wrist.height,
            self.config.wrist.width,
            3,
        ):
            raise ValueError("camera bridge wrist color contract mismatch")
        return metadata


class _CameraRequestHandler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        super().setup()
        server = self.server
        assert isinstance(server, _CameraUnixServer)
        server.register_client(self.request)

    def handle(self) -> None:
        server = self.server
        assert isinstance(server, _CameraUnixServer)
        try:
            while True:
                request = receive_packet(self.request, max_bytes=_MAX_REQUEST_BYTES)
                if request is None:
                    return
                send_packet(self.request, server.bridge.next_response(request))
        except (ConnectionError, OSError):
            return

    def finish(self) -> None:
        server = self.server
        assert isinstance(server, _CameraUnixServer)
        server.unregister_client(self.request)
        super().finish()


class _CameraUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, bridge: CameraBridgeServer) -> None:
        prepare_unix_socket(path)
        self.bridge = bridge
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        super().__init__(str(path), _CameraRequestHandler)
        path.chmod(0o600)

    def register_client(self, client: socket.socket) -> None:
        with self._clients_lock:
            self._clients.add(client)

    def unregister_client(self, client: socket.socket) -> None:
        with self._clients_lock:
            self._clients.discard(client)

    def close_clients(self) -> None:
        with self._clients_lock:
            clients = tuple(self._clients)
        for client in clients:
            with suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()


def _shape3(value: Any, *, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise ValueError(f"camera bridge {label} shape is invalid")
    return tuple(value)


def _encode_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(
            f"{label} must have shape {shape}, got {getattr(value, 'shape', None)}"
        )
    if value.dtype != dtype:
        raise ValueError(f"{label} must have dtype {dtype}, got {value.dtype}")
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(shape),
        "dtype": dtype.str,
        "data": contiguous.tobytes(),
    }


def _decode_front(value: Any, config: SystemCamerasConfig) -> CameraSample:
    payload = _sample_payload(value, label="front")
    front = config.front
    color = _decode_array(
        payload.get("color_bgr"),
        shape=(front.height, front.width, 3),
        dtype=np.dtype(np.uint8),
        label="front color",
    )
    depth_enabled = isinstance(front, SystemRealSenseCameraConfig) and front.depth
    raw_depth = payload.get("depth_mm")
    depth = None
    if depth_enabled:
        depth = _decode_array(
            raw_depth,
            shape=(front.height, front.width),
            dtype=np.dtype(np.uint16),
            label="front aligned depth",
        )
    elif raw_depth is not None:
        raise ValueError("camera bridge returned unconfigured depth")
    return CameraSample(
        seq=_plain_int(payload.get("seq"), label="front.seq"),
        monotonic_s=_finite_float(
            payload.get("monotonic_s"), label="front.monotonic_s"
        ),
        value=RealSenseFrameSet(color_bgr=color, depth_mm=depth),
    )


def _decode_wrist(value: Any, config: SystemCamerasConfig) -> CameraSample:
    payload = _sample_payload(value, label="wrist")
    wrist = config.wrist
    color = _decode_array(
        payload.get("color_bgr"),
        shape=(wrist.height, wrist.width, 3),
        dtype=np.dtype(np.uint8),
        label="wrist color",
    )
    return CameraSample(
        seq=_plain_int(payload.get("seq"), label="wrist.seq"),
        monotonic_s=_finite_float(
            payload.get("monotonic_s"), label="wrist.monotonic_s"
        ),
        value=color,
    )


def _sample_payload(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} camera bridge sample must be a map")
    return value


def _decode_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"{label} payload must be a map")
    if value.get("shape") != list(shape) or value.get("dtype") != dtype.str:
        raise ValueError(f"{label} payload shape or dtype does not match config")
    data = value.get("data")
    if not isinstance(data, bytes) or len(data) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f"{label} payload byte length does not match config")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _plain_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted
