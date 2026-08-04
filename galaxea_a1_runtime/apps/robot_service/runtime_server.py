"""Fail-closed host for the A1-specific Runtime service."""

from __future__ import annotations

import math
import os
import stat
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass

import grpc
from galaxea_a1_protocol.codec import (
    health_to_proto,
    manifest_to_proto,
    values_from_proto,
    values_to_proto,
)
from galaxea_a1_protocol.contracts import (
    RuntimeContractError,
    RuntimeDevice,
    RuntimeLifecycleError,
)
from galaxea_a1_protocol.endpoint import unix_socket_path
from galaxea_a1_protocol.types import PROTOCOL_VERSION, SessionMode
from galaxea_a1_protocol.v1 import a1_runtime_pb2, a1_runtime_pb2_grpc
from google.protobuf.empty_pb2 import Empty


@dataclass(slots=True)
class _Session:
    mode: SessionMode
    last_seen: float
    last_command: float | None = None
    last_command_sequence: int = 0
    last_command_timestamp_ns: int = 0


class _A1RuntimeService(a1_runtime_pb2_grpc.A1RuntimeServiceServicer):
    def __init__(
        self,
        device: RuntimeDevice,
        *,
        lease_timeout_s: float,
        command_timeout_s: float,
        monotonic=time.monotonic,
    ) -> None:
        self.device = device
        self.lease_timeout_s = lease_timeout_s
        self.command_timeout_s = command_timeout_s
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self._command_session_id: str | None = None
        self._closed = False
        self._fatal_error: RuntimeLifecycleError | None = None

    def Describe(self, request, context):
        self._require_protocol(request.protocol_version, context)
        return a1_runtime_pb2.DescribeResponse(
            protocol_version=PROTOCOL_VERSION,
            manifest=manifest_to_proto(self.device.manifest),
        )

    def Open(self, request, context):
        self._require_protocol(request.protocol_version, context)
        try:
            mode = _session_mode_from_proto(request.mode)
            _validate_client_identity(request.client_name, request.client_version)
            # Validate and serialize the manifest before acquiring either hardware
            # ownership or the command lease. A bad manifest must fail without
            # leaving a partially opened session behind.
            manifest = manifest_to_proto(self.device.manifest)
            with self._lock:
                if self._closed:
                    raise RuntimeLifecycleError("A1 Runtime service is stopping")
                self._expire_sessions_locked()
                if mode is SessionMode.COMMAND and self._command_session_id is not None:
                    raise _CommandLeaseUnavailable(
                        "another client already owns A1 command access"
                    )
                connected_here = not self.device.is_connected
                if connected_here:
                    self.device.connect()
                try:
                    if mode is SessionMode.COMMAND:
                        self.device.acquire_command_lease()
                except BaseExceptionGroup as exc:
                    self._fail_closed_locked(
                        "A1 command acquisition cleanup failed",
                        exc,
                    )
                    raise self._fatal_error from exc
                except Exception:
                    if connected_here and not self._sessions:
                        self._disconnect_device_locked()
                    raise
                session_id = uuid.uuid4().hex
                opened_at = self._monotonic()
                self._sessions[session_id] = _Session(
                    mode=mode,
                    last_seen=opened_at,
                    last_command=(opened_at if mode is SessionMode.COMMAND else None),
                )
                if mode is SessionMode.COMMAND:
                    self._command_session_id = session_id
                return a1_runtime_pb2.OpenResponse(
                    session_id=session_id,
                    lease_timeout_ms=round(self.lease_timeout_s * 1000),
                    manifest=manifest,
                    command_timeout_ms=round(self.command_timeout_s * 1000),
                )
        except _CommandLeaseUnavailable as exc:
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except (RuntimeContractError, RuntimeLifecycleError, ValueError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001 - RPC boundary must contain failures
            context.abort(grpc.StatusCode.INTERNAL, f"A1 Runtime connect failed: {exc}")

    def Heartbeat(self, request, context):
        try:
            with self._lock:
                self._require_session_locked(request.session_id)
            return Empty()
        except RuntimeLifecycleError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def Observe(self, request, context):
        try:
            with self._lock:
                self._require_session_locked(request.session_id)
                values = self.device.observe()
                return a1_runtime_pb2.ValuesResponse(
                    values=values_to_proto(
                        values, self.device.manifest.observation_features
                    )
                )
        except (RuntimeContractError, RuntimeLifecycleError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001 - RPC boundary must contain failures
            context.abort(grpc.StatusCode.INTERNAL, f"A1 observation failed: {exc}")

    def Command(self, request, context):
        try:
            with self._lock:
                session = self._require_session_locked(request.session_id)
                if session.mode is not SessionMode.COMMAND:
                    raise RuntimeLifecycleError(
                        "A1 Runtime session does not own command access"
                    )
                if request.sequence != session.last_command_sequence + 1:
                    raise RuntimeLifecycleError(
                        "A1 command sequence must be contiguous and increasing"
                    )
                if (
                    request.sent_monotonic_ns <= 0
                    or request.sent_monotonic_ns <= session.last_command_timestamp_ns
                ):
                    raise RuntimeLifecycleError(
                        "A1 command timestamp must be positive and increasing"
                    )
                action = values_from_proto(
                    request.values, self.device.manifest.action_features
                )
                try:
                    accepted = self.device.command(action)
                    response = a1_runtime_pb2.ValuesResponse(
                        values=values_to_proto(
                            accepted, self.device.manifest.action_features
                        )
                    )
                except Exception:
                    self._close_command_session_locked()
                    raise
                completed_at = self._monotonic()
                session.last_command_sequence = request.sequence
                session.last_command_timestamp_ns = request.sent_monotonic_ns
                session.last_seen = completed_at
                session.last_command = completed_at
                return response
        except (RuntimeContractError, RuntimeLifecycleError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001 - RPC boundary must contain failures
            context.abort(
                grpc.StatusCode.INTERNAL,
                f"A1 command failed and the session was closed: {exc}",
            )

    def Health(self, request, context):
        try:
            with self._lock:
                self._require_session_locked(request.session_id)
                return health_to_proto(self.device.health())
        except (RuntimeContractError, RuntimeLifecycleError) as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        except Exception as exc:  # noqa: BLE001 - RPC boundary must contain failures
            context.abort(grpc.StatusCode.INTERNAL, f"A1 health check failed: {exc}")

    def Close(self, request, context):
        try:
            with self._lock:
                session = self._sessions.get(request.session_id)
                if session is None:
                    return Empty()
                if session.mode is SessionMode.COMMAND:
                    self._close_command_session_locked()
                else:
                    del self._sessions[request.session_id]
                    if not self._sessions:
                        self._disconnect_device_locked()
            return Empty()
        except Exception as exc:  # noqa: BLE001 - RPC boundary must contain failures
            context.abort(grpc.StatusCode.INTERNAL, f"A1 disconnect failed: {exc}")

    def expire_sessions(self) -> None:
        with self._lock:
            if self._fatal_error is not None:
                raise self._fatal_error
            if self._closed:
                return
            self._expire_sessions_locked()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._disconnect_all_locked()

    def _require_protocol(self, version: int, context) -> None:
        if version != PROTOCOL_VERSION:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"unsupported A1 Runtime protocol {version}; expected {PROTOCOL_VERSION}",
            )

    def _require_session_locked(self, session_id: str) -> _Session:
        self._expire_sessions_locked()
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeLifecycleError("A1 Runtime session is missing or expired")
        session.last_seen = self._monotonic()
        return session

    def _expire_sessions_locked(self) -> None:
        now = self._monotonic()
        command_session = self._sessions.get(self._command_session_id or "")
        if (
            command_session is not None
            and command_session.last_command is not None
            and now - command_session.last_command > self.command_timeout_s
        ):
            self._close_command_session_locked()
            now = self._monotonic()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_seen > self.lease_timeout_s
        ]
        if self._command_session_id in expired:
            self._close_command_session_locked()
            return
        for session_id in expired:
            del self._sessions[session_id]
        if expired and not self._sessions:
            self._disconnect_device_locked()

    def _disconnect_all_locked(self) -> None:
        had_command_session = self._command_session_id is not None
        self._sessions.clear()
        self._command_session_id = None
        release_error: Exception | None = None
        if had_command_session:
            try:
                self.device.release_command_lease()
            except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
                release_error = exc
        try:
            self._disconnect_device_locked()
        except Exception as exc:
            if release_error is not None:
                raise RuntimeLifecycleError(
                    f"A1 command release failed ({release_error}); disconnect also failed ({exc})"
                ) from exc
            raise
        if release_error is not None:
            raise release_error

    def _close_command_session_locked(self) -> None:
        session_id = self._command_session_id
        if session_id is None:
            return
        self._sessions.pop(session_id, None)
        self._command_session_id = None
        try:
            self.device.release_command_lease()
        except Exception as release_error:
            # A failed release may leave ROS publishers or timers alive. Never
            # allow another command owner into this process after that point.
            self._fail_closed_locked("A1 command release failed", release_error)
            raise self._fatal_error from release_error
        if not self._sessions:
            self._disconnect_device_locked()

    def _disconnect_device_locked(self) -> None:
        try:
            if self.device.is_connected:
                self.device.disconnect()
        except Exception as exc:
            self._closed = True
            self._sessions.clear()
            self._command_session_id = None
            self._fatal_error = RuntimeLifecycleError(
                f"A1 device cleanup failed: {exc}"
            )
            raise self._fatal_error from exc

    def _fail_closed_locked(self, label: str, error: BaseException) -> None:
        self._closed = True
        self._sessions.clear()
        self._command_session_id = None
        try:
            self._disconnect_device_locked()
        except Exception as disconnect_error:  # noqa: BLE001 - preserve both failures
            self._fatal_error = RuntimeLifecycleError(
                f"{label} ({error}); disconnect also failed ({disconnect_error})"
            )
        else:
            self._fatal_error = RuntimeLifecycleError(f"{label}: {error}")


class A1RuntimeServer:
    """Serve one Runtime-owned A1 backend over a local Unix socket."""

    def __init__(
        self,
        device: RuntimeDevice,
        *,
        endpoint: str,
        lease_timeout_s: float,
        command_timeout_s: float | None = None,
        max_workers: int = 8,
    ) -> None:
        if not math.isfinite(lease_timeout_s) or lease_timeout_s < 0.05:
            raise ValueError("lease_timeout_s must be finite and at least 0.05 seconds")
        if command_timeout_s is None:
            command_timeout_s = lease_timeout_s
        if (
            not math.isfinite(command_timeout_s)
            or command_timeout_s < 0.05
            or command_timeout_s > lease_timeout_s
        ):
            raise ValueError(
                "command_timeout_s must be finite, at least 0.05 seconds, "
                "and no greater than lease_timeout_s"
            )
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.endpoint = endpoint
        self.socket_path = unix_socket_path(endpoint)
        self._service = _A1RuntimeService(
            device,
            lease_timeout_s=lease_timeout_s,
            command_timeout_s=command_timeout_s,
        )
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        a1_runtime_pb2_grpc.add_A1RuntimeServiceServicer_to_server(
            self._service, self._server
        )
        self._lease_timeout_s = lease_timeout_s
        self._stop_requested = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._watchdog_error: Exception | None = None
        self._started = False
        self._socket_identity: tuple[int, int] | None = None

    def start(self) -> None:
        if self._started or self._stop_requested.is_set():
            raise RuntimeLifecycleError(
                "A1 Runtime server instances cannot be restarted"
            )
        self._prepare_socket_parent()
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise RuntimeLifecycleError(
                f"A1 Runtime socket path already exists: {self.socket_path}"
            )
        if self._server.add_insecure_port(self.endpoint) == 0:
            raise RuntimeLifecycleError(
                f"could not bind A1 Runtime endpoint: {self.endpoint}"
            )
        self._server.start()
        try:
            socket_stat = os.lstat(self.socket_path)
        except OSError as exc:
            self._server.stop(0).wait(timeout=1.0)
            raise RuntimeLifecycleError(
                f"A1 Runtime started without creating its socket: {self.socket_path}"
            ) from exc
        if not stat.S_ISSOCK(socket_stat.st_mode):
            self._server.stop(0).wait(timeout=1.0)
            raise RuntimeLifecycleError(
                f"A1 Runtime endpoint is not a Unix socket: {self.socket_path}"
            )
        if socket_stat.st_uid != os.geteuid():
            self._server.stop(0).wait(timeout=1.0)
            raise RuntimeLifecycleError(
                f"A1 Runtime socket is not owned by the current user: {self.socket_path}"
            )
        self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
        try:
            os.chmod(self.socket_path, 0o600)
            secured_stat = os.lstat(self.socket_path)
        except OSError as exc:
            self._server.stop(0).wait(timeout=1.0)
            self._remove_owned_socket()
            raise RuntimeLifecycleError(
                f"could not secure A1 Runtime socket: {self.socket_path}"
            ) from exc
        if (
            not stat.S_ISSOCK(secured_stat.st_mode)
            or (secured_stat.st_dev, secured_stat.st_ino) != self._socket_identity
            or secured_stat.st_uid != os.geteuid()
            or stat.S_IMODE(secured_stat.st_mode) != 0o600
        ):
            self._server.stop(0).wait(timeout=1.0)
            self._remove_owned_socket()
            raise RuntimeLifecycleError(
                f"A1 Runtime socket security validation failed: {self.socket_path}"
            )
        self._started = True
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="galaxea-a1-runtime-lease-watchdog",
            daemon=True,
        )
        self._watchdog.start()

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        if not self._started:
            raise RuntimeLifecycleError("A1 Runtime server is not started")
        timed_out = self._server.wait_for_termination(timeout)
        if not timed_out and self._watchdog_error is not None:
            raise RuntimeLifecycleError(
                f"A1 Runtime lease watchdog failed: {self._watchdog_error}"
            ) from self._watchdog_error
        return timed_out

    def stop(self, grace_s: float = 0.0) -> None:
        if not self._started:
            return
        self._stop_requested.set()
        disconnect_error: Exception | None = None
        try:
            self._service.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must aggregate failures
            disconnect_error = exc
        finally:
            event = self._server.stop(grace_s)
            event.wait(timeout=max(1.0, grace_s + 1.0))
            if self._watchdog is not None:
                self._watchdog.join(timeout=1.0)
            self._watchdog = None
            self._started = False
            self._remove_owned_socket()
        if disconnect_error is not None:
            raise RuntimeLifecycleError(
                f"A1 Runtime disconnect failed while stopping: {disconnect_error}"
            ) from disconnect_error
        if self._watchdog_error is not None:
            raise RuntimeLifecycleError(
                f"A1 Runtime lease watchdog failed: {self._watchdog_error}"
            ) from self._watchdog_error

    def _watchdog_loop(self) -> None:
        interval_s = min(0.25, self._lease_timeout_s / 4)
        while not self._stop_requested.wait(interval_s):
            try:
                self._service.expire_sessions()
            except Exception as exc:  # noqa: BLE001 - watchdog failures are process-fatal
                self._watchdog_error = exc
                self._stop_requested.set()
                self._server.stop(0)
                return

    def _prepare_socket_parent(self) -> None:
        parent = self.socket_path.parent
        try:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_stat = os.lstat(parent)
        except OSError as exc:
            raise RuntimeLifecycleError(
                f"could not prepare A1 Runtime socket directory: {parent}"
            ) from exc
        if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_ISLNK(parent_stat.st_mode):
            raise RuntimeLifecycleError(
                f"A1 Runtime socket directory must be a real directory: {parent}"
            )
        if parent_stat.st_uid != os.geteuid():
            raise RuntimeLifecycleError(
                f"A1 Runtime socket directory is not owned by the current user: {parent}"
            )
        try:
            os.chmod(parent, 0o700)
            secured_stat = os.lstat(parent)
        except OSError as exc:
            raise RuntimeLifecycleError(
                f"could not secure A1 Runtime socket directory: {parent}"
            ) from exc
        if (
            not stat.S_ISDIR(secured_stat.st_mode)
            or stat.S_ISLNK(secured_stat.st_mode)
            or secured_stat.st_uid != os.geteuid()
            or stat.S_IMODE(secured_stat.st_mode) != 0o700
        ):
            raise RuntimeLifecycleError(
                f"A1 Runtime socket directory security validation failed: {parent}"
            )

    def _remove_owned_socket(self) -> None:
        try:
            socket_stat = os.lstat(self.socket_path)
        except FileNotFoundError:
            return
        identity = (socket_stat.st_dev, socket_stat.st_ino)
        if stat.S_ISSOCK(socket_stat.st_mode) and identity == self._socket_identity:
            self.socket_path.unlink()


class _CommandLeaseUnavailable(RuntimeLifecycleError):
    pass


def _session_mode_from_proto(value: int) -> SessionMode:
    if value == a1_runtime_pb2.SESSION_MODE_OBSERVE:
        return SessionMode.OBSERVE
    if value == a1_runtime_pb2.SESSION_MODE_COMMAND:
        return SessionMode.COMMAND
    raise RuntimeContractError(f"unsupported A1 Runtime session mode: {value}")


def _validate_client_identity(name: str, version: str) -> None:
    if (
        not name
        or name.strip() != name
        or any(character.isspace() for character in name)
    ):
        raise RuntimeContractError(f"invalid A1 Runtime client name: {name!r}")
    if (
        not version
        or version.strip() != version
        or any(character.isspace() for character in version)
    ):
        raise RuntimeContractError(f"invalid A1 Runtime client version: {version!r}")
