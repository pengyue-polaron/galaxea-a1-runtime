"""LeRobot-side client for the A1-specific Runtime service."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import grpc

from ._version import __version__
from .codec import (
    health_from_proto,
    manifest_from_proto,
    values_from_proto,
    values_to_proto,
)
from .contracts import (
    HealthReport,
    HealthStatus,
    RuntimeContractError,
    RuntimeLifecycleError,
    RuntimeManifest,
    RuntimeRpcError,
    validate_feature_values,
)
from .endpoint import unix_socket_path
from .types import PROTOCOL_VERSION, SessionMode
from .v1 import a1_runtime_pb2, a1_runtime_pb2_grpc


class A1RuntimeClient:
    """Thin client; the Runtime process remains the sole A1 hardware owner."""

    def __init__(
        self,
        *,
        endpoint: str,
        mode: SessionMode = SessionMode.COMMAND,
        client_name: str = "lerobot-galaxea-a1",
        connect_timeout_s: float = 5.0,
        rpc_timeout_s: float = 2.0,
    ) -> None:
        unix_socket_path(endpoint)
        _validate_positive_timeout("connect_timeout_s", connect_timeout_s)
        _validate_positive_timeout("rpc_timeout_s", rpc_timeout_s)
        _validate_client_identity(client_name)
        if not isinstance(mode, SessionMode):
            raise RuntimeContractError(f"unsupported A1 session mode: {mode!r}")
        self.endpoint = endpoint
        self.mode = mode
        self.client_name = client_name
        self.connect_timeout_s = float(connect_timeout_s)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self._channel: grpc.Channel | None = None
        self._stub: a1_runtime_pb2_grpc.A1RuntimeServiceStub | None = None
        self._manifest: RuntimeManifest | None = None
        self._session_id: str | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._fatal_error: RuntimeRpcError | None = None
        self._command_lock = threading.Lock()
        self._next_command_sequence = 1
        self._last_command_timestamp_ns = 0

    @property
    def manifest(self) -> RuntimeManifest:
        if self._manifest is None:
            response = self._call(
                lambda stub: stub.Describe(
                    a1_runtime_pb2.DescribeRequest(protocol_version=PROTOCOL_VERSION),
                    timeout=self.connect_timeout_s,
                ),
                operation="describe",
            )
            if response.protocol_version != PROTOCOL_VERSION:
                raise RuntimeRpcError(
                    f"server selected protocol {response.protocol_version}; "
                    f"expected {PROTOCOL_VERSION}"
                )
            self._manifest = manifest_from_proto(response.manifest)
        return self._manifest

    @property
    def is_connected(self) -> bool:
        # LeRobot uses this property to permit disconnect. A failed session is
        # still locally owned until disconnect releases the client resources.
        return self._session_id is not None

    def connect(self) -> None:
        if self._session_id is not None:
            raise RuntimeLifecycleError("A1 Runtime client is already connected")
        manifest = self.manifest
        response = self._call(
            lambda stub: stub.Open(
                a1_runtime_pb2.OpenRequest(
                    protocol_version=PROTOCOL_VERSION,
                    mode=_session_mode_to_proto(self.mode),
                    client_name=self.client_name,
                    client_version=__version__,
                ),
                timeout=self.connect_timeout_s,
            ),
            operation="connect",
        )
        if not response.session_id:
            raise RuntimeRpcError("A1 Runtime returned an empty session identifier")
        try:
            opened_manifest = manifest_from_proto(response.manifest)
        except (RuntimeContractError, ValueError) as exc:
            self._best_effort_close(response.session_id)
            raise RuntimeRpcError(
                f"A1 Runtime returned an invalid session manifest: {exc}"
            ) from exc
        if opened_manifest != manifest:
            self._best_effort_close(response.session_id)
            raise RuntimeRpcError("A1 Runtime manifest changed while opening a session")
        if response.lease_timeout_ms <= 0:
            self._best_effort_close(response.session_id)
            raise RuntimeRpcError("A1 Runtime returned an invalid session lease")
        if self.mode is SessionMode.COMMAND and (
            response.command_timeout_ms <= 0
            or response.command_timeout_ms > response.lease_timeout_ms
        ):
            self._best_effort_close(response.session_id)
            raise RuntimeRpcError("A1 Runtime returned an invalid command timeout")
        self._session_id = response.session_id
        self._fatal_error = None
        self._next_command_sequence = 1
        self._last_command_timestamp_ns = 0
        # Command RPCs refresh their own session, while the server's shorter
        # command-inactivity deadline closes an idle command owner before its
        # session lease. A concurrent heartbeat would only contend with the
        # first command while it stages and activates the current-joint hold.
        if self.mode is SessionMode.OBSERVE:
            self._start_heartbeat(response.lease_timeout_ms / 1000)

    def observe(self) -> Mapping[str, float]:
        session_id = self._require_session()
        response = self._call(
            lambda stub: stub.Observe(
                a1_runtime_pb2.SessionRequest(session_id=session_id),
                timeout=self.rpc_timeout_s,
            ),
            operation="observe",
        )
        return values_from_proto(response.values, self.manifest.observation_features)

    def command(self, action: Mapping[str, object]) -> Mapping[str, float]:
        if self.mode is not SessionMode.COMMAND:
            raise RuntimeLifecycleError("A1 Runtime session does not own command access")
        session_id = self._require_session()
        requested = validate_feature_values(action, self.manifest.action_features)
        with self._command_lock:
            timestamp_ns = max(time.monotonic_ns(), self._last_command_timestamp_ns + 1)
            sequence = self._next_command_sequence
            timeout_s = self.connect_timeout_s if sequence == 1 else self.rpc_timeout_s
            try:
                response = self._call(
                    lambda stub: stub.Command(
                        a1_runtime_pb2.CommandRequest(
                            session_id=session_id,
                            sequence=sequence,
                            sent_monotonic_ns=timestamp_ns,
                            values=values_to_proto(requested, self.manifest.action_features),
                        ),
                        timeout=timeout_s,
                    ),
                    operation="command",
                )
            except RuntimeRpcError as exc:
                self._fatal_error = exc
                self._heartbeat_stop.set()
                raise
            self._next_command_sequence += 1
            self._last_command_timestamp_ns = timestamp_ns
        return values_from_proto(response.values, self.manifest.action_features)

    def health(self) -> HealthReport:
        if self._session_id is None:
            return HealthReport(HealthStatus.UNKNOWN, "A1 Runtime client is disconnected")
        session_id = self._require_session()
        response = self._call(
            lambda stub: stub.Health(
                a1_runtime_pb2.SessionRequest(session_id=session_id),
                timeout=self.rpc_timeout_s,
            ),
            operation="health",
        )
        return health_from_proto(response)

    def disconnect(self) -> None:
        session_id, self._session_id = self._session_id, None
        self._stop_heartbeat()
        error: RuntimeRpcError | None = None
        if session_id is not None:
            try:
                self._best_effort_close(session_id, suppress_errors=False)
            except RuntimeRpcError as exc:
                error = exc
        self._fatal_error = None
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None
        if error is not None:
            raise error

    def _require_session(self) -> str:
        if self._fatal_error is not None:
            raise RuntimeLifecycleError(f"A1 Runtime session failed: {self._fatal_error}")
        if self._session_id is None:
            raise RuntimeLifecycleError("A1 Runtime client is not connected")
        return self._session_id

    def _call(self, call: Callable[[Any], Any], *, operation: str) -> Any:
        try:
            return call(self._ensure_stub())
        except grpc.RpcError as exc:
            code = exc.code().name if exc.code() is not None else "UNKNOWN"
            details = exc.details() or "no server details"
            raise RuntimeRpcError(f"A1 Runtime RPC {operation} failed [{code}]: {details}") from exc

    def _ensure_stub(self) -> a1_runtime_pb2_grpc.A1RuntimeServiceStub:
        if self._stub is None:
            self._channel = grpc.insecure_channel(self.endpoint)
            self._stub = a1_runtime_pb2_grpc.A1RuntimeServiceStub(self._channel)
        return self._stub

    def _start_heartbeat(self, lease_timeout_s: float) -> None:
        self._heartbeat_stop.clear()
        interval_s = lease_timeout_s / 3
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval_s,),
            name="galaxea-a1-runtime-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self, interval_s: float) -> None:
        while not self._heartbeat_stop.wait(interval_s):
            session_id = self._session_id
            if session_id is None:
                return
            try:
                self._call(
                    lambda stub, session_id=session_id: stub.Heartbeat(
                        a1_runtime_pb2.SessionRequest(session_id=session_id),
                        timeout=min(self.rpc_timeout_s, interval_s),
                    ),
                    operation="heartbeat",
                )
            except RuntimeRpcError as exc:
                if not self._heartbeat_stop.is_set():
                    self._fatal_error = exc
                return

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self.rpc_timeout_s + 0.5)
        self._heartbeat_thread = None

    def _best_effort_close(self, session_id: str, *, suppress_errors: bool = True) -> None:
        try:
            self._call(
                lambda stub: stub.Close(
                    a1_runtime_pb2.SessionRequest(session_id=session_id),
                    timeout=self.rpc_timeout_s,
                ),
                operation="disconnect",
            )
        except RuntimeRpcError:
            if not suppress_errors:
                raise


def _validate_positive_timeout(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_client_identity(value: str) -> None:
    if not value or value.strip() != value or any(character.isspace() for character in value):
        raise RuntimeContractError(f"invalid A1 Runtime client name: {value!r}")


def _session_mode_to_proto(mode: SessionMode) -> int:
    if mode is SessionMode.OBSERVE:
        return a1_runtime_pb2.SESSION_MODE_OBSERVE
    if mode is SessionMode.COMMAND:
        return a1_runtime_pb2.SESSION_MODE_COMMAND
    raise RuntimeContractError(f"unsupported A1 Runtime session mode: {mode!r}")
