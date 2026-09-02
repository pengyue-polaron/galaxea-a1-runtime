"""Current-user-only IPC for one supervised operator workflow."""

from __future__ import annotations

import socket
import socketserver
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from embodied_ops.operator_panel import OperatorPanelApplication

from galaxea_a1_runtime.runtime.local_ipc import (
    prepare_unix_socket,
    process_socket_path,
    receive_packet,
    send_packet,
)


OPERATOR_SESSION_PROTOCOL_VERSION = 1
_SOCKET_NAME = "a1-operator-session.sock"
_MAX_REQUEST_BYTES = 256 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class OperatorSessionUnavailable(RuntimeError):
    """The local workflow owner could not be reached."""


def operator_session_socket_path(*, state_root: Path | None = None) -> Path:
    return process_socket_path(_SOCKET_NAME, state_root=state_root)


class OperatorSessionServer:
    """Expose one OperatorPanelApplication over a private Unix socket."""

    def __init__(
        self,
        application: OperatorPanelApplication,
        *,
        socket_path: Path | None = None,
    ) -> None:
        self.application = application
        self.socket_path = socket_path or operator_session_socket_path()
        self._server = _OperatorUnixServer(self.socket_path, self)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Operator Session server was already started")
        self._thread = threading.Thread(
            target=self._serve,
            name="a1-operator-session",
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
                raise RuntimeError("Operator Session server did not stop")
        self._thread = None
        try:
            if self.socket_path.is_socket():
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def exception(self) -> BaseException | None:
        return self._error

    def response(self, request: Any) -> dict[str, Any]:
        try:
            result = self._dispatch(request)
        except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
            return {
                "schema_version": OPERATOR_SESSION_PROTOCOL_VERSION,
                "ok": False,
                "error": str(exc),
            }
        return {
            "schema_version": OPERATOR_SESSION_PROTOCOL_VERSION,
            "ok": True,
            "result": result,
        }

    def _dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("Operator Session request must be an object")
        if request.get("schema_version") != OPERATOR_SESSION_PROTOCOL_VERSION:
            raise ValueError("Operator Session protocol version mismatch")
        operation = request.get("operation")
        if operation == "status":
            _require_exact_keys(request, {"schema_version", "operation"})
            return self.application.workflow.snapshot()
        if operation == "start":
            _require_exact_keys(
                request,
                {"schema_version", "operation", "workflow", "values"},
            )
            workflow = request["workflow"]
            values = request["values"]
            if not isinstance(workflow, str) or not isinstance(values, dict):
                raise ValueError("Operator Session start requires workflow and values")
            return self.application.start({"workflow": workflow, "values": values})
        if operation == "input":
            _require_exact_keys(
                request,
                {
                    "schema_version",
                    "operation",
                    "action",
                    "run_id",
                    "input_revision",
                },
            )
            return self.application.input(
                {
                    "action": request["action"],
                    "run_id": request["run_id"],
                    "input_revision": request["input_revision"],
                }
            )
        if operation == "stop":
            _require_exact_keys(
                request,
                {"schema_version", "operation", "run_id"},
            )
            return self.application.stop({"run_id": request["run_id"]})
        raise ValueError(f"unsupported Operator Session operation: {operation!r}")

    def _serve(self) -> None:
        try:
            self._server.serve_forever(poll_interval=0.1)
        except BaseException as exc:
            self._error = exc


class OperatorSessionClient:
    def __init__(
        self,
        *,
        socket_path: Path | None = None,
        timeout_s: float = 0.25,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("Operator Session timeout must be positive")
        self.socket_path = socket_path or operator_session_socket_path()
        self.timeout_s = timeout_s

    def status(self) -> dict[str, Any]:
        return self._request("status")

    def start(self, workflow: str, values: dict[str, Any]) -> dict[str, Any]:
        return self._request("start", workflow=workflow, values=values)

    def input(
        self,
        action: str,
        *,
        run_id: str,
        input_revision: int,
    ) -> dict[str, Any]:
        return self._request(
            "input",
            action=action,
            run_id=run_id,
            input_revision=input_revision,
        )

    def stop(self, *, run_id: str) -> dict[str, Any]:
        return self._request("stop", run_id=run_id)

    def _request(self, operation: str, **payload: Any) -> dict[str, Any]:
        request = {
            "schema_version": OPERATOR_SESSION_PROTOCOL_VERSION,
            "operation": operation,
            **payload,
        }
        active_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        active_socket.settimeout(self.timeout_s)
        try:
            active_socket.connect(str(self.socket_path))
            send_packet(active_socket, request)
            response = receive_packet(active_socket, max_bytes=_MAX_RESPONSE_BYTES)
        except (ConnectionError, OSError, ValueError) as exc:
            raise OperatorSessionUnavailable(
                f"Operator Session unavailable at {self.socket_path}: {exc}"
            ) from exc
        finally:
            active_socket.close()
        if not isinstance(response, dict):
            raise OperatorSessionUnavailable(
                "Operator Session response must be an object"
            )
        if response.get("schema_version") != OPERATOR_SESSION_PROTOCOL_VERSION:
            raise OperatorSessionUnavailable(
                "Operator Session response version mismatch"
            )
        if response.get("ok") is not True:
            error = response.get("error")
            raise RuntimeError(
                error if isinstance(error, str) else "Operator Session failed"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise OperatorSessionUnavailable(
                "Operator Session result must be an object"
            )
        return result


class _OperatorRequestHandler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        super().setup()
        server = self.server
        assert isinstance(server, _OperatorUnixServer)
        server.register_client(self.request)

    def handle(self) -> None:
        server = self.server
        assert isinstance(server, _OperatorUnixServer)
        try:
            request = receive_packet(self.request, max_bytes=_MAX_REQUEST_BYTES)
            if request is not None:
                send_packet(self.request, server.owner.response(request))
        except (ConnectionError, OSError, ValueError):
            return

    def finish(self) -> None:
        server = self.server
        assert isinstance(server, _OperatorUnixServer)
        server.unregister_client(self.request)
        super().finish()


class _OperatorUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, owner: OperatorSessionServer) -> None:
        prepare_unix_socket(path)
        self.owner = owner
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        super().__init__(str(path), _OperatorRequestHandler)
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


def _require_exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError(
            "Operator Session request keys must be " + ", ".join(sorted(expected))
        )
