"""Serve the A1 operational device without exposing ROS to SDK adapters."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from embodied_ops.rpc import DeviceRpcServer

from galaxea_a1_runtime.configuration.system import SystemConfig, load_system_config
from galaxea_a1_runtime.console import ArgumentParser, info, success
from galaxea_a1_runtime.embodied_ops_device import A1RuntimeDevice, SessionFactory


ServerFactory = Callable[..., DeviceRpcServer]


def build_server(
    system: SystemConfig,
    *,
    session_factory: SessionFactory | None = None,
    server_factory: ServerFactory = DeviceRpcServer,
) -> DeviceRpcServer:
    """Build the service from one typed System config without opening ROS."""

    device = A1RuntimeDevice(
        system=system,
        session_factory=session_factory,
    )
    return server_factory(
        device,
        endpoint=system.embodied_ops.endpoint,
        lease_timeout_s=system.embodied_ops.lease_timeout_s,
        command_timeout_s=system.embodied_ops.command_timeout_s,
    )


def serve(system: SystemConfig, *, stop_requested: threading.Event) -> int:
    server = build_server(system)
    server.start()
    success(f"embodied-ops RPC ready at {system.embodied_ops.endpoint}")
    try:
        while not stop_requested.is_set():
            server.wait_for_termination(timeout=0.25)
    finally:
        server.stop(grace_s=system.embodied_ops.server_shutdown_timeout_s)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Serve the Galaxea A1 embodied-ops RPC")
    parser.add_argument("--system-config", type=Path, required=True)
    args = parser.parse_args(argv)
    system = load_system_config(args.system_config)
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    info(f"System config: {system.path}")
    return serve(system, stop_requested=stop_requested)


if __name__ == "__main__":
    raise SystemExit(main())
