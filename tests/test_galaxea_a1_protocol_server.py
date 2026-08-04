from __future__ import annotations

import os
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from galaxea_a1_protocol.client import A1RuntimeClient
from galaxea_a1_protocol.contracts import (
    A1_CONTROL_FEATURE_NAMES,
    FeatureSpec,
    HealthReport,
    HealthStatus,
    RuntimeLifecycleError,
    RuntimeManifest,
    RuntimeRpcError,
)
from galaxea_a1_protocol.types import SessionMode

from galaxea_a1_runtime.apps.robot_service.runtime_server import A1RuntimeServer


@dataclass
class FakeRuntimeDevice:
    connected: bool = False
    lease_active: bool = False
    release_count: int = 0
    fail_release: bool = False

    @property
    def manifest(self) -> RuntimeManifest:
        features = tuple(FeatureSpec(name) for name in A1_CONTROL_FEATURE_NAMES)
        return RuntimeManifest(
            identifier="galaxea-a1",
            observation_features=features,
            action_features=features,
        )

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.connected = True

    def observe(self) -> Mapping[str, object]:
        return {name: 0.0 for name in A1_CONTROL_FEATURE_NAMES}

    def acquire_command_lease(self) -> None:
        self.lease_active = True

    def release_command_lease(self) -> None:
        if self.fail_release:
            raise RuntimeError("injected release failure")
        self.lease_active = False
        self.release_count += 1

    def command(self, action: Mapping[str, object]) -> Mapping[str, object]:
        return action

    def health(self) -> HealthReport:
        return HealthReport(HealthStatus.HEALTHY, "ready")

    def disconnect(self) -> None:
        self.connected = False


def test_runtime_server_owns_exclusive_command_deadman(short_socket_dir: Path) -> None:
    endpoint = f"unix://{short_socket_dir / 'leased.sock'}"
    device = FakeRuntimeDevice()
    server = A1RuntimeServer(
        device,
        endpoint=endpoint,
        lease_timeout_s=1.0,
        command_timeout_s=0.1,
    )
    owner = A1RuntimeClient(endpoint=endpoint, client_name="owner")
    observer = A1RuntimeClient(
        endpoint=endpoint,
        mode=SessionMode.OBSERVE,
        client_name="observer",
    )
    contender = A1RuntimeClient(endpoint=endpoint, client_name="contender")
    server.start()
    try:
        owner.connect()
        observer.connect()
        with pytest.raises(RuntimeRpcError, match="RESOURCE_EXHAUSTED"):
            contender.connect()
        deadline = time.monotonic() + 1.0
        while device.lease_active and time.monotonic() < deadline:
            time.sleep(0.01)
        assert device.lease_active is False
        assert observer.observe()[A1_CONTROL_FEATURE_NAMES[0]] == 0.0
        with pytest.raises(RuntimeRpcError, match="missing or expired"):
            owner.command(device.observe())
    finally:
        contender.disconnect()
        owner.disconnect()
        observer.disconnect()
        server.stop()


def test_failed_command_cleanup_is_process_fatal(short_socket_dir: Path) -> None:
    endpoint = f"unix://{short_socket_dir / 'fatal.sock'}"
    device = FakeRuntimeDevice(fail_release=True)
    server = A1RuntimeServer(
        device,
        endpoint=endpoint,
        lease_timeout_s=1.0,
        command_timeout_s=0.05,
    )
    owner = A1RuntimeClient(endpoint=endpoint, client_name="owner")
    server.start()
    owner.connect()
    try:
        with pytest.raises(RuntimeLifecycleError, match="watchdog failed"):
            server.wait_for_termination(timeout=1.0)
        assert device.lease_active is True
    finally:
        try:
            owner.disconnect()
        except RuntimeRpcError:
            pass
        with pytest.raises(RuntimeLifecycleError, match="watchdog failed"):
            server.stop()


def test_runtime_socket_is_private(short_socket_dir: Path) -> None:
    path = short_socket_dir / "private" / "a1.sock"
    server = A1RuntimeServer(
        FakeRuntimeDevice(),
        endpoint=f"unix://{path}",
        lease_timeout_s=1.0,
    )
    previous_umask = os.umask(0o022)
    try:
        server.start()
    finally:
        os.umask(previous_umask)
    try:
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        server.stop()


def test_runtime_refuses_symlink_socket_parent(short_socket_dir: Path) -> None:
    real = short_socket_dir / "real"
    real.mkdir()
    linked = short_socket_dir / "linked"
    linked.symlink_to(real, target_is_directory=True)
    server = A1RuntimeServer(
        FakeRuntimeDevice(),
        endpoint=f"unix://{linked / 'a1.sock'}",
        lease_timeout_s=1.0,
    )

    with pytest.raises(RuntimeLifecycleError, match="real directory"):
        server.start()
