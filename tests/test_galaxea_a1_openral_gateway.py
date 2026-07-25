from __future__ import annotations

import socket
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot.openral_gateway import (
    PROTOCOL_VERSION,
    LingBotOpenRalPolicy,
    OpenRalPolicyGateway,
    bounded_joint_substep,
)
from galaxea_a1_runtime.runtime.local_ipc import receive_packet, send_packet

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/deployments/lingbot/fruit_placement_eef.toml"


class _FakePolicy:
    def __init__(self) -> None:
        self.closed = False

    def describe(self) -> dict[str, object]:
        return {"protocol": PROTOCOL_VERSION}

    def close(self) -> None:
        self.closed = True


def test_openral_gateway_accepts_a_narrower_named_joint_contract() -> None:
    policy = LingBotOpenRalPolicy.__new__(LingBotOpenRalPolicy)
    policy.config = load_lingbot_config(CONFIG, repo_root=REPO_ROOT)
    policy._contract = None

    policy.configure(
        {
            "joint_names": [f"arm_joint{index}" for index in range(1, 7)],
            "lower_limits": [
                -2.8798,
                0.0,
                -3.3161,
                -2.8798,
                -1.6581,
                -2.8798,
            ],
            "upper_limits": [
                2.8798,
                3.1415,
                0.0,
                2.8798,
                1.6581,
                2.8798,
            ],
            "max_joint_step_rad": 0.045,
        }
    )

    assert policy._contract is not None
    assert policy._contract.max_joint_step_rad == 0.045
    assert policy._solver is not None


def test_openral_gateway_rejects_step_bound_above_runtime_alignment_limit() -> None:
    policy = LingBotOpenRalPolicy.__new__(LingBotOpenRalPolicy)
    policy.config = load_lingbot_config(CONFIG, repo_root=REPO_ROOT)
    policy._contract = None

    with pytest.raises(ValueError, match="initial-alignment tolerance"):
        policy.configure(
            {
                "joint_names": [f"arm_joint{index}" for index in range(1, 7)],
                "lower_limits": list(policy.config.system.joint_safety.lower_limits),
                "upper_limits": list(policy.config.system.joint_safety.upper_limits),
                "max_joint_step_rad": 0.051,
            }
        )


def test_openral_gateway_joint_substep_preserves_the_full_target() -> None:
    current = np.zeros(6, dtype=np.float64)
    target = np.array([0.16, -0.08, 0.04, 0.0, 0.0, 0.0])

    position, reached = bounded_joint_substep(
        current,
        target,
        max_step_rad=0.04,
    )
    assert not reached
    assert float(np.max(np.abs(position - current))) < 0.04

    for _ in range(4):
        position, reached = bounded_joint_substep(
            position,
            target,
            max_step_rad=0.04,
        )
    assert reached
    assert position == pytest.approx(target)


def test_openral_gateway_latches_chunk_boundary_failure() -> None:
    future: Future[dict[str, object]] = Future()
    future.set_exception(ValueError("inference failed"))
    policy = LingBotOpenRalPolicy.__new__(LingBotOpenRalPolicy)
    policy._boundary_future = future
    policy._terminal_error = None

    with pytest.raises(RuntimeError, match="restart the OpenRAL policy gateway"):
        policy._poll_chunk_boundary()
    with pytest.raises(RuntimeError, match="restart the OpenRAL policy gateway"):
        policy._raise_terminal_error()


def test_openral_gateway_rejects_a_competing_client_and_stops_active_session(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "openral-policy.sock"
    policy = _FakePolicy()
    gateway = OpenRalPolicyGateway(policy, socket_path=socket_path)  # type: ignore[arg-type]
    stop_requested = threading.Event()
    server_thread = threading.Thread(
        target=gateway.serve,
        args=(stop_requested,),
        daemon=False,
    )
    server_thread.start()
    owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    competitor = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        owner.connect(str(socket_path))
        send_packet(
            owner,
            {"protocol": PROTOCOL_VERSION, "op": "describe"},
        )
        response = receive_packet(owner, max_bytes=64 * 1024)
        assert isinstance(response, dict)
        assert response["ok"] is True

        competitor.settimeout(2.0)
        competitor.connect(str(socket_path))
        rejected = receive_packet(competitor, max_bytes=64 * 1024)
        assert isinstance(rejected, dict)
        assert rejected["ok"] is False
        assert "another OpenRAL client" in rejected["error"]
    finally:
        stop_requested.set()
        gateway.close_client()
        owner.close()
        competitor.close()
        server_thread.join(timeout=2.0)

    assert not server_thread.is_alive()
    assert policy.closed
    assert not socket_path.exists()
