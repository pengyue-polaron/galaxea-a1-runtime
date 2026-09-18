"""Upstream TRAC-IK Distance adapter; only numerical IK, no ROS/hardware IO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
import weakref

import numpy as np

from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.hardware.eef_ik import (
    A1EefKinematics,
    A1EefIkTargetRejected,
    IkSolution,
    _finite_vector,
    _quat_to_matrix,
    _rotation_vector,
)


# Adapter protocol/build details, fixed independently of deployment settings.
TRAC_IK_EPSILON = 0.00001
RPC_TIMEOUT_S = 1.0
STARTUP_TIMEOUT_S = 10.0


def trac_ik_binary(system: SystemConfig) -> Path:
    return discover_repo_root(system.path) / ".cache/trac_ik/a1_trac_ik"


def verify_trac_ik_build(system: SystemConfig) -> dict:
    config = system.eef_ik
    if config.timeout_s >= RPC_TIMEOUT_S:
        raise ValueError("TRAC-IK solve timeout must be below the worker RPC timeout")
    if TRAC_IK_EPSILON >= min(
        config.position_tolerance_m, config.orientation_tolerance_rad
    ):
        raise ValueError("TRAC-IK acceptance tolerances must exceed solver epsilon")
    binary = trac_ik_binary(system)
    source = Path(__file__).with_name("native") / "trac_ik_worker.cpp"
    try:
        receipt = json.loads(binary.with_suffix(".json").read_text())
        for key, path in (("source_sha256", source), ("binary_sha256", binary)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt[key]:
                raise ValueError(f"TRAC-IK {key} differs from build receipt")
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(
            "TRAC-IK build missing/stale; run just trac-ik-setup"
        ) from exc
    image_id = receipt.get("image_id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise RuntimeError("TRAC-IK image receipt invalid; run just trac-ik-setup")
    try:
        subprocess.run(
            ["docker", "image", "inspect", image_id],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=STARTUP_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "TRAC-IK build image unavailable; check Docker and run just trac-ik-setup"
        ) from exc
    return receipt


def _read_replies(stream, replies: queue.Queue) -> None:
    try:
        for line in stream:
            replies.put(json.loads(line))
    except Exception as exc:
        replies.put(exc)
    finally:
        replies.put(EOFError("TRAC-IK worker exited"))


def _close_worker(process: subprocess.Popen, name: str) -> None:
    try:
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        subprocess.run(
            ["docker", "rm", "-f", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        process.kill()
        process.wait(timeout=3)
    finally:
        if process.stdout is not None:
            process.stdout.close()


class TracIkSolver(A1EefKinematics):
    """Use fresh measured joints as the seed and independently validate FK."""

    def __init__(self, *, system: SystemConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        config = system.eef_ik
        receipt = verify_trac_ik_build(system)
        self.receipt = receipt
        self._lock = threading.Lock()
        self._rpc_timeout = RPC_TIMEOUT_S
        self._sequence = 0
        self._replies = queue.Queue()
        name = "a1-ik-" + uuid.uuid4().hex
        # Private pipes isolate the Noetic C++ ABI from the Python/model environment.
        # The container has neither network access nor any device/command endpoint.
        self._process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--label",
                "io.galaxea.a1-runtime.managed=true",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--mount",
                f"type=bind,source={trac_ik_binary(system)},target=/a1_trac_ik,readonly",
                "--env",
                "LD_LIBRARY_PATH=/opt/ros/noetic/lib",
                "--entrypoint",
                "/a1_trac_ik",
                receipt["image_id"],
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._finalizer = weakref.finalize(self, _close_worker, self._process, name)
        self._reader = threading.Thread(
            target=_read_replies,
            args=(self._process.stdout, self._replies),
            daemon=True,
        )
        self._reader.start()
        try:
            self._send(
                {
                    "urdf": system.eef_ik.urdf.read_text(),
                    "base_link": config.base_link,
                    "tip_link": config.tip_link,
                    "joint_names": self.joint_names,
                    "lower": self.lower_limits.tolist(),
                    "upper": self.upper_limits.tolist(),
                    "timeout_s": config.timeout_s,
                    "epsilon": TRAC_IK_EPSILON,
                    # Native per-axis search envelopes enclose the norm-accepted
                    # region. Filter candidates by norm before Distance selection.
                    "bounds": [self.position_tolerance_m] * 3
                    + [self.orientation_tolerance_rad] * 3,
                    "position_tolerance": self.position_tolerance_m,
                    "orientation_tolerance": self.orientation_tolerance_rad,
                }
            )
            response = self._receive(STARTUP_TIMEOUT_S)
            if response != {"ready": True, "joint_names": list(self.joint_names)}:
                raise RuntimeError(f"Invalid TRAC-IK handshake: {response}")
        except BaseException:
            self.close()
            raise

    def _send(self, data: dict) -> None:
        if not self._finalizer.alive or self._process.poll() is not None:
            raise RuntimeError("TRAC-IK worker is unavailable")
        self._process.stdin.write(json.dumps(data, allow_nan=False) + "\n")
        self._process.stdin.flush()

    def _receive(self, timeout: float) -> dict:
        try:
            response = self._replies.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError("TRAC-IK worker response timed out") from exc
        if isinstance(response, BaseException):
            raise RuntimeError("TRAC-IK worker protocol failed") from response
        return response

    def close(self) -> None:
        self._finalizer()

    def solve(
        self,
        current_joint_positions,
        target_xyz,
        target_quat_xyzw,
        *,
        max_joint_delta_rad=None,
    ) -> IkSolution:
        start = _finite_vector(current_joint_positions, len(self.joints), "IK seed")
        xyz = _finite_vector(target_xyz, 3, "target xyz")
        quat = _finite_vector(target_quat_xyzw, 4, "target quaternion")
        rotation = _quat_to_matrix(quat)
        quat /= np.linalg.norm(quat)
        delta = self.max_solution_delta_rad
        if max_joint_delta_rad is not None:
            if not np.isfinite(max_joint_delta_rad) or max_joint_delta_rad <= 0:
                raise ValueError(
                    "IK joint delta restriction must be finite and positive"
                )
            delta = min(delta, max_joint_delta_rad)
        if np.any(start < self.lower_limits) or np.any(start > self.upper_limits):
            raise ValueError("current joint positions violate tracked limits")
        began = time.monotonic()
        with self._lock:
            self._sequence += 1
            try:
                self._send(
                    {
                        "id": self._sequence,
                        "seed": start.tolist(),
                        "pose": [*xyz.tolist(), *quat.tolist()],
                        "delta_limit": delta,
                    }
                )
                response = self._receive(self._rpc_timeout)
                if response["id"] != self._sequence:
                    raise RuntimeError("TRAC-IK response sequence mismatch")
            except BaseException:
                self.close()
                raise
        if response["code"] < 0:
            raise A1EefIkTargetRejected(
                "TRAC-IK did not converge within joint and solution-delta bounds"
            )
        q = _finite_vector(response["positions"], len(self.joints), "TRAC-IK solution")
        max_delta = float(np.max(np.abs(q - start)))
        if (
            np.any(q < self.lower_limits)
            or np.any(q > self.upper_limits)
            or max_delta > delta
        ):
            raise A1EefIkTargetRejected(
                "TRAC-IK solution violates joint or solution-delta limits"
            )
        transform = self._kinematics(q)
        position_error = float(np.linalg.norm(xyz - transform[:3, 3]))
        orientation_error = float(
            np.linalg.norm(_rotation_vector(rotation @ transform[:3, :3].T))
        )
        if (
            position_error > self.position_tolerance_m
            or orientation_error > self.orientation_tolerance_rad
        ):
            raise A1EefIkTargetRejected(
                f"TRAC-IK FK verification failed: position={position_error} orientation={orientation_error}"
            )
        # Independent KDL and Runtime FK must describe the same tip and frame.
        native_fk = _finite_vector(response["fk"], 7, "TRAC-IK FK")
        if (
            np.linalg.norm(native_fk[:3] - transform[:3, 3]) > 1e-8
            or np.linalg.norm(_quat_to_matrix(native_fk[3:]) - transform[:3, :3]) > 1e-8
        ):
            raise RuntimeError("TRAC-IK and Runtime URDF FK disagree")
        return IkSolution(
            tuple(float(value) for value in q),
            position_error,
            orientation_error,
            max_delta,
            time.monotonic() - began,
        )
