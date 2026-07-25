"""Runtime-owned LingBot EEF lowering service for OpenRAL deployments.

This process owns model-contract validation, episode-relative EEF semantics,
temporal-cache replay, and A1 IK. It never imports ROS or publishes commands;
OpenRAL remains the action safety and HAL boundary.
"""

from __future__ import annotations

import signal
import socket
import socketserver
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata
from galaxea_a1_runtime.apps.lingbot.rollout import LingBotActionChunk
from galaxea_a1_runtime.console import ArgumentParser, info, success
from galaxea_a1_runtime.hardware.eef_ik import A1EefIkSolver
from galaxea_a1_runtime.policies.eef_actions import (
    absolute_action_to_relative,
    build_action_transform_config,
    relative_action_to_absolute,
    validate_policy_action,
)
from galaxea_a1_runtime.runtime.local_ipc import (
    prepare_unix_socket,
    process_socket_path,
    receive_packet,
    send_packet,
)

PROTOCOL_VERSION = "galaxea_a1_openral_policy_v1"
_SOCKET_NAME = "a1-openral-policy.sock"
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_COLOR_SHAPE_LENGTH = 3


def openral_policy_socket_path(*, state_root: Path | None = None) -> Path:
    """Return the private local endpoint for the OpenRAL policy gateway."""

    return process_socket_path(_SOCKET_NAME, state_root=state_root)


def bounded_joint_substep(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_step_rad: float,
) -> tuple[np.ndarray, bool]:
    """Return a proportional joint target inside one declared step bound."""

    current64 = np.asarray(current, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    if (
        current64.shape != target64.shape
        or current64.ndim != 1
        or not np.isfinite(current64).all()
        or not np.isfinite(target64).all()
    ):
        raise ValueError("bounded joint substep requires matching finite vectors")
    if not np.isfinite(max_step_rad) or max_step_rad <= 0:
        raise ValueError("max_step_rad must be finite and positive")
    delta = target64 - current64
    largest = float(np.max(np.abs(delta)))
    if largest <= max_step_rad:
        return target64.copy(), True
    float32_bound = float(np.nextafter(np.float32(max_step_rad), np.float32(0.0)))
    return current64 + delta * (float32_bound / largest), False


@dataclass(frozen=True)
class OpenRalPolicyContract:
    joint_names: tuple[str, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    max_joint_step_rad: float


class LingBotOpenRalPolicy:
    """Stateful LingBot chunk replay that returns action proposals only."""

    def __init__(self, config: LingBotConfig) -> None:
        self.config = config
        self.action_config = build_action_transform_config(system=config.system)
        server = config.server
        self.client = LingBotClient(
            server.host,
            server.port,
            connect_timeout_s=server.connect_timeout_s,
            close_timeout_s=server.close_timeout_s,
            expected_metadata=server_metadata(config),
        )
        self._solver: A1EefIkSolver | None = None
        self._contract: OpenRalPolicyContract | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="a1-openral-lingbot",
        )
        self._boundary_future: Future[dict[str, Any]] | None = None
        self._boundary_deadline = 0.0
        self._terminal_error: RuntimeError | None = None
        self._instruction: str | None = None
        self._origin_pose7: np.ndarray | None = None
        self._chunk: LingBotActionChunk | None = None
        self._steps: list[tuple[int, int, int, np.ndarray]] = []
        self._step_index = 0
        self._pending_observation = False
        self._key_frames: list[dict[str, np.ndarray]] = []
        self._first = True
        self._model_calls = 0
        self._active_joint_target: np.ndarray | None = None
        self._active_gripper: float | None = None
        self._active_cache_frame_index: int | None = None
        self._active_action_index: int | None = None
        self._last_joint_target: np.ndarray | None = None
        self._last_gripper_target: float | None = None

    def describe(self) -> dict[str, Any]:
        """Describe identity without exposing Runtime configuration paths."""

        policy = self.config.policy_server
        metadata = server_metadata(self.config)
        return {
            "protocol": PROTOCOL_VERSION,
            "lingbot_contract_sha256": metadata["contract_sha256"],
            "model_repo_id": policy.model.source.repo_id,
            "model_revision": policy.model.source.revision,
            "joint_names": list(self.config.system.joint_safety.names),
            "camera_keys": [
                self.config.observations.front_key,
                self.config.observations.wrist_key,
            ],
            "camera_shapes": metadata["camera_shapes"],
            "action_dim": 7,
            "gripper_range": [0.0, 1.0],
        }

    def configure(self, value: Any) -> None:
        """Install the narrower OpenRAL command envelope before inference."""

        if not isinstance(value, dict):
            raise ValueError("OpenRAL policy configure payload must be a map")
        names_raw = value.get("joint_names")
        if not isinstance(names_raw, list) or any(
            not isinstance(name, str) for name in names_raw
        ):
            raise ValueError("OpenRAL joint_names must be a string list")
        names = tuple(names_raw)
        system_joints = self.config.system.joint_safety
        if names != system_joints.names:
            raise ValueError("OpenRAL joint names do not match the Runtime A1 contract")
        lower = _finite_vector(value.get("lower_limits"), len(names), "lower limits")
        upper = _finite_vector(value.get("upper_limits"), len(names), "upper limits")
        if np.any(lower >= upper):
            raise ValueError("OpenRAL joint limits are not ordered")
        runtime_lower = np.asarray(system_joints.lower_limits, dtype=np.float64)
        runtime_upper = np.asarray(system_joints.upper_limits, dtype=np.float64)
        if np.any(lower < runtime_lower) or np.any(upper > runtime_upper):
            raise ValueError("OpenRAL joint limits exceed the Runtime safety envelope")
        max_step = float(value.get("max_joint_step_rad"))
        if (
            not np.isfinite(max_step)
            or max_step <= 0
            or max_step > system_joints.initial_alignment_tolerance_rad
        ):
            raise ValueError(
                "OpenRAL max_joint_step_rad must be positive and no greater than "
                "the Runtime initial-alignment tolerance"
            )
        contract = OpenRalPolicyContract(
            joint_names=names,
            lower_limits=tuple(float(item) for item in lower),
            upper_limits=tuple(float(item) for item in upper),
            max_joint_step_rad=max_step,
        )
        if self._contract is not None and self._contract != contract:
            raise RuntimeError(
                "OpenRAL policy gateway is already configured differently"
            )
        ik = self.config.system.eef_ik
        self._solver = A1EefIkSolver(
            urdf_path=ik.urdf,
            joint_names=names,
            lower_limits=lower,
            upper_limits=upper,
            max_iterations=ik.max_iterations,
            damping=ik.damping,
            orientation_weight=ik.orientation_weight,
            max_iteration_step_rad=ik.max_iteration_step_rad,
            position_tolerance_m=ik.position_tolerance_m,
            orientation_tolerance_rad=ik.orientation_tolerance_rad,
            max_solution_delta_rad=ik.max_solution_delta_rad,
        )
        self._contract = contract

    def reset(self, instruction: str) -> None:
        """Reset the model and all episode-relative replay state."""

        self._raise_terminal_error()
        if not instruction.strip():
            raise ValueError("OpenRAL instruction must be non-empty")
        if self._boundary_future is not None:
            raise RuntimeError("cannot reset LingBot while cache inference is active")
        self.client.reset(instruction)
        self._instruction = instruction
        self._origin_pose7 = None
        self._chunk = None
        self._steps.clear()
        self._step_index = 0
        self._pending_observation = False
        self._key_frames.clear()
        self._first = True
        self._model_calls = 0
        self._clear_active_action()
        self._last_joint_target = None
        self._last_gripper_target = None

    def step(
        self,
        *,
        joints: np.ndarray,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        instruction: str,
        timeout_s: float,
    ) -> np.ndarray:
        """Return one absolute six-joint plus normalized-gripper proposal."""

        if self._solver is None or self._contract is None:
            raise RuntimeError("OpenRAL policy gateway must be configured before step")
        self._raise_terminal_error()
        joint_state = _finite_vector(joints, 6, "joint state")
        if self._instruction != instruction:
            self.reset(instruction)
        observation = self._model_observation(front_rgb, wrist_rgb)
        if self._origin_pose7 is None:
            xyz, quat = self._solver.forward(joint_state)
            self._origin_pose7 = np.concatenate([xyz, quat])
        if self._pending_observation:
            self._key_frames.append(observation)
            self._pending_observation = False
        boundary_hold = self._poll_chunk_boundary()
        if boundary_hold is not None:
            return boundary_hold
        if self._chunk is not None and self._step_index >= len(self._steps):
            self._start_chunk_boundary(observation, instruction, timeout_s=timeout_s)
            return self._hold_action()
        if self._chunk is None:
            self._infer_chunk(observation, instruction)
        if self._active_joint_target is None:
            _frame, action_index, cache_frame_index, raw_action = self._steps[
                self._step_index
            ]
            self._begin_action(
                raw_action,
                joints=joint_state,
                cache_frame_index=cache_frame_index,
                action_index=action_index,
            )
        return self._step_active_action(joint_state)

    def close(self) -> None:
        """Close the inference connection and worker."""

        self.client.close()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _infer_chunk(
        self,
        observation: dict[str, np.ndarray],
        instruction: str,
    ) -> None:
        limit = self.config.execution.max_model_calls
        if limit > 0 and self._model_calls >= limit:
            raise RuntimeError(
                f"LingBot rollout reached configured max_model_calls={limit}"
            )
        response = self.client.infer({"obs": [observation], "prompt": instruction})
        self._install_chunk(response, first=self._first)

    def _install_chunk(self, response: dict[str, Any], *, first: bool) -> None:
        raw = response.get("action")
        policy = self.config.policy_server
        chunk = LingBotActionChunk.from_response(
            raw,
            expected_shape=(
                len(policy.action_channel_ids),
                policy.frame_chunk_size,
                policy.action_per_frame,
            ),
            first=first,
            execute_frames=self.config.execution.execute_frames,
            observations_per_frame=self.config.execution.kv_observations_per_frame,
        )
        self._chunk = chunk
        self._steps = list(chunk.steps())
        self._step_index = 0
        self._key_frames.clear()
        self._model_calls += 1
        self._first = first

    def _start_chunk_boundary(
        self,
        observation: dict[str, np.ndarray],
        instruction: str,
        *,
        timeout_s: float,
    ) -> None:
        if self._chunk is None or self._boundary_future is not None:
            raise RuntimeError("invalid LingBot chunk-boundary state")
        if not self._key_frames:
            raise RuntimeError("LingBot chunk ended without KV-cache observations")
        limit = self.config.execution.max_model_calls
        if limit > 0 and self._model_calls >= limit:
            raise RuntimeError(
                f"LingBot rollout reached configured max_model_calls={limit}"
            )
        cache_request = {
            "obs": [
                {key: np.array(value, copy=True) for key, value in frame.items()}
                for frame in self._key_frames
            ],
            "compute_kv_cache": True,
            "imagine": False,
            "state": np.array(self._chunk.cache_state, copy=True),
        }
        next_request = {
            "obs": [
                {key: np.array(value, copy=True) for key, value in observation.items()}
            ],
            "prompt": instruction,
        }

        def infer_next() -> dict[str, Any]:
            self.client.infer(cache_request)
            return self.client.infer(next_request)

        self._boundary_future = self._executor.submit(infer_next)
        self._boundary_deadline = time.monotonic() + timeout_s
        self._first = False
        self._chunk = None
        self._steps.clear()
        self._step_index = 0
        self._key_frames.clear()

    def _poll_chunk_boundary(self) -> np.ndarray | None:
        if self._boundary_future is None:
            return None
        if not self._boundary_future.done():
            if time.monotonic() > self._boundary_deadline:
                self._terminal_error = RuntimeError(
                    "LingBot cache/inference exceeded the request timeout"
                )
                self.client.close()
                raise self._terminal_error
            return self._hold_action()
        future, self._boundary_future = self._boundary_future, None
        try:
            response = future.result()
        except Exception as exc:
            self._terminal_error = RuntimeError(
                "LingBot cache/inference failed; restart the OpenRAL policy gateway"
            )
            raise self._terminal_error from exc
        self._install_chunk(response, first=False)
        return None

    def _raise_terminal_error(self) -> None:
        if self._terminal_error is not None:
            raise self._terminal_error

    def _hold_action(self) -> np.ndarray:
        if self._last_joint_target is None or self._last_gripper_target is None:
            raise RuntimeError("LingBot cannot hold before its first validated action")
        return np.concatenate(
            [self._last_joint_target, [self._last_gripper_target]]
        ).astype(np.float32)

    def _begin_action(
        self,
        raw_action: np.ndarray,
        *,
        joints: np.ndarray,
        cache_frame_index: int,
        action_index: int,
    ) -> None:
        assert self._solver is not None
        assert self._origin_pose7 is not None
        if self.config.action.pose_mode == "absolute":
            absolute = np.asarray(raw_action, dtype=np.float64)
        else:
            absolute = relative_action_to_absolute(
                raw_action,
                self._origin_pose7,
                min_quat_norm=self.action_config.min_quat_norm,
            )
        validated = validate_policy_action(absolute, self.action_config)
        solution = self._solver.solve(joints, validated[:3], validated[3:7])
        self._active_joint_target = np.asarray(
            solution.joint_positions,
            dtype=np.float64,
        )
        self._active_gripper = float(validated[7])
        self._active_cache_frame_index = cache_frame_index
        self._active_action_index = action_index

    def _step_active_action(self, joints: np.ndarray) -> np.ndarray:
        assert self._contract is not None
        assert self._active_joint_target is not None
        assert self._active_gripper is not None
        target, reaches_solution = bounded_joint_substep(
            joints,
            self._active_joint_target,
            max_step_rad=self._contract.max_joint_step_rad,
        )
        action = np.concatenate([target, [self._active_gripper]]).astype(np.float32)
        self._last_joint_target = target.copy()
        self._last_gripper_target = self._active_gripper
        if not reaches_solution:
            return action
        assert self._solver is not None
        assert self._chunk is not None
        assert self._active_cache_frame_index is not None
        assert self._active_action_index is not None
        assert self._origin_pose7 is not None
        xyz, quat = self._solver.forward(target)
        executed = np.concatenate([xyz, quat, [self._active_gripper]])
        if self.config.action.pose_mode == "absolute":
            cache_action = executed
        else:
            cache_action = absolute_action_to_relative(
                executed,
                self._origin_pose7,
                min_quat_norm=self.action_config.min_quat_norm,
            )
        self._chunk.cache_state[
            :,
            self._active_cache_frame_index,
            self._active_action_index,
        ] = cache_action
        self._pending_observation = self._chunk.needs_observation_after(
            self._active_action_index
        )
        self._step_index += 1
        self._clear_active_action()
        return action

    def _clear_active_action(self) -> None:
        self._active_joint_target = None
        self._active_gripper = None
        self._active_cache_frame_index = None
        self._active_action_index = None

    def _model_observation(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> dict[str, np.ndarray]:
        metadata = self.describe()
        result: dict[str, np.ndarray] = {}
        for value, key, shape in zip(
            (front_rgb, wrist_rgb),
            metadata["camera_keys"],
            metadata["camera_shapes"],
            strict=True,
        ):
            image = np.asarray(value)
            if image.shape != tuple(shape) or image.dtype != np.uint8:
                raise ValueError(
                    f"OpenRAL image {key!r} must be {tuple(shape)} uint8 RGB"
                )
            result[str(key)] = image
        return result


class OpenRalPolicyGateway:
    """Length-prefixed MessagePack Unix service around one policy session."""

    def __init__(
        self,
        policy: LingBotOpenRalPolicy,
        *,
        socket_path: Path | None = None,
    ) -> None:
        self.policy = policy
        self.socket_path = socket_path or openral_policy_socket_path()
        self._server = _PolicyUnixServer(self.socket_path, self)

    def serve(self, stop_requested: threading.Event) -> None:
        self._server.timeout = 0.25
        try:
            while not stop_requested.is_set():
                self._server.handle_request()
        finally:
            self._server.close_clients()
            self._server.server_close()
            self.policy.close()
            with suppress(FileNotFoundError):
                if self.socket_path.is_socket():
                    self.socket_path.unlink()

    def close_client(self) -> None:
        """Interrupt the active OpenRAL session so the service can stop."""

        self._server.close_clients()

    def response(self, request: Any) -> dict[str, Any]:
        try:
            return self._response(request)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _response(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("OpenRAL policy request must be a map")
        if request.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("OpenRAL policy protocol mismatch")
        operation = request.get("op")
        if operation == "describe":
            return {"ok": True, "metadata": self.policy.describe()}
        if operation == "configure":
            self.policy.configure(request.get("contract"))
            return {"ok": True}
        if operation == "reset":
            self.policy.reset(_instruction(request))
            return {"ok": True}
        if operation == "step":
            timeout_s = float(request.get("timeout_s"))
            if not np.isfinite(timeout_s) or timeout_s <= 0:
                raise ValueError("OpenRAL policy timeout_s must be positive")
            action = self.policy.step(
                joints=_decode_array(
                    request.get("joints"),
                    shape=(6,),
                    dtype=np.dtype(np.float64),
                    label="joints",
                ),
                front_rgb=_decode_color(request.get("front_rgb"), label="front_rgb"),
                wrist_rgb=_decode_color(request.get("wrist_rgb"), label="wrist_rgb"),
                instruction=_instruction(request),
                timeout_s=timeout_s,
            )
            return {"ok": True, "action": _encode_array(action)}
        raise ValueError("unsupported OpenRAL policy operation")


class _PolicyRequestHandler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        super().setup()
        server = self.server
        assert isinstance(server, _PolicyUnixServer)
        self._owns_session = server.claim_client(self.request)

    def handle(self) -> None:
        server = self.server
        assert isinstance(server, _PolicyUnixServer)
        if not self._owns_session:
            with suppress(ConnectionError, OSError):
                send_packet(
                    self.request,
                    {
                        "ok": False,
                        "error": "RuntimeError: another OpenRAL client owns the policy session",
                    },
                )
            return
        try:
            while True:
                request = receive_packet(self.request, max_bytes=_MAX_REQUEST_BYTES)
                if request is None:
                    return
                send_packet(self.request, server.gateway.response(request))
        except (ConnectionError, OSError):
            return

    def finish(self) -> None:
        server = self.server
        assert isinstance(server, _PolicyUnixServer)
        if self._owns_session:
            server.release_client(self.request)
        super().finish()


class _PolicyUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, gateway: OpenRalPolicyGateway) -> None:
        prepare_unix_socket(path)
        self.gateway = gateway
        self._client: socket.socket | None = None
        self._client_lock = threading.Lock()
        super().__init__(str(path), _PolicyRequestHandler)
        path.chmod(0o600)

    def claim_client(self, client: socket.socket) -> bool:
        with self._client_lock:
            if self._client is not None:
                return False
            self._client = client
            return True

    def release_client(self, client: socket.socket) -> None:
        with self._client_lock:
            if self._client is client:
                self._client = None

    def close_clients(self) -> None:
        with self._client_lock:
            client = self._client
        if client is not None:
            with suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()


def _instruction(request: dict[str, Any]) -> str:
    value = request.get("instruction")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("OpenRAL policy instruction must be a non-empty string")
    return value


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"OpenRAL {label} must contain {length} finite values")
    return result.copy()


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "data": array.tobytes(),
    }


def _decode_color(value: Any, *, label: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"OpenRAL {label} must be an array payload")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != _COLOR_SHAPE_LENGTH
        or shape[2] != 3
    ):
        raise ValueError(f"OpenRAL {label} must be an HWC color array")
    return _decode_array(
        value,
        shape=tuple(shape),
        dtype=np.dtype(np.uint8),
        label=label,
    )


def _decode_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"shape", "dtype", "data"}:
        raise ValueError(f"OpenRAL {label} array payload is invalid")
    if value["shape"] != list(shape) or value["dtype"] != dtype.str:
        raise ValueError(f"OpenRAL {label} array contract mismatch")
    data = value["data"]
    if not isinstance(data, bytes) or len(data) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError(f"OpenRAL {label} array byte length mismatch")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Serve Runtime-owned LingBot EEF lowering to OpenRAL"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    config_path = args.config or default_config_path(repo_root)
    config = load_lingbot_config(config_path, repo_root=repo_root)
    stop_requested = threading.Event()
    policy = LingBotOpenRalPolicy(config)
    try:
        gateway = OpenRalPolicyGateway(policy)
    except BaseException:
        policy.close()
        raise

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()
        gateway.close_client()
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with suppress(OSError):
            probe.connect(str(openral_policy_socket_path()))
        probe.close()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    info(f"LingBot deployment config: {config.path}")
    success(f"OpenRAL policy gateway ready at {gateway.socket_path}")
    gateway.serve(stop_requested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
