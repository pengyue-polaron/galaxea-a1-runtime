"""Execute TFP-Ultra through Camera Bridge and the guarded A1 Runtime service."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from galaxea_a1_runtime.apps.tfp.action import tfp_action_to_command
from galaxea_a1_runtime.apps.tfp.config import default_config_path, load_tfp_config
from galaxea_a1_runtime.apps.tfp.config_schema import TFPConfig
from galaxea_a1_runtime.apps.tfp.verify import verify_deployment
from galaxea_a1_runtime.console import ArgumentParser, info, success, warning


class Runner(Protocol):
    def reset(self) -> None: ...
    def step(self, front_rgb: Any, wrist_rgb: Any, timestamp_seconds: float) -> Any: ...


class Cameras(Protocol):
    def read_observation(self) -> Mapping[str, Any] | None: ...
    def close(self) -> None: ...


class Robot(Protocol):
    @property
    def is_connected(self) -> bool: ...
    def connect(self) -> None: ...
    def command(self, action: Mapping[str, object]) -> Mapping[str, float]: ...
    def disconnect(self) -> None: ...


class Diagnostics(Protocol):
    def after_action(self, action_index: int, raw_action: Any) -> None: ...


def run_control_loop(
    config: TFPConfig,
    *,
    runner: Runner,
    cameras: Cameras,
    robot: Robot | None,
    stop_requested: threading.Event,
    initial_observation: Mapping[str, Any] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_live: Callable[[], None] = lambda: None,
    diagnostics: Diagnostics | None = None,
) -> int:
    if config.execution.execute and robot is None:
        raise ValueError("TFP execution requires an A1 Runtime client")
    runner.reset()
    period_s = 1.0 / config.execution.exec_rate
    last_timestamp: float | None = None
    observation = initial_observation
    announced_live = False
    actions = 0
    while not stop_requested.is_set() and (
        config.execution.max_actions == 0 or actions < config.execution.max_actions
    ):
        started = monotonic()
        if observation is None:
            observation = _wait_observation(
                config,
                cameras=cameras,
                stop_requested=stop_requested,
                monotonic=monotonic,
                sleep=sleep,
            )
        if observation is None:
            break
        timestamp = monotonic()
        if last_timestamp is not None:
            timestamp = max(timestamp, last_timestamp + 1e-9)
        raw_action = runner.step(
            observation[config.observations.front_key],
            observation[config.observations.wrist_key],
            timestamp,
        )
        command = tfp_action_to_command(raw_action, config.system)
        if config.execution.print_actions:
            info(f"TFP action {actions + 1}: {command}")
        if config.execution.execute:
            assert robot is not None
            robot.command(command)
            if not announced_live:
                on_live()
                announced_live = True
        if diagnostics is not None:
            diagnostics.after_action(actions + 1, raw_action)
        actions += 1
        last_timestamp = timestamp
        observation = None
        if not stop_requested.is_set():
            sleep(max(period_s - (monotonic() - started), 0.0))
    runner.reset()
    return actions


def _wait_observation(
    config: TFPConfig,
    *,
    cameras: Cameras,
    stop_requested: threading.Event,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> Mapping[str, Any] | None:
    deadline = monotonic() + config.execution.observation_timeout_s
    while not stop_requested.is_set() and monotonic() < deadline:
        observation = cameras.read_observation()
        if observation is not None:
            return observation
        sleep(0.005)
    if stop_requested.is_set():
        return None
    raise RuntimeError("No fresh TFP camera pair within the configured timeout")


def run(config: TFPConfig) -> int:
    verify_deployment(config)
    task = config.task_catalog.task(config.task_id)
    info(
        f"TFP task label: {task.task_id} distribution={task.distribution} "
        f"prompt={task.prompt!r} (the checkpoint itself has no language input)"
    )
    from galaxea_a1_protocol import A1RuntimeClient
    from lerobot.policies.diffusion_ltc.deployment import GalaxeaRunner
    import torch

    from galaxea_a1_runtime.apps.policy_camera import PolicyCameraSession

    runner = GalaxeaRunner(
        config.model.checkpoint,
        config.model.metadata,
        device=config.engine.device,
    )
    # Match the upstream deployment smoke: seed after model construction so
    # diffusion sampling is reproducible across process launches.
    torch.manual_seed(config.engine.seed)
    cameras: PolicyCameraSession | None = None
    robot: A1RuntimeClient | None = None
    diagnostics = None
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    try:
        if config.diagnostics.enabled:
            from galaxea_a1_runtime.apps.tfp.diagnostics import TFPDiagnostics

            diagnostics = TFPDiagnostics(config, runner)
        cameras = PolicyCameraSession(
            config.system,
            front_key=config.observations.front_key,
            wrist_key=config.observations.wrist_key,
        )
        initial = _wait_observation(
            config,
            cameras=cameras,
            stop_requested=stop_requested,
            monotonic=time.monotonic,
            sleep=time.sleep,
        )
        if config.execution.execute:
            robot = A1RuntimeClient(
                endpoint=config.system.robot_service.endpoint,
                client_name="tfp-ultra-a1",
                connect_timeout_s=config.system.robot_service.device_connect_timeout_s,
                rpc_timeout_s=config.system.robot_service.rpc_timeout_s,
            )
            robot.connect()
        else:
            warning(
                "TFP dry run: actions are validated but not sent to the A1 Runtime."
            )
        actions = run_control_loop(
            config,
            runner=runner,
            cameras=cameras,
            robot=robot,
            stop_requested=stop_requested,
            initial_observation=initial,
            on_live=lambda: success("relay ACTIVE; TFP joint policy is live"),
            diagnostics=diagnostics,
        )
        success(f"TFP rollout finished after {actions} actions")
        return 0
    finally:
        errors: list[BaseException] = []
        if diagnostics is not None:
            try:
                diagnostics.close()
            except BaseException as exc:
                errors.append(exc)
        runner.reset()
        if robot is not None and robot.is_connected:
            try:
                robot.disconnect()
            except BaseException as exc:
                errors.append(exc)
        if cameras is not None:
            try:
                cameras.close()
            except BaseException as exc:
                errors.append(exc)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if errors:
            raise BaseExceptionGroup("TFP rollout cleanup failed", errors)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = load_tfp_config(
        args.config or default_config_path(repo_root), repo_root=repo_root
    )
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
