"""LeRobot-composed Galaxea A1 leader-to-runtime teleoperation."""

from __future__ import annotations

import math
import signal
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot_robot_galaxea_a1 import GalaxeaA1Config
from lerobot_teleoperator_galaxea_a1_so_leader import GalaxeaA1SOLeaderConfig

from galaxea_a1_runtime.console import info
from galaxea_a1_runtime.lerobot.hardware import make_a1_teleop_processors
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


class TeleoperatorDevice(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect(self, calibrate: bool = True) -> None: ...

    def get_action(self) -> Mapping[str, object]: ...

    def disconnect(self) -> None: ...


class RobotDevice(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect(self, calibrate: bool = True) -> None: ...

    def get_observation(self) -> Mapping[str, object]: ...

    def send_action(self, action: Mapping[str, object]) -> Mapping[str, object]: ...

    def disconnect(self) -> None: ...


Processor = Callable[[Any], Mapping[str, object]]
ProcessorSet = tuple[Processor, Processor, Processor]


def run(config: TeleopConfig) -> int:
    """Construct both LeRobot plugins and run them through the tracked processor."""

    teleop = make_teleoperator_from_config(
        GalaxeaA1SOLeaderConfig(
            id=config.leader.id,
            port=config.leader.port,
            motor_write_retries=config.leader.motor_write_retries,
        )
    )
    robot = make_robot_from_config(
        GalaxeaA1Config(
            system_config=config.system.path,
            connect_timeout_s=config.bridge.a1_state_timeout_s,
        )
    )
    processors = make_a1_teleop_processors(config)
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    try:
        run_teleop_session(
            teleop=teleop,
            robot=robot,
            processors=processors,
            hz=config.bridge.hz,
            stop_requested=stop_requested,
            on_live=lambda: info("relay ACTIVE; teleop is live"),
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def run_teleop_session(
    *,
    teleop: TeleoperatorDevice,
    robot: RobotDevice,
    processors: ProcessorSet,
    hz: float,
    stop_requested: threading.Event,
    on_live: Callable[[], None] = lambda: None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run one LeRobot processor loop and always close both device owners."""

    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("teleoperation rate must be finite and positive")
    try:
        teleop.connect(calibrate=False)
        robot.connect(calibrate=False)
        return _control_loop(
            teleop=teleop,
            robot=robot,
            processors=processors,
            hz=hz,
            stop_requested=stop_requested,
            on_live=on_live,
            monotonic=monotonic,
            sleep=sleep,
        )
    finally:
        _disconnect_devices(robot=robot, teleop=teleop, sleep=sleep)


def _control_loop(
    *,
    teleop: TeleoperatorDevice,
    robot: RobotDevice,
    processors: ProcessorSet,
    hz: float,
    stop_requested: threading.Event,
    on_live: Callable[[], None],
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    """Apply LeRobot's teleop-action then robot-action pipeline ordering."""

    teleop_action_processor, robot_action_processor, _ = processors
    period_s = 1.0 / hz
    frames = 0
    last_log = monotonic()
    frames_at_last_log = 0
    announced_live = False
    while not stop_requested.is_set():
        started = monotonic()
        observation = dict(robot.get_observation())
        raw_action = dict(teleop.get_action())
        teleop_action = dict(teleop_action_processor((raw_action, observation)))
        robot_action = dict(robot_action_processor((teleop_action, observation)))
        robot.send_action(robot_action)
        frames += 1
        if not announced_live:
            on_live()
            announced_live = True

        now = monotonic()
        if now - last_log >= 2.0:
            info(
                "LeRobot teleop processor loop "
                f"{(frames - frames_at_last_log) / (now - last_log):.1f} Hz"
            )
            last_log = now
            frames_at_last_log = frames
        if stop_requested.is_set():
            break
        sleep(max(period_s - (monotonic() - started), 0.0))
    return frames


def _disconnect_devices(
    *,
    robot: RobotDevice,
    teleop: TeleoperatorDevice,
    sleep: Callable[[float], None],
) -> None:
    """Lock the robot first, then release the serial leader owner."""

    errors: list[BaseException] = []
    if robot.is_connected:
        try:
            robot.disconnect()
            sleep(0.1)
        except BaseException as exc:
            errors.append(exc)
    if teleop.is_connected:
        try:
            teleop.disconnect()
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise BaseExceptionGroup("LeRobot teleoperation cleanup failed", errors)
