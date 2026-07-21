from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest
from lerobot_robot_galaxea_a1.runtime.contracts import (
    FeatureSpec,
    HealthReport,
    HealthStatus,
    RuntimeManifest,
)
from lerobot_robot_galaxea_a1.runtime.server import A1RuntimeServer
from lerobot_robot_galaxea_a1 import GalaxeaA1, GalaxeaA1Config
from lerobot_teleoperator_galaxea_a1_so_leader import (
    GalaxeaA1SOLeader,
    GalaxeaA1SOLeaderConfig,
)

import lerobot_teleoperator_galaxea_a1_so_leader.galaxea_a1_so_leader as leader_module

import galaxea_a1_runtime.apps.teleop.bridge as bridge_module
from galaxea_a1_runtime.apps.teleop.bridge import run_teleop_session
from galaxea_a1_runtime.apps.teleop.processors import make_a1_teleop_processors
from galaxea_a1_runtime.teleop.config import load_teleop_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/teleop/a1_so100.toml"
JOINT_KEYS = tuple(f"joint_{index}_rad" for index in range(1, 7))
CANONICAL_FEATURES = tuple(
    FeatureSpec(name, unit="rad", minimum=-10.0, maximum=10.0) for name in JOINT_KEYS
) + (FeatureSpec("gripper_normalized", minimum=0.0, maximum=1.0),)


class OfflineLeaderBus:
    def __init__(
        self,
        *,
        port: str,
        motors: dict,
        calibration: dict,
        frames: list[dict[str, float]],
    ) -> None:
        self.port = port
        self.motors = motors
        self.calibration = calibration
        self.frames = iter(frames)
        self.is_connected = False
        self.is_calibrated = True

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def disable_torque(self, **_kwargs) -> None: ...

    def configure_motors(self) -> None: ...

    def write(self, *_args, **_kwargs) -> None: ...

    def sync_read(self, register: str) -> dict[str, float]:
        assert register == "Present_Position"
        return next(self.frames)


class OfflineA1RuntimeDevice:
    def __init__(self, stop_requested: threading.Event) -> None:
        self.stop_requested = stop_requested
        self.is_connected = False
        self.actions: list[dict[str, object]] = []

    @property
    def manifest(self) -> RuntimeManifest:
        return RuntimeManifest(
            identifier="galaxea-a1",
            observation_features=CANONICAL_FEATURES,
            action_features=CANONICAL_FEATURES,
        )

    def connect(self) -> None:
        self.is_connected = True

    def observe(self) -> dict[str, float]:
        return {name: 0.0 for name in (*JOINT_KEYS, "gripper_normalized")}

    def acquire_command_lease(self) -> None:
        pass

    def release_command_lease(self) -> None:
        pass

    def command(self, action: dict[str, object]) -> dict[str, object]:
        self.actions.append(dict(action))
        if len(self.actions) >= 2:
            self.stop_requested.set()
        return action

    def health(self) -> HealthReport:
        return HealthReport(HealthStatus.HEALTHY, "offline")

    def disconnect(self) -> None:
        self.is_connected = False


class FakeLeader:
    def __init__(self, actions: list[dict[str, float]], events: list[str]) -> None:
        self.actions = iter(actions)
        self.events = events
        self.is_connected = False

    def connect(self, calibrate: bool = True) -> None:
        assert calibrate is False
        self.is_connected = True
        self.events.append("leader.connect")

    def get_action(self) -> dict[str, float]:
        self.events.append("leader.action")
        return next(self.actions)

    def disconnect(self) -> None:
        self.is_connected = False
        self.events.append("leader.disconnect")


class FakeRobot:
    def __init__(
        self,
        *,
        observation: dict[str, float],
        events: list[str],
        stop_requested: threading.Event,
        stop_after: int,
        fail_send: bool = False,
    ) -> None:
        self.observation = observation
        self.events = events
        self.stop_requested = stop_requested
        self.stop_after = stop_after
        self.fail_send = fail_send
        self.actions: list[dict[str, object]] = []
        self.is_connected = False

    def connect(self, calibrate: bool = True) -> None:
        assert calibrate is False
        self.is_connected = True
        self.events.append("robot.connect")

    def get_observation(self) -> dict[str, float]:
        self.events.append("robot.observation")
        return dict(self.observation)

    def send_action(self, action: dict[str, object]) -> dict[str, object]:
        self.events.append("robot.action")
        if self.fail_send:
            raise RuntimeError("fake command failure")
        self.actions.append(dict(action))
        if len(self.actions) >= self.stop_after:
            self.stop_requested.set()
        return action

    def disconnect(self) -> None:
        self.is_connected = False
        self.events.append("robot.disconnect")


def _leader_action(*, joints_deg: float, gripper: float) -> dict[str, float]:
    return {
        **{f"joint{index}.pos": joints_deg for index in range(6)},
        "gripper.pos": gripper,
    }


def test_tracked_bridge_wires_both_plugin_configs_and_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_teleop_config(CONFIG, repo_root=REPO)
    captured: dict[str, object] = {}
    teleop = object()
    robot = object()
    processors = object()

    def make_teleop(plugin_config: object) -> object:
        captured["teleop_config"] = plugin_config
        return teleop

    def make_robot(plugin_config: object) -> object:
        captured["robot_config"] = plugin_config
        return robot

    def run_session(**kwargs) -> int:
        captured["session"] = kwargs
        return 0

    monkeypatch.setattr(bridge_module, "make_teleoperator_from_config", make_teleop)
    monkeypatch.setattr(bridge_module, "make_robot_from_config", make_robot)
    monkeypatch.setattr(
        bridge_module, "make_a1_teleop_processors", lambda _config: processors
    )
    monkeypatch.setattr(bridge_module, "run_teleop_session", run_session)

    assert bridge_module.run(config) == 0

    teleop_config = captured["teleop_config"]
    assert isinstance(teleop_config, GalaxeaA1SOLeaderConfig)
    assert teleop_config.port == config.leader.port
    assert teleop_config.motor_write_retries == config.leader.motor_write_retries
    robot_config = captured["robot_config"]
    assert isinstance(robot_config, GalaxeaA1Config)
    assert robot_config.endpoint == config.system.robot_service.endpoint
    assert robot_config.connect_timeout_s == config.runtime.bridge_startup_timeout_s
    assert robot_config.rpc_timeout_s == config.system.robot_service.rpc_timeout_s
    session = captured["session"]
    assert isinstance(session, dict)
    assert session["teleop"] is teleop
    assert session["robot"] is robot
    assert session["processors"] is processors
    assert session["hz"] == config.bridge.hz


def test_offline_lerobot_plugins_run_one_composed_control_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_teleop_config(CONFIG, repo_root=REPO)
    stop_requested = threading.Event()
    frames = [
        {**{f"joint{index}": 0.0 for index in range(6)}, "gripper": 0.0},
        {**{f"joint{index}": 10.0 for index in range(6)}, "gripper": 26.58},
    ]
    monkeypatch.setattr(
        leader_module,
        "FeetechMotorsBus",
        lambda **kwargs: OfflineLeaderBus(**kwargs, frames=frames),
    )
    device = OfflineA1RuntimeDevice(stop_requested)
    endpoint = f"unix://{tmp_path / 'a1.sock'}"
    server = A1RuntimeServer(device, endpoint=endpoint, lease_timeout_s=1.0)
    server.start()
    try:
        leader = GalaxeaA1SOLeader(
            GalaxeaA1SOLeaderConfig(
                id="offline-leader",
                port="/dev/offline-leader",
                motor_write_retries=config.leader.motor_write_retries,
            )
        )
        robot = GalaxeaA1(
            GalaxeaA1Config(
                id="offline-a1",
                endpoint=endpoint,
            )
        )
        live_events: list[str] = []
        completed_frames = run_teleop_session(
            teleop=leader,
            robot=robot,
            processors=make_a1_teleop_processors(config),
            hz=config.bridge.hz,
            stop_requested=stop_requested,
            on_live=lambda: live_events.append("live"),
            sleep=lambda _seconds: None,
        )
    finally:
        server.stop()

    assert completed_frames == 2
    assert device.actions[0] == {
        **{name: 0.0 for name in JOINT_KEYS},
        "gripper_normalized": 0.0,
    }
    expected_delta = math.radians(10.0)
    expected_joints = tuple(
        min(upper, max(lower, sign * expected_delta))
        for sign, lower, upper in zip(
            config.bridge.mapping.sign,
            config.system.joint_safety.lower_limits,
            config.system.joint_safety.upper_limits,
            strict=True,
        )
    )
    assert tuple(device.actions[1][name] for name in JOINT_KEYS) == pytest.approx(
        expected_joints
    )
    assert device.actions[1]["gripper_normalized"] == pytest.approx(0.5)
    assert live_events == ["live"]
    assert robot.is_connected is False
    assert leader.is_connected is False


def test_offline_lerobot_loop_disconnects_both_devices_after_command_failure() -> None:
    config = load_teleop_config(CONFIG, repo_root=REPO)
    stop_requested = threading.Event()
    events: list[str] = []
    observation = {**{name: 0.0 for name in JOINT_KEYS}, "gripper_normalized": 0.0}
    leader = FakeLeader(
        [
            _leader_action(joints_deg=0.0, gripper=0.0),
        ],
        events,
    )
    robot = FakeRobot(
        observation=observation,
        events=events,
        stop_requested=stop_requested,
        stop_after=1,
        fail_send=True,
    )

    with pytest.raises(RuntimeError, match="fake command failure"):
        run_teleop_session(
            teleop=leader,
            robot=robot,
            processors=make_a1_teleop_processors(config),
            hz=config.bridge.hz,
            stop_requested=stop_requested,
            sleep=lambda _seconds: None,
        )

    assert events[-2:] == ["robot.disconnect", "leader.disconnect"]
