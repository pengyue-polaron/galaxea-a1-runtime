#!/usr/bin/env python3
# ruff: noqa: E402
"""Move A1 and the SO leader to the tracked collection start pose."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import threading
import time
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
    module=r"torch\.cuda.*",
)

ROOT_DIR = Path(__file__).resolve().parents[3]
_A1_SDK = ROOT_DIR / "third_party" / "A1_SDK" / "install"
_ROS1_OVERLAY = ROOT_DIR / ".cache" / "ros1_python_overlay"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for candidate in (
    str(ROOT_DIR / "third_party" / "lerobot" / "src"),
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
    str(_ROS1_OVERLAY),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.teleop.a1_so_leader import A1SOLeader, SOLeaderTeleopConfig


DEFAULT_CONFIG = ROOT_DIR / "configs" / "poses" / "a1_so100_collection_start.toml"


@dataclass(frozen=True)
class HomeTopics:
    joint_states: str
    target: str
    staged_command: str
    relay_enable: str
    relay_status: str
    gripper_command: str


@dataclass(frozen=True)
class HomeMotion:
    hz: float
    min_duration_s: float
    max_velocity_rad_s: float
    tracker_alignment_timeout_s: float
    tracker_alignment_tolerance_rad: float
    relay_enable_timeout_s: float
    max_relay_status_age_s: float
    hold_s: float
    goal_tolerance_rad: float


@dataclass(frozen=True)
class A1GripperHome:
    enabled: bool
    closed_stroke_mm: float
    publish_hz: float
    publish_s: float


@dataclass(frozen=True)
class LeaderHome:
    enabled: bool
    port: str
    id: str
    use_degrees: bool
    action: dict[str, float]


@dataclass(frozen=True)
class LeaderMotion:
    hz: float
    min_duration_s: float
    max_velocity_units_s: float
    hold_s: float
    goal_tolerance_units: float
    gripper_goal_tolerance_units: float


@dataclass(frozen=True)
class HomePose:
    path: Path
    names: tuple[str, ...]
    positions: tuple[float, ...]
    topics: HomeTopics
    motion: HomeMotion
    gripper: A1GripperHome
    leader: LeaderHome | None
    leader_motion: LeaderMotion | None


class Latest:
    def __init__(self):
        self._lock = threading.Lock()
        self.value: Any | None = None
        self.updated: float | None = None

    def set(self, value: Any) -> None:
        with self._lock:
            self.value = value
            self.updated = time.monotonic()

    def get(self) -> tuple[Any | None, float | None]:
        with self._lock:
            return self.value, self.updated


class ResetProgress:
    def __init__(self, devices: tuple[str, ...]):
        self.devices = devices
        self.values = {device: 0 for device in devices}
        self.reported = {device: -1 for device in devices}
        self.lock = threading.Lock()
        self.interactive = sys.stdout.isatty()
        self.color = self.interactive and not os.environ.get("NO_COLOR")

    def update(self, device: str, percent: float) -> None:
        value = max(0, min(100, int(round(percent))))
        with self.lock:
            if value == self.reported[device]:
                return
            self.values[device] = value
            self.reported[device] = value
            if self.interactive:
                status = " | ".join(f"{name} {self.values[name]:3d}%" for name in self.devices)
                prefix = "\033[1;36mReset\033[0m" if self.color else "Reset"
                print(f"\r\033[2K{prefix}  {status}", end="", flush=True)
            elif value in {0, 25, 50, 75, 100}:
                print(f"[Reset] {device} {value}%", flush=True)

    def finish(self, *, success: bool) -> None:
        if self.interactive:
            print("\r\033[2K", end="")
        text = "[Reset] Complete" if success else "[Reset] Failed"
        if self.color:
            code = "\033[1;32m" if success else "\033[1;31m"
            text = f"{code}{text}\033[0m"
        print(text, flush=True)


class A1HomeRunner:
    def __init__(self, pose: HomePose, progress: ResetProgress):
        self.pose = pose
        self.progress = progress
        self.joints = Latest()
        self.staged = Latest()
        self.relay = Latest()
        rospy.init_node("a1_return_home", anonymous=False, disable_signals=True)
        self.target_pub = rospy.Publisher(pose.topics.target, JointState, queue_size=1)
        self.gripper_pub = rospy.Publisher(
            pose.topics.gripper_command,
            gripper_position_control,
            queue_size=1,
        )
        self.enable_pub = rospy.Publisher(pose.topics.relay_enable, Bool, queue_size=1, latch=True)
        rospy.Subscriber(pose.topics.joint_states, JointState, self.joints.set, queue_size=1)
        rospy.Subscriber(pose.topics.staged_command, arm_control, self.staged.set, queue_size=1)
        rospy.Subscriber(pose.topics.relay_status, String, self._relay_cb, queue_size=1)

    def _relay_cb(self, msg: String) -> None:
        try:
            self.relay.set(json.loads(msg.data))
        except json.JSONDecodeError:
            self.relay.set({"state": "UNKNOWN", "raw": msg.data})

    def run_a1(self) -> None:
        motion = self.pose.motion
        rate = rospy.Rate(motion.hz)
        self.enable_pub.publish(Bool(data=False))
        current = self.wait_for_joints()
        self.progress.update("A1", 0)
        self.close_a1_gripper()

        self.wait_for_staged_alignment(
            current,
            timeout_s=motion.tracker_alignment_timeout_s,
            tolerance_rad=motion.tracker_alignment_tolerance_rad,
            rate=rate,
        )
        self.enable_pub.publish(Bool(data=True))
        self.wait_for_relay_state("ACTIVE", timeout_s=motion.relay_enable_timeout_s)

        try:
            self.move_smooth(current, rate)
            final = self.hold_target(rate)
            self.close_a1_gripper()
            err = max_error(final, self.pose.positions)
            if err > motion.goal_tolerance_rad:
                raise RuntimeError(
                    f"Reset pose error {err:.4f} rad exceeds tolerance "
                    f"{motion.goal_tolerance_rad:.4f} rad"
                )
            self.progress.update("A1", 100)
        finally:
            self.enable_pub.publish(Bool(data=False))
            time.sleep(0.2)

    def wait_for_joints(self, timeout_s: float = 10.0) -> tuple[float, ...]:
        deadline = time.monotonic() + timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            msg, _ = self.joints.get()
            positions = joint_positions(msg, self.pose.names)
            if positions is not None and all(math.isfinite(value) for value in positions):
                return positions
            time.sleep(0.05)
        raise RuntimeError(f"Reset has no usable joint feedback on {self.pose.topics.joint_states}")

    def wait_for_staged_alignment(
        self,
        target: tuple[float, ...],
        *,
        timeout_s: float,
        tolerance_rad: float,
        rate: Any,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_error: float | None = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.publish_target(target)
            staged = staged_positions(self.staged.get()[0], len(target))
            if staged is not None:
                last_error = max_error(staged, target)
                if last_error <= tolerance_rad:
                    return
            rate.sleep()
        detail = "no staged command" if last_error is None else f"last error {last_error:.4f} rad"
        raise RuntimeError(
            f"Tracker did not align within {timeout_s:.1f}s "
            f"({detail}, tolerance {tolerance_rad:.4f} rad)"
        )

    def wait_for_relay_state(self, state: str, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last: Any = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            last, updated = self.relay.get()
            if (
                isinstance(last, dict)
                and last.get("state") == state
                and is_fresh(updated, self.pose.motion.max_relay_status_age_s)
            ):
                return
            if isinstance(last, dict) and last.get("state") == "FAULT":
                break
            time.sleep(0.05)
        raise RuntimeError(f"Relay did not reach {state}: last={last}")

    def move_smooth(self, start: tuple[float, ...], rate: Any) -> None:
        target = self.pose.positions
        motion = self.pose.motion
        max_delta = max_error(start, target)
        duration_s = max(motion.min_duration_s, max_delta / motion.max_velocity_rad_s)
        steps = max(1, int(duration_s * motion.hz))
        for step in range(steps + 1):
            alpha = step / steps
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            command = tuple(start[i] + (target[i] - start[i]) * smooth for i in range(len(target)))
            self.publish_target(command)
            self.progress.update("A1", alpha * 100.0)
            relay, updated = self.relay.get()
            if not (
                isinstance(relay, dict)
                and relay.get("state") == "ACTIVE"
                and is_fresh(updated, motion.max_relay_status_age_s)
            ):
                raise RuntimeError(f"Relay left ACTIVE while homing: {relay}")
            rate.sleep()

    def hold_target(self, rate: Any) -> tuple[float, ...]:
        deadline = time.monotonic() + self.pose.motion.hold_s
        final = self.wait_for_joints(timeout_s=1.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.publish_target(self.pose.positions)
            final = self.wait_for_joints(timeout_s=0.5)
            if max_error(final, self.pose.positions) <= self.pose.motion.goal_tolerance_rad:
                return final
            rate.sleep()
        return final

    def publish_target(self, target: tuple[float, ...]) -> None:
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.name = list(self.pose.names)
        msg.position = list(target)
        self.target_pub.publish(msg)

    def close_a1_gripper(self) -> None:
        gripper = self.pose.gripper
        if not gripper.enabled:
            return
        rate = rospy.Rate(gripper.publish_hz)
        deadline = time.monotonic() + gripper.publish_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            msg = gripper_position_control()
            msg.header.stamp = rospy.Time.now()
            msg.gripper_stroke = float(gripper.closed_stroke_mm)
            self.gripper_pub.publish(msg)
            rate.sleep()


def load_home_pose(path: Path) -> HomePose:
    path = path.expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    data = tomllib.loads(path.read_text())
    names = tuple(_required_list(data["joints"], "names", str))
    positions = tuple(float(value) for value in _required_list(data["joints"], "position_rad", float))
    if len(names) != len(positions):
        raise ValueError("joints.names and joints.position_rad must have the same length")
    topics = data["topics"]
    motion = data["motion"]
    leader = _load_leader_home(data)
    home = HomePose(
        path=path,
        names=names,
        positions=positions,
        topics=HomeTopics(
            joint_states=_topic(topics, "joint_states"),
            target=_topic(topics, "target"),
            staged_command=_topic(topics, "staged_command"),
            relay_enable=_topic(topics, "relay_enable"),
            relay_status=_topic(topics, "relay_status"),
            gripper_command=_topic(topics, "gripper_command"),
        ),
        motion=HomeMotion(
            hz=float(motion["hz"]),
            min_duration_s=float(motion["min_duration_s"]),
            max_velocity_rad_s=float(motion["max_velocity_rad_s"]),
            tracker_alignment_timeout_s=float(motion["tracker_alignment_timeout_s"]),
            tracker_alignment_tolerance_rad=float(motion["tracker_alignment_tolerance_rad"]),
            relay_enable_timeout_s=float(motion["relay_enable_timeout_s"]),
            max_relay_status_age_s=float(motion["max_relay_status_age_s"]),
            hold_s=float(motion["hold_s"]),
            goal_tolerance_rad=float(motion["goal_tolerance_rad"]),
        ),
        gripper=_load_a1_gripper_home(data),
        leader=leader[0],
        leader_motion=leader[1],
    )
    validate_home_pose(home)
    return home


def validate_home_pose(home: HomePose) -> None:
    if not home.names:
        raise ValueError("home pose must contain at least one joint")
    if any(not name for name in home.names):
        raise ValueError("joint names must be non-empty")
    if any(not math.isfinite(value) for value in home.positions):
        raise ValueError("joint positions must be finite")
    for field, value in home.motion.__dict__.items():
        if value <= 0:
            raise ValueError(f"motion.{field} must be positive")
    if home.gripper.closed_stroke_mm < 0:
        raise ValueError("gripper.closed_stroke_mm must be non-negative")
    if home.gripper.publish_hz <= 0 or home.gripper.publish_s <= 0:
        raise ValueError("gripper publish_hz/publish_s must be positive")
    if home.leader is not None:
        if not home.leader.port:
            raise ValueError("leader.port must be non-empty")
        if not home.leader.action:
            raise ValueError("leader.action must contain at least one target")
    if home.leader_motion is not None:
        for field, value in home.leader_motion.__dict__.items():
            if value <= 0:
                raise ValueError(f"leader_motion.{field} must be positive")


def reset_leader_home(home: HomePose, progress: ResetProgress) -> None:
    if home.leader is None or not home.leader.enabled:
        progress.update("Leader", 100)
        return
    if home.leader_motion is None:
        raise RuntimeError("leader_motion is required when leader reset is enabled")

    leader_home = home.leader
    motion = home.leader_motion
    leader = A1SOLeader(
        SOLeaderTeleopConfig(
            id=leader_home.id,
            port=leader_home.port,
            use_degrees=leader_home.use_degrees,
        )
    )
    leader.connect(calibrate=False)
    try:
        current = {key: float(value) for key, value in leader.get_action().items()}
        target = leader_home.action
        missing = sorted(key for key in target if key not in current)
        if missing:
            raise RuntimeError(f"leader action missing keys: {missing}")

        start = {key: current[key] for key in target}
        progress.update("Leader", 0)
        leader.enable_torque()
        move_leader_smooth(leader, start, target, motion, progress)
        final = {key: float(value) for key, value in leader.get_action().items() if key in target}
        errors = mapping_errors(final, target)
        body_error = max(
            (error for key, error in errors.items() if key != "gripper.pos"),
            default=0.0,
        )
        gripper_error = errors.get("gripper.pos", 0.0)
        if body_error > motion.goal_tolerance_units:
            raise RuntimeError(
                f"Leader body reset error {body_error:.3f} exceeds tolerance "
                f"{motion.goal_tolerance_units:.3f}"
            )
        if gripper_error > motion.gripper_goal_tolerance_units:
            raise RuntimeError(
                f"Leader gripper reset error {gripper_error:.3f} exceeds tolerance "
                f"{motion.gripper_goal_tolerance_units:.3f}"
            )
        progress.update("Leader", 100)
    finally:
        try:
            leader.disable_torque()
        finally:
            leader.disconnect()


def move_leader_smooth(
    leader: A1SOLeader,
    start: dict[str, float],
    target: dict[str, float],
    motion: LeaderMotion,
    progress: ResetProgress,
) -> None:
    max_delta = max(abs(target[key] - start[key]) for key in target)
    duration_s = max(motion.min_duration_s, max_delta / motion.max_velocity_units_s)
    steps = max(1, int(duration_s * motion.hz))
    for step in range(steps + 1):
        alpha = step / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        command = {
            key: start[key] + (target[key] - start[key]) * smooth
            for key in target
        }
        leader.send_feedback(command)
        progress.update("Leader", alpha * 100.0)
        time.sleep(1.0 / motion.hz)

    deadline = time.monotonic() + motion.hold_s
    while time.monotonic() < deadline:
        leader.send_feedback(target)
        time.sleep(1.0 / motion.hz)


def joint_positions(msg: Any, ordered_names: tuple[str, ...]) -> tuple[float, ...] | None:
    if msg is None or len(getattr(msg, "position", ())) < len(ordered_names):
        return None
    names = list(getattr(msg, "name", ()))
    values = list(getattr(msg, "position", ()))
    name_to_index = {name: index for index, name in enumerate(names)}
    if names and all(name in name_to_index for name in ordered_names):
        indices = [name_to_index[name] for name in ordered_names]
        if all(index < len(values) for index in indices):
            return tuple(float(values[index]) for index in indices)
    return tuple(float(value) for value in values[: len(ordered_names)])


def staged_positions(msg: Any, count: int) -> tuple[float, ...] | None:
    if msg is None or len(getattr(msg, "p_des", ())) < count:
        return None
    return tuple(float(value) for value in msg.p_des[:count])


def max_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError(f"length mismatch: {len(left)} != {len(right)}")
    return max(abs(left[i] - right[i]) for i in range(len(left)))


def is_fresh(updated: float | None, max_age_s: float) -> bool:
    return updated is not None and time.monotonic() - updated <= max_age_s


def format_values(values: tuple[float, ...]) -> list[str]:
    return [f"{value:.4f}" for value in values]


def format_mapping(values: dict[str, float]) -> dict[str, str]:
    return {key: f"{values[key]:.3f}" for key in sorted(values)}


def max_mapping_error(left: dict[str, float], right: dict[str, float]) -> float:
    return max(mapping_errors(left, right).values())


def mapping_errors(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    missing = sorted(key for key in right if key not in left)
    if missing:
        raise ValueError(f"missing keys: {missing}")
    return {key: abs(left[key] - right[key]) for key in right}


def _load_leader_home(data: dict[str, Any]) -> tuple[LeaderHome | None, LeaderMotion | None]:
    leader_data = data.get("leader")
    if leader_data is None:
        return None, None
    if not isinstance(leader_data, dict):
        raise ValueError("leader must be a table")
    enabled = bool(leader_data.get("enabled", True))
    keys = _required_list(leader_data, "action_keys", str)
    values = _required_number_list(leader_data, "action_values")
    if len(keys) != len(values):
        raise ValueError("leader.action_keys and leader.action_values must have the same length")

    motion_data = data.get("leader_motion")
    if not isinstance(motion_data, dict):
        raise ValueError("leader_motion table is required when leader is configured")
    return (
        LeaderHome(
            enabled=enabled,
            port=_required_string(leader_data, "port"),
            id=_required_string(leader_data, "id"),
            use_degrees=bool(leader_data.get("use_degrees", True)),
            action=dict(zip(keys, (float(value) for value in values), strict=True)),
        ),
        LeaderMotion(
            hz=float(motion_data["hz"]),
            min_duration_s=float(motion_data["min_duration_s"]),
            max_velocity_units_s=float(motion_data["max_velocity_units_s"]),
            hold_s=float(motion_data["hold_s"]),
            goal_tolerance_units=float(motion_data["goal_tolerance_units"]),
            gripper_goal_tolerance_units=float(
                motion_data.get(
                    "gripper_goal_tolerance_units",
                    motion_data["goal_tolerance_units"],
                )
            ),
        ),
    )


def _load_a1_gripper_home(data: dict[str, Any]) -> A1GripperHome:
    gripper = data.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("gripper table is required")
    return A1GripperHome(
        enabled=bool(gripper.get("enabled", True)),
        closed_stroke_mm=float(gripper["closed_stroke_mm"]),
        publish_hz=float(gripper["publish_hz"]),
        publish_s=float(gripper["publish_s"]),
    )


def _required_list(data: dict[str, Any], key: str, item_type: type) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, item_type) for item in value):
        raise ValueError(f"{key} must be a list of {item_type.__name__}")
    return value


def _required_number_list(data: dict[str, Any], key: str) -> list[float | int]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, int | float) for item in value):
        raise ValueError(f"{key} must be a number list")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _topic(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"topics.{key} must be an absolute ROS topic")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset A1 and leader to the tracked start pose.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    pose = load_home_pose(parse_args().config)
    devices = ("A1", "Leader") if pose.leader is not None and pose.leader.enabled else ("A1",)
    progress = ResetProgress(devices)
    runner = A1HomeRunner(pose, progress)
    jobs = {"A1": runner.run_a1}
    if pose.leader is not None and pose.leader.enabled:
        jobs["leader"] = lambda: reset_leader_home(pose, progress)

    errors: list[tuple[str, BaseException]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(job): name for name, job in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except BaseException as exc:
                errors.append((name, exc))
    if errors:
        progress.finish(success=False)
        details = "; ".join(f"{name}: {exc}" for name, exc in errors)
        raise RuntimeError(f"Reset failed ({details})") from errors[0][1]
    progress.finish(success=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
