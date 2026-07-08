#!/usr/bin/env python3
# ruff: noqa: E402
"""LingBot-VA -> Galaxea A1 end-effector pose bridge.

This is intentionally dry-run by default. Pass --execute to publish commands to
/a1_ee_target and /gripper_position_control_host.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path

# Keep ROS1 ahead of any ROS2 paths, and expose A1 custom messages.
ROOT_DIR = Path(__file__).resolve().parents[3]
_A1_SDK = ROOT_DIR / "third_party" / "A1_SDK" / "install"
_A1_SDK_RUNTIME = ROOT_DIR / "third_party" / "A1_SDK_runtime" / "install"
_ROS1_OVERLAY = ROOT_DIR / ".cache" / "ros1_python_overlay"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for p in list(sys.path):
    if "/opt/ros/humble" in p:
        sys.path.remove(p)
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK_RUNTIME / "lib" / "python3" / "dist-packages"),
    str(_ROS1_OVERLAY),
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import cv2
import msgpack
import numpy as np
import websockets.sync.client

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

import rospy
from galaxea_a1_runtime.apps.eef_bridge import (
    EefCommandPublisher,
    RelayStatus,
    condition_state_from_action8,
    decode_relay_status,
    format_xyz_direction,
    pose_msg_to_xyz_quat,
    relay_state_summary,
    relay_status_is_fresh,
)
from galaxea_a1_runtime.apps.lingbot.actions import (
    LingBotActionConfig,
    clamp_notes,
    gripper_norm_from_stroke,
    gripper_stroke_from_norm,
    normalize_condition_action,
    prepare_policy_action,
    sanitize_policy_action,
    tracker_command_action,
)
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class LingBotClient:
    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.packer = Packer()
        print(f"[LingBot] Connecting to {self.uri} ...")
        self.ws = websockets.sync.client.connect(
            self.uri, compression=None, max_size=None, ping_interval=None, close_timeout=10
        )
        self.metadata = unpackb(self.ws.recv())
        print(f"[LingBot] Connected. metadata={self.metadata}")

    def infer(self, obs: dict) -> dict:
        self.ws.send(self.packer.pack(obs))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return unpackb(response)

    def reset(self, prompt: str) -> None:
        self.infer({"reset": True, "prompt": prompt})


class OpenCVCamera:
    def __init__(self, device: str, width: int, height: int, fps: int, backend_api: str = "v4l2"):
        source = int(device) if str(device).isdigit() else str(device)
        api = cv2.CAP_V4L2 if backend_api == "v4l2" else 0
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device={device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(10):
            self.cap.read()

    def read_rgb(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class RealSenseCamera:
    def __init__(
        self,
        serial: str | None,
        width: int,
        height: int,
        fps: int,
        auto_exposure: bool = True,
        exposure: int = 140,
        gain: int = 32,
        auto_white_balance: bool = True,
        white_balance: int = 4600,
    ):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed in this environment")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(cfg)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1 if auto_exposure else 0)
        if not auto_exposure:
            color_sensor.set_option(rs.option.exposure, float(exposure))
            color_sensor.set_option(rs.option.gain, float(gain))
        if color_sensor.supports(rs.option.enable_auto_white_balance):
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1 if auto_white_balance else 0)
        if not auto_white_balance and color_sensor.supports(rs.option.white_balance):
            color_sensor.set_option(rs.option.white_balance, float(white_balance))
        for _ in range(30):
            self.pipeline.wait_for_frames()

    def read_rgb(self) -> np.ndarray | None:
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        bgr = np.asanyarray(color.get_data())
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.pipeline.stop()


class A1LingBotEEBridge:
    def __init__(self, args):
        self.args = args
        self.latest_pose = None
        self.latest_pose_monotonic = None
        self.latest_gripper = None
        self.motion_enabled = False
        self.relay_status = None
        self.relay_status_monotonic = None
        self.reported_condition_state = False
        self.action_config = self._action_config_from_args(args)
        rospy.init_node("lingbot_va_ee_bridge", anonymous=False)
        pose_pub = rospy.Publisher(args.cmd_pose_topic, PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher(args.cmd_gripper_topic, gripper_position_control, queue_size=10)
        motion_enable_pub = rospy.Publisher(args.motion_enable_topic, Bool, queue_size=1, latch=True)
        self.commander = EefCommandPublisher(
            rospy=rospy,
            pose_pub=pose_pub,
            gripper_pub=gripper_pub,
            motion_enable_pub=motion_enable_pub,
            pose_msg_type=PoseStamped,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            command_frame=args.command_frame,
            gripper_to_stroke=lambda value: gripper_stroke_from_norm(value, self.action_config),
            execute=args.execute,
        )
        rospy.Subscriber(args.state_pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber(args.state_gripper_topic, JointState, self._gripper_cb, queue_size=1)
        rospy.Subscriber(args.relay_status_topic, String, self._relay_status_cb, queue_size=1)
        self.pose_keepalive_timer = rospy.Timer(rospy.Duration(0.05), self._publish_active_pose_target)

        self.cam_high = RealSenseCamera(
            args.cam0_serial,
            args.cam_width,
            args.cam_height,
            args.cam_fps,
            auto_exposure=args.cam0_auto_exposure,
            exposure=args.cam0_exposure,
            gain=args.cam0_gain,
            auto_white_balance=args.cam0_auto_white_balance,
            white_balance=args.cam0_white_balance,
        )
        self.cam_wrist = OpenCVCamera(args.cam1_device, args.cam_width, args.cam_height, args.cam_fps, args.cam1_backend_api)
        self.client = LingBotClient(args.host, args.port)
        self.client.reset(args.prompt)

    @staticmethod
    def _action_config_from_args(args) -> LingBotActionConfig:
        return LingBotActionConfig(
            xyz_min=tuple(float(v) for v in args.xyz_min),
            xyz_max=tuple(float(v) for v in args.xyz_max),
            min_quat_norm=float(args.min_quat_norm),
            orientation_mode=args.orientation_mode,
            eef_servo_gain=float(args.eef_servo_gain),
            eef_servo_max_extra=float(args.eef_servo_max_extra),
            gripper_stroke_scale=float(args.gripper_stroke_scale),
            gripper_stroke_offset=float(args.gripper_stroke_offset),
            gripper_stroke_min=float(args.gripper_stroke_min),
            gripper_stroke_max=float(args.gripper_stroke_max),
        )

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose = msg
        self.latest_pose_monotonic = time.monotonic()

    def _gripper_cb(self, msg: JointState):
        if msg.position:
            self.latest_gripper = float(msg.position[0])

    def _relay_status_cb(self, msg: String):
        self.relay_status = decode_relay_status(msg.data)
        self.relay_status_monotonic = time.monotonic()

    def _set_active_pose_target(self, msg: PoseStamped) -> None:
        self.commander.set_active_pose_target_from_msg(msg)

    def _publish_active_pose_target(self, _event=None) -> None:
        self.commander.publish_active_pose_target()

    def _hold_current_pose(self) -> None:
        if self.latest_pose is None:
            raise RuntimeError("Cannot hold EE pose before receiving feedback")
        self._set_active_pose_target(self.latest_pose)
        self._publish_active_pose_target()

    def _relay_status_is_fresh(self) -> bool:
        return relay_status_is_fresh(
            self.relay_status_monotonic,
            max_age_s=self.args.max_relay_status_age,
        )

    def _relay_state_summary(self) -> str:
        return relay_state_summary(
            self.relay_status,
            self.relay_status_monotonic,
            max_age_s=self.args.max_relay_status_age,
        )

    def _enable_motion(self) -> None:
        if self.motion_enabled:
            status = self.relay_status or RelayStatus(state="UNKNOWN")
            if self._relay_status_is_fresh() and status.state == "ACTIVE":
                return
            self.motion_enabled = False
            self.commander.publish_motion_enable(False)
            raise RuntimeError(
                "A1 relay is no longer confirmed ACTIVE; refusing to publish "
                f"more commands. Last relay state: {self._relay_state_summary()}"
            )
        self.commander.publish_motion_enable(True)
        deadline = time.monotonic() + self.args.relay_enable_timeout
        last_state = "no status"
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status = self.relay_status or RelayStatus(state="UNKNOWN")
            last_state = self._relay_state_summary()
            if self._relay_status_is_fresh() and status.state == "ACTIVE":
                self.motion_enabled = True
                print("[Bridge] Real arm command relay is ACTIVE.")
                return
            if self._relay_status_is_fresh() and status.state == "FAULT":
                break
            time.sleep(0.05)
        self.commander.publish_motion_enable(False)
        raise RuntimeError(f"A1 relay did not become ACTIVE: {last_state}")

    def _read_lingbot_obs(self) -> dict | None:
        high = self.cam_high.read_rgb()
        wrist = self.cam_wrist.read_rgb()
        if high is None or wrist is None:
            return None
        obs = {
            "observation.images.cam_high": high,
            "observation.images.cam_wrist": wrist,
        }
        packet = {"obs": [obs], "prompt": self.args.prompt}
        if self.args.condition_on_ee_state:
            state = self._lingbot_state_condition()
            if state is not None:
                packet["state"] = state
                if not self.reported_condition_state:
                    first = state[:, 0, 0]
                    print(
                        "[Bridge] Conditioning LingBot on EE state "
                        f"xyz={np.round(first[:3], 4).tolist()} "
                        f"quat={np.round(first[3:7], 4).tolist()} "
                        f"gripper_norm={first[7]:.3f}"
                    )
                    self.reported_condition_state = True
            elif not self.reported_condition_state:
                print("[Bridge WARNING] EE state conditioning requested, but no fresh/current or --initial-ee-pose is available")
                self.reported_condition_state = True
        return packet

    def _feedback_is_fresh(self) -> bool:
        if self.latest_pose_monotonic is None:
            return False
        return time.monotonic() - self.latest_pose_monotonic <= self.args.max_feedback_age

    def _wait_for_fresh_feedback(self) -> None:
        deadline = time.monotonic() + self.args.feedback_wait_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self._feedback_is_fresh() and self._current_quat() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No fresh {self.args.state_pose_topic} feedback within "
            f"{self.args.feedback_wait_timeout:.1f}s; check /dev/a1 and the A1 driver"
        )

    def _current_xyz(self) -> np.ndarray | None:
        xyz_quat = pose_msg_to_xyz_quat(self.latest_pose)
        return None if xyz_quat is None else xyz_quat[0]

    def _current_quat(self) -> np.ndarray | None:
        xyz_quat = pose_msg_to_xyz_quat(self.latest_pose)
        return None if xyz_quat is None else xyz_quat[1]

    def _normalize_condition_action(self, action8: np.ndarray) -> np.ndarray:
        return normalize_condition_action(action8, self.action_config)

    def _current_or_initial_action8(self) -> np.ndarray | None:
        if self.args.initial_ee_pose is not None:
            return self._normalize_condition_action(np.asarray(self.args.initial_ee_pose, dtype=np.float64))

        if not self._feedback_is_fresh():
            return None
        xyz = self._current_xyz()
        quat = self._current_quat()
        if xyz is None or quat is None:
            return None
        if self.latest_gripper is not None and self.args.gripper_stroke_scale != 0:
            gripper_norm = gripper_norm_from_stroke(float(self.latest_gripper), self.action_config)
        else:
            gripper_norm = 0.0
        return self._normalize_condition_action(np.concatenate([xyz, quat, [gripper_norm]]))

    def _lingbot_state_condition(self) -> np.ndarray | None:
        action = self._current_or_initial_action8()
        if action is None:
            return None
        return condition_state_from_action8(
            action,
            frame_chunk_size=self.args.lingbot_frame_chunk_size,
            action_per_frame=self.args.lingbot_action_per_frame,
        )

    def _sanitize_action(self, raw8: np.ndarray) -> np.ndarray:
        return sanitize_policy_action(raw8, self.action_config, current_xyz=self._current_xyz())

    def _prepare_action_for_execution(self, raw8: np.ndarray) -> np.ndarray:
        return prepare_policy_action(
            raw8,
            self.action_config,
            current_xyz=self._current_xyz(),
            current_quat=self._current_quat(),
            require_current_orientation=self.args.execute,
        )

    def _tracker_command_action(self, policy_action8: np.ndarray) -> np.ndarray:
        return tracker_command_action(policy_action8, self.action_config, current_xyz=self._current_xyz())

    def _actual_action_state(self, fallback: np.ndarray) -> np.ndarray:
        actual = np.asarray(fallback, dtype=np.float64).reshape(8).copy()
        xyz = self._current_xyz()
        quat = self._current_quat()
        if xyz is not None:
            actual[:3] = xyz
        if quat is not None:
            actual[3:7] = quat
        if self.latest_gripper is not None and self.args.gripper_stroke_scale != 0:
            actual[7] = gripper_norm_from_stroke(float(self.latest_gripper), self.action_config)
        return actual

    def _wait_for_target_tracking(self, policy_action8: np.ndarray, started: float) -> float:
        if self.args.eef_servo_settle <= 0:
            return float("nan")
        deadline = time.monotonic() + self.args.eef_servo_settle
        err_norm = float("inf")
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            cur = self._current_xyz()
            if cur is not None:
                err = np.asarray(policy_action8[:3], dtype=np.float64) - cur
                err_norm = float(np.linalg.norm(err))
                if err_norm <= self.args.eef_servo_tolerance:
                    break
            time.sleep(0.03)
        cur = self._current_xyz()
        if cur is not None:
            err_norm = float(np.linalg.norm(np.asarray(policy_action8[:3], dtype=np.float64) - cur))
            print(
                "[Track] "
                f"waited={time.monotonic() - started:.2f}s actual_xyz={np.round(cur, 4).tolist()} "
                f"target_err_cm={err_norm * 100.0:.2f}"
            )
        return err_norm

    def _publish_pose_and_gripper(self, action8: np.ndarray, *, publish_gripper: bool) -> None:
        self.commander.publish_action(action8, publish_gripper=publish_gripper)

    def _publish_ee_action(self, action8: np.ndarray) -> np.ndarray:
        if not self._feedback_is_fresh():
            raise RuntimeError("A1 end-effector feedback is missing or stale; refusing to publish")

        policy_action = self._prepare_action_for_execution(action8)
        started = time.monotonic()
        last_command = self._tracker_command_action(policy_action)

        # Publish the target before arming the relay. While the relay is locked,
        # this can only refresh the staged tracker command; it cannot move the arm.
        # The gripper bypasses the relay, so it is delayed until ACTIVE is confirmed.
        self._publish_pose_and_gripper(last_command, publish_gripper=False)
        self._enable_motion()
        self._publish_pose_and_gripper(last_command, publish_gripper=True)

        err_norm = self._wait_for_target_tracking(policy_action, started)
        corrections = max(0, self.args.eef_servo_corrections)
        for correction_i in range(corrections):
            if err_norm <= self.args.eef_servo_tolerance:
                break
            command = self._tracker_command_action(policy_action)
            if np.allclose(command[:3], last_command[:3], atol=1e-4):
                break
            last_command = command
            print(
                "[Track correction] "
                f"{correction_i + 1}/{corrections} command_xyz={np.round(command[:3], 4).tolist()} "
                f"policy_xyz={np.round(policy_action[:3], 4).tolist()}"
            )
            self._publish_pose_and_gripper(command, publish_gripper=False)
            err_norm = self._wait_for_target_tracking(policy_action, started)

        grip_mm = gripper_stroke_from_norm(float(policy_action[7]), self.action_config)
        print(
            "[Publish] xyz="
            f"{np.round(policy_action[:3], 4).tolist()} "
            f"cmd_xyz={np.round(last_command[:3], 4).tolist()} "
            f"quat={np.round(policy_action[3:7], 4).tolist()} "
            f"gripper_mm={round(float(grip_mm), 3)}"
        )
        return self._actual_action_state(policy_action) if self.args.cache_actual_feedback else policy_action

    @staticmethod
    def _ask_next(prompt: str) -> str:
        try:
            return input(prompt).strip().lower()
        except EOFError:
            return "q"

    def _xyz_direction(self, delta: np.ndarray) -> str:
        return format_xyz_direction(delta, deadband_m=self.args.review_deadband)

    def _clamp_notes(self, raw8: np.ndarray) -> list[str]:
        return clamp_notes(raw8, self.action_config, current_xyz=self._current_xyz())

    def _print_step_preview(self, call_idx: int, frame_i: int, step_i: int, raw8: np.ndarray, safe: np.ndarray) -> None:
        cur = self._current_xyz()
        if cur is None:
            condition_action = self._current_or_initial_action8()
            if condition_action is not None:
                cur = condition_action[:3]
        raw_xyz = np.asarray(raw8[:3], dtype=np.float64)
        safe_delta = None if cur is None else safe[:3] - cur
        raw_delta = None if cur is None else raw_xyz - cur
        grip_mm = gripper_stroke_from_norm(float(safe[7]), self.action_config)
        clamp_notes = self._clamp_notes(raw8)
        print(
            f"[Next] call={call_idx + 1} frame={frame_i} step={step_i} "
            f"raw={np.round(raw8, 4).tolist()} safe={np.round(safe, 4).tolist()}"
        )
        if cur is not None:
            print(
                "       current_xyz="
                f"{np.round(cur, 4).tolist()} "
                f"raw_delta_cm={np.round(raw_delta * 100.0, 2).tolist()} "
                f"safe_delta_cm={np.round(safe_delta * 100.0, 2).tolist()} "
                f"safe_norm_cm={np.linalg.norm(safe_delta) * 100.0:.2f} "
                f"direction={self._xyz_direction(safe_delta)}"
            )
            tracker_cmd = self._tracker_command_action(safe)
            if not np.allclose(tracker_cmd[:3], safe[:3], atol=1e-5):
                tracker_delta = tracker_cmd[:3] - cur
                print(
                    "       tracker_cmd_xyz="
                    f"{np.round(tracker_cmd[:3], 4).tolist()} "
                    f"tracker_cmd_delta_cm={np.round(tracker_delta * 100.0, 2).tolist()} "
                    f"servo_gain={self.args.eef_servo_gain:.2f}"
                )
        print(
            f"       gripper_norm={safe[7]:.3f} gripper_mm={grip_mm:.1f} "
            f"orientation_mode={self.args.orientation_mode} execute={self.args.execute} "
            f"clamp={','.join(clamp_notes) if clamp_notes else 'none'}"
        )

    def run(self):
        first = True
        call_idx = 0
        if self.args.execute:
            self._wait_for_fresh_feedback()
            self._hold_current_pose()
            rospy.sleep(1.0)
            print("[Bridge] Holding the current EE pose continuously while waiting for Enter.")
        while not rospy.is_shutdown():
            if self.args.max_model_calls > 0 and call_idx >= self.args.max_model_calls:
                break
            if self.args.step_mode:
                print(
                    f"\n========== INFERENCE #{call_idx + 1} READY ==========\n"
                    "Press Enter once to run ONE new LingBot action inference.\n"
                    "q=quit without running inference."
                )
                cmd = self._ask_next(f"[INFERENCE #{call_idx + 1} READY] Enter=infer, q=quit: ")
                if cmd in {"q", "quit", "exit"}:
                    break
            if self.args.no_kv_update and not first:
                self.client.reset(self.args.prompt)
            obs = None
            while obs is None and not rospy.is_shutdown():
                obs = self._read_lingbot_obs()
                if obs is None:
                    time.sleep(0.01)
            if rospy.is_shutdown():
                break

            print(f"[INFERENCE #{call_idx + 1} START] Captured camera observation; model is running...", flush=True)
            t0 = time.monotonic()
            ret = self.client.infer(obs)
            dt = time.monotonic() - t0
            action = np.asarray(ret["action"], dtype=np.float32)
            if action.ndim != 3 or action.shape[0] != 8:
                raise RuntimeError(
                    f"Expected LingBot action shape (8, F, H), got {action.shape}. "
                    "Restart the server with the corrected Galaxea A1 config."
                )
            if action.shape[2] % 4 != 0:
                raise RuntimeError(f"Action horizon must be divisible by 4, got {action.shape[2]}")
            print(
                f"\a[INFERENCE #{call_idx + 1} DONE] infer={dt:.3f}s "
                f"action_shape={action.shape} obs_frames={len(obs['obs'])}",
                flush=True,
            )

            key_frames = []
            start_frame = 1 if first else 0
            end_frame = min(action.shape[1], start_frame + self.args.execute_frames)
            if end_frame <= start_frame:
                raise RuntimeError(
                    f"No executable LingBot frames: first={first}, action_shape={action.shape}"
                )
            total_steps = (end_frame - start_frame) * action.shape[2]
            if self.args.step_actions:
                print(
                    f"[EXECUTION #{call_idx + 1}] This inference produced {total_steps} EE steps.\n"
                    f"The next {total_steps} Enter presses publish these existing steps; "
                    "they DO NOT run new inference."
                )
            actions_per_observation = action.shape[2] // 4
            cache_state = action[:, :end_frame].copy() if first else action[:, start_frame:end_frame].copy()
            stop_requested = False
            cache_eligible = True
            for frame_i in range(start_frame, end_frame):
                for step_i in range(action.shape[2]):
                    raw8 = action[:, frame_i, step_i]
                    safe = self._prepare_action_for_execution(raw8)
                    cache_frame_i = frame_i if first else frame_i - start_frame
                    cache_state[:, cache_frame_i, step_i] = safe
                    should_preview = self.args.print_actions and (step_i == 0 or self.args.step_actions)
                    if should_preview:
                        self._print_step_preview(call_idx, frame_i, step_i, raw8, safe)
                    if self.args.step_actions:
                        cmd = self._ask_next("       Next=publish this EE step, s=skip, q=quit: ")
                        if cmd in {"q", "quit", "exit"}:
                            stop_requested = True
                            break
                        if cmd in {"s", "skip"}:
                            cache_eligible = False
                            continue
                    if self.args.execute:
                        executed_state = self._publish_ee_action(safe)
                        cache_state[:, cache_frame_i, step_i] = executed_state
                    time.sleep(max(0.0, 1.0 / self.args.exec_rate))

                    if (step_i + 1) % actions_per_observation == 0:
                        obs2 = self._read_lingbot_obs()
                        if obs2 is None:
                            raise RuntimeError("Camera frame unavailable during KV-cache collection")
                        key_frames.extend(obs2["obs"])
                if stop_requested:
                    break

            if stop_requested:
                break

            cache_updated = False
            if self.args.execute and key_frames and cache_eligible and not self.args.no_kv_update:
                # Keep LingBot's internal KV/cache state aligned with what was executed.
                try:
                    self.client.infer({
                        "obs": key_frames,
                        "compute_kv_cache": True,
                        "imagine": False,
                        "state": cache_state,
                    })
                    cache_updated = True
                    print(
                        f"[CACHE UPDATE #{call_idx + 1}] Context synchronized; "
                        "this did NOT generate new actions. "
                        f"rgb_frames={len(key_frames)} action_frames={cache_state.shape[1]}"
                    )
                except Exception as exc:
                    print(f"[Bridge WARNING] compute_kv_cache failed: {exc}")
                    self.client.reset(self.args.prompt)
                    print("[Bridge] Server reset after KV-cache failure")
            elif self.args.execute and not cache_eligible and not self.args.no_kv_update:
                self.client.reset(self.args.prompt)
                print("[Bridge] Server reset because one or more actions were skipped")
            if self.args.execute:
                first = not (self.args.no_kv_update or cache_updated)
            print(
                f"\a========== INFERENCE #{call_idx + 1} EXECUTION COMPLETE ==========\n"
                "The next READY prompt is the boundary before another new model inference.",
                flush=True,
            )
            call_idx += 1

    def close(self):
        self.commander.publish_motion_enable(False)
        self.pose_keepalive_timer.shutdown()
        self.cam_high.close()
        self.cam_wrist.close()
        self.client.ws.close()


def parse_args():
    p = argparse.ArgumentParser(description="LingBot-VA EE-pose bridge for Galaxea A1")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1106)
    p.add_argument("--prompt", default="pick up that bowl")
    p.add_argument("--execute", action="store_true", help="Actually publish EE commands. Default is dry-run.")
    p.add_argument("--step-mode", action=argparse.BooleanOptionalAction, default=True,
                   help="Wait for Enter before each model inference chunk.")
    p.add_argument("--step-actions", action="store_true", default=False,
                   help="Wait for Enter before every individual EE pose command inside the predicted chunk.")
    p.add_argument("--no-kv-update", action="store_true", default=False,
                   help="Skip LingBot KV-cache update after executing a chunk; useful for isolated manual probing.")
    p.add_argument("--max-model-calls", type=int, default=1,
                   help="Stop after this many model calls. 0 means run until q/Ctrl-C.")
    p.add_argument("--execute-frames", type=int, default=1, help="How many LingBot frame chunks to execute per model call")
    p.add_argument("--condition-on-ee-state", action=argparse.BooleanOptionalAction, default=True,
                   help="Include current/initial 8D EE pose as LingBot action-state conditioning.")
    p.add_argument("--initial-ee-pose", type=float, nargs=8, default=None,
                   metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW", "GRIP"),
                   help="Fallback 8D EE state condition when live feedback is unavailable; gripper is normalized 0..1.")
    p.add_argument("--lingbot-frame-chunk-size", type=int, default=4,
                   help="LingBot action-state frame dimension for first-frame conditioning.")
    p.add_argument("--lingbot-action-per-frame", type=int, default=20,
                   help="LingBot action-state horizon dimension for first-frame conditioning.")
    p.add_argument("--exec-rate", type=float, default=30.0)
    p.add_argument("--print-actions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--review-deadband", type=float, default=0.001,
                   help="XYZ delta below this many meters is printed as hold for direction review.")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--cam0-serial", default="341522300456")
    p.add_argument("--cam0-auto-exposure", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cam0-exposure", type=int, default=140)
    p.add_argument("--cam0-gain", type=int, default=32)
    p.add_argument("--cam0-auto-white-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cam0-white-balance", type=int, default=4600)
    p.add_argument("--cam1-device", default="/dev/video0")
    p.add_argument("--cam1-backend-api", default="v4l2")
    p.add_argument("--state-pose-topic", default="/end_effector_pose")
    p.add_argument("--state-gripper-topic", default="/gripper_stroke_host")
    p.add_argument("--cmd-pose-topic", default="/a1_ee_target")
    p.add_argument("--cmd-gripper-topic", default="/gripper_position_control_host")
    p.add_argument("--motion-enable-topic", default="/a1_arm_motion_enable")
    p.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    p.add_argument("--relay-enable-timeout", type=float, default=2.0)
    p.add_argument("--max-relay-status-age", type=float, default=1.0,
                   help="Maximum age in seconds for trusting /a1_arm_relay_status while executing.")
    p.add_argument("--command-frame", default="world")
    p.add_argument("--orientation-mode", choices=["hold-current", "model-quat"], default="hold-current",
                   help="Safest default holds current EE orientation; model-quat uses LingBot channels 3..6 directly.")
    p.add_argument("--eef-servo-gain", type=float, default=1.0,
                   help="Gain >1 sends an amplified tracker target toward the policy target to compensate under-tracking.")
    p.add_argument("--eef-servo-max-extra", type=float, default=0.04,
                   help="Maximum extra overshoot distance in meters when eef-servo-gain > 1. 0 means unlimited before workspace clamp.")
    p.add_argument("--eef-servo-settle", type=float, default=0.0,
                   help="Seconds to hold each command and measure target tracking error after publish.")
    p.add_argument("--eef-servo-tolerance", type=float, default=0.01,
                   help="XYZ norm tolerance in meters for servo settle/correction.")
    p.add_argument("--eef-servo-corrections", type=int, default=0,
                   help="Additional correction publishes after settle if actual EEF is still far from the policy target.")
    p.add_argument("--cache-actual-feedback", action=argparse.BooleanOptionalAction, default=True,
                   help="Use measured /end_effector_pose, not commanded target, for LingBot KV-cache action state.")
    p.add_argument("--xyz-min", type=float, nargs=3, default=[0.06, -0.27, 0.06])
    p.add_argument("--xyz-max", type=float, nargs=3, default=[0.44, 0.14, 0.50])
    p.add_argument("--min-quat-norm", type=float, default=0.25)
    p.add_argument("--max-feedback-age", type=float, default=0.5)
    p.add_argument("--feedback-wait-timeout", type=float, default=5.0)
    p.add_argument("--gripper-stroke-scale", type=float, default=60.0)
    p.add_argument("--gripper-stroke-offset", type=float, default=0.0)
    p.add_argument("--gripper-stroke-min", type=float, default=0.0)
    p.add_argument("--gripper-stroke-max", type=float, default=60.0)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.execute:
        print("[Bridge] DRY RUN: not publishing robot commands. Pass --execute to move the robot.")
    bridge = A1LingBotEEBridge(args)
    try:
        bridge.run()
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
