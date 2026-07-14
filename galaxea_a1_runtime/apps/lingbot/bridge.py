#!/usr/bin/env python3
# ruff: noqa: E402
"""LingBot-VA -> Galaxea A1 end-effector pose bridge implementation.

This is intentionally dry-run by default. Pass --execute to publish commands to
/a1_ee_target and the staged gripper target consumed by the safe relay.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Keep ROS1 ahead of any ROS2 paths, and expose A1 custom messages.
ROOT_DIR = Path(__file__).resolve().parents[3]
_A1_SDK = ROOT_DIR / "third_party" / "A1_SDK" / "install"
_ROS1_OVERLAY = ROOT_DIR / ".cache" / "ros1_python_overlay"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
for p in list(sys.path):
    if "/opt/ros/humble" in p:
        sys.path.remove(p)
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
    str(_ROS1_OVERLAY),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import numpy as np

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
    absolute_action_to_relative,
    clamp_notes,
    gripper_norm_from_stroke,
    gripper_stroke_from_norm,
    normalize_condition_action,
    prepare_policy_action,
    relative_action_to_absolute,
    sanitize_policy_action,
    tracker_command_action,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String


from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.apps.lingbot.camera_session import LingBotCameraSession


class A1LingBotEEBridge:
    def __init__(self, args):
        self.args = args
        self.latest_pose = None
        self.latest_pose_monotonic = None
        self.latest_gripper = None
        self.latest_gripper_monotonic = None
        self.motion_enabled = False
        self.relay_status = None
        self.relay_status_monotonic = None
        self.reported_condition_state = False
        self.episode_origin = None
        self.action_config = self._action_config_from_args(args)
        self.front_roi = _front_roi_from_args(args)
        rospy.init_node("lingbot_va_ee_bridge", anonymous=False)
        pose_pub = rospy.Publisher(args.cmd_pose_topic, PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher(
            args.cmd_gripper_topic, gripper_position_control, queue_size=10
        )
        motion_enable_pub = rospy.Publisher(
            args.motion_enable_topic, Bool, queue_size=1, latch=True
        )
        self.commander = EefCommandPublisher(
            rospy=rospy,
            pose_pub=pose_pub,
            gripper_pub=gripper_pub,
            motion_enable_pub=motion_enable_pub,
            pose_msg_type=PoseStamped,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            command_frame=args.command_frame,
            gripper_to_stroke=lambda value: gripper_stroke_from_norm(
                value, self.action_config
            ),
            execute=args.execute,
        )
        rospy.Subscriber(
            args.state_pose_topic, PoseStamped, self._pose_cb, queue_size=1
        )
        rospy.Subscriber(
            args.state_gripper_topic, JointState, self._gripper_cb, queue_size=1
        )
        rospy.Subscriber(
            args.relay_status_topic, String, self._relay_status_cb, queue_size=1
        )
        self.pose_keepalive_timer = rospy.Timer(
            rospy.Duration(0.05), self._publish_active_pose_target
        )

        self.cameras = LingBotCameraSession(args, self.front_roi)
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
            gripper_stroke_min=float(args.gripper_stroke_min),
            gripper_stroke_max=float(args.gripper_stroke_max),
        )

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose = msg
        self.latest_pose_monotonic = time.monotonic()

    def _gripper_cb(self, msg: JointState):
        if msg.position:
            self.latest_gripper = float(msg.position[0])
            self.latest_gripper_monotonic = time.monotonic()

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
        obs = self.cameras.read_observation()
        if obs is None:
            return None
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
                print(
                    "[Bridge WARNING] EE state conditioning requested, but no fresh/current or --initial-ee-pose is available"
                )
                self.reported_condition_state = True
        return packet

    def _feedback_is_fresh(self) -> bool:
        if self.latest_pose_monotonic is None:
            return False
        return (
            time.monotonic() - self.latest_pose_monotonic <= self.args.max_feedback_age
        )

    def _gripper_feedback_is_fresh(self) -> bool:
        if self.latest_gripper_monotonic is None:
            return False
        return (
            time.monotonic() - self.latest_gripper_monotonic
            <= self.args.max_feedback_age
        )

    def _wait_for_fresh_feedback(self) -> None:
        deadline = time.monotonic() + self.args.feedback_wait_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if (
                self._feedback_is_fresh()
                and self._gripper_feedback_is_fresh()
                and self._current_quat() is not None
            ):
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No fresh pose and gripper feedback within "
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

    def _current_absolute_action8(self) -> np.ndarray | None:
        if not self._feedback_is_fresh() or not self._gripper_feedback_is_fresh():
            if self.args.initial_ee_pose is None:
                return None
            return self._normalize_condition_action(
                np.asarray(self.args.initial_ee_pose, dtype=np.float64)
            )
        xyz = self._current_xyz()
        quat = self._current_quat()
        if xyz is None or quat is None:
            return None
        if self._gripper_feedback_is_fresh() and self.latest_gripper is not None:
            gripper_norm = gripper_norm_from_stroke(
                float(self.latest_gripper), self.action_config
            )
        else:
            gripper_norm = 0.0
        return self._normalize_condition_action(
            np.concatenate([xyz, quat, [gripper_norm]])
        )

    def _ensure_episode_origin(self) -> np.ndarray | None:
        if self.episode_origin is not None:
            return self.episode_origin
        if self.args.initial_ee_pose is not None:
            origin = self._normalize_condition_action(
                np.asarray(self.args.initial_ee_pose, dtype=np.float64)
            )
        else:
            origin = self._current_absolute_action8()
        if origin is None:
            return None
        self.episode_origin = origin
        print(
            "[Bridge] Episode EEF origin "
            f"xyz={np.round(origin[:3], 4).tolist()} "
            f"quat={np.round(origin[3:7], 4).tolist()} "
            f"pose_mode={self.args.action_pose_mode}"
        )
        return self.episode_origin

    def _absolute_to_model_action(self, absolute8: np.ndarray) -> np.ndarray:
        if self.args.action_pose_mode == "absolute":
            return np.asarray(absolute8, dtype=np.float64).reshape(8).copy()
        origin = self._ensure_episode_origin()
        if origin is None:
            raise RuntimeError("Episode EEF origin is unavailable")
        return absolute_action_to_relative(
            absolute8, origin, min_quat_norm=self.args.min_quat_norm
        )

    def _model_to_absolute_action(self, model8: np.ndarray) -> np.ndarray:
        if self.args.action_pose_mode == "absolute":
            return np.asarray(model8, dtype=np.float64).reshape(8).copy()
        origin = self._ensure_episode_origin()
        if origin is None:
            raise RuntimeError("Episode EEF origin is unavailable")
        return relative_action_to_absolute(
            model8, origin, min_quat_norm=self.args.min_quat_norm
        )

    def _current_or_initial_action8(self) -> np.ndarray | None:
        absolute = self._current_absolute_action8()
        if absolute is None:
            return None
        return self._normalize_condition_action(
            self._absolute_to_model_action(absolute)
        )

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
        absolute = self._model_to_absolute_action(raw8)
        return sanitize_policy_action(
            absolute, self.action_config, current_xyz=self._current_xyz()
        )

    def _prepare_action_for_execution(self, raw8: np.ndarray) -> np.ndarray:
        absolute = self._model_to_absolute_action(raw8)
        return prepare_policy_action(
            absolute,
            self.action_config,
            current_xyz=self._current_xyz(),
            current_quat=self._current_quat(),
            require_current_orientation=self.args.execute,
        )

    def _tracker_command_action(self, policy_action8: np.ndarray) -> np.ndarray:
        return tracker_command_action(
            policy_action8, self.action_config, current_xyz=self._current_xyz()
        )

    def _actual_action_state(self, fallback: np.ndarray) -> np.ndarray:
        actual = np.asarray(fallback, dtype=np.float64).reshape(8).copy()
        xyz = self._current_xyz()
        quat = self._current_quat()
        if xyz is not None:
            actual[:3] = xyz
        if quat is not None:
            actual[3:7] = quat
        if self._gripper_feedback_is_fresh() and self.latest_gripper is not None:
            actual[7] = gripper_norm_from_stroke(
                float(self.latest_gripper), self.action_config
            )
        return actual

    def _wait_for_target_tracking(
        self, policy_action8: np.ndarray, started: float
    ) -> float:
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
            err_norm = float(
                np.linalg.norm(np.asarray(policy_action8[:3], dtype=np.float64) - cur)
            )
            print(
                "[Track] "
                f"waited={time.monotonic() - started:.2f}s actual_xyz={np.round(cur, 4).tolist()} "
                f"target_err_cm={err_norm * 100.0:.2f}"
            )
        return err_norm

    def _publish_pose_and_gripper(
        self, action8: np.ndarray, *, publish_gripper: bool
    ) -> None:
        self.commander.publish_action(action8, publish_gripper=publish_gripper)

    def _publish_ee_action(self, policy_action: np.ndarray) -> np.ndarray:
        if not self._feedback_is_fresh() or not self._gripper_feedback_is_fresh():
            raise RuntimeError(
                "A1 pose or gripper feedback is missing or stale; refusing to publish"
            )

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
        return (
            self._actual_action_state(policy_action)
            if self.args.cache_actual_feedback
            else last_command
        )

    @staticmethod
    def _ask_next(prompt: str) -> str:
        try:
            return input(prompt).strip().lower()
        except EOFError:
            return "q"

    def _xyz_direction(self, delta: np.ndarray) -> str:
        return format_xyz_direction(delta, deadband_m=self.args.review_deadband)

    def _clamp_notes(self, raw8: np.ndarray) -> list[str]:
        absolute = self._model_to_absolute_action(raw8)
        return clamp_notes(
            absolute, self.action_config, current_xyz=self._current_xyz()
        )

    def _print_step_preview(
        self,
        call_idx: int,
        frame_i: int,
        step_i: int,
        raw8: np.ndarray,
        safe: np.ndarray,
    ) -> None:
        cur = self._current_xyz()
        if cur is None:
            condition_action = self._current_absolute_action8()
            if condition_action is not None:
                cur = condition_action[:3]
        raw_absolute = self._model_to_absolute_action(raw8)
        raw_xyz = np.asarray(raw_absolute[:3], dtype=np.float64)
        safe_delta = None if cur is None else safe[:3] - cur
        raw_delta = None if cur is None else raw_xyz - cur
        grip_mm = gripper_stroke_from_norm(float(safe[7]), self.action_config)
        clamp_notes = self._clamp_notes(raw8)
        print(
            f"[Next] call={call_idx + 1} frame={frame_i} step={step_i} "
            f"model={np.round(raw8, 4).tolist()} absolute={np.round(raw_absolute, 4).tolist()} "
            f"safe={np.round(safe, 4).tolist()}"
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
            if self._ensure_episode_origin() is None:
                raise RuntimeError("Cannot establish the episode EEF origin")
            self._hold_current_pose()
            rospy.sleep(1.0)
            if self.args.step_mode:
                print(
                    "[Bridge] Holding the current EE pose continuously while waiting for Enter."
                )
            else:
                cache_source = (
                    "measured-feedback"
                    if self.args.cache_actual_feedback
                    else "tracker-command"
                )
                print(
                    "[Bridge] Continuous execution armed: "
                    f"calls={self.args.max_model_calls or 'unbounded'} "
                    f"frames_per_call={self.args.execute_frames} rate={self.args.exec_rate:.1f}Hz "
                    f"cache_action_source={cache_source}"
                )
        while not rospy.is_shutdown():
            if self.args.max_model_calls > 0 and call_idx >= self.args.max_model_calls:
                break
            if self.args.step_mode:
                print(
                    f"\n========== INFERENCE #{call_idx + 1} READY ==========\n"
                    "Press Enter once to run ONE new LingBot action inference.\n"
                    "q=quit without running inference."
                )
                cmd = self._ask_next(
                    f"[INFERENCE #{call_idx + 1} READY] Enter=infer, q=quit: "
                )
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

            print(
                f"[INFERENCE #{call_idx + 1} START] Captured camera observation; model is running...",
                flush=True,
            )
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
                raise RuntimeError(
                    f"Action horizon must be divisible by 4, got {action.shape[2]}"
                )
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
            cache_state = (
                action[:, :end_frame].copy()
                if first
                else action[:, start_frame:end_frame].copy()
            )
            stop_requested = False
            cache_eligible = True
            for frame_i in range(start_frame, end_frame):
                for step_i in range(action.shape[2]):
                    raw8 = action[:, frame_i, step_i]
                    safe = self._prepare_action_for_execution(raw8)
                    cache_frame_i = frame_i if first else frame_i - start_frame
                    cache_state[:, cache_frame_i, step_i] = (
                        self._absolute_to_model_action(safe)
                    )
                    should_preview = self.args.print_actions and (
                        step_i == 0 or self.args.step_actions
                    )
                    if should_preview:
                        self._print_step_preview(call_idx, frame_i, step_i, raw8, safe)
                    if self.args.step_actions:
                        cmd = self._ask_next(
                            "       Next=publish this EE step, s=skip, q=quit: "
                        )
                        if cmd in {"q", "quit", "exit"}:
                            stop_requested = True
                            break
                        if cmd in {"s", "skip"}:
                            cache_eligible = False
                            continue
                    if self.args.execute:
                        executed_state = self._publish_ee_action(safe)
                        cache_state[:, cache_frame_i, step_i] = (
                            self._absolute_to_model_action(executed_state)
                        )
                    time.sleep(max(0.0, 1.0 / self.args.exec_rate))

                    if (step_i + 1) % actions_per_observation == 0:
                        obs2 = self._read_lingbot_obs()
                        if obs2 is None:
                            raise RuntimeError(
                                "Camera frame unavailable during KV-cache collection"
                            )
                        key_frames.extend(obs2["obs"])
                if stop_requested:
                    break

            if stop_requested:
                break

            cache_updated = False
            if (
                self.args.execute
                and key_frames
                and cache_eligible
                and not self.args.no_kv_update
            ):
                # Keep LingBot's internal KV/cache state aligned with what was executed.
                try:
                    self.client.infer(
                        {
                            "obs": key_frames,
                            "compute_kv_cache": True,
                            "imagine": False,
                            "state": cache_state,
                        }
                    )
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
            elif (
                self.args.execute and not cache_eligible and not self.args.no_kv_update
            ):
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
        self.cameras.close()
        self.client.ws.close()


def _front_roi_from_args(args) -> ImageRoi:
    if not args.cam0_crop_enabled:
        raise ValueError(
            "--cam0-crop-enabled is required for the inference input contract"
        )
    roi = ImageRoi(
        x=args.cam0_crop_x,
        y=args.cam0_crop_y,
        width=args.cam0_crop_width,
        height=args.cam0_crop_height,
    )
    roi.validate(
        image_width=args.cam_width,
        image_height=args.cam_height,
        label="AgentView inference crop",
    )
    if roi.width != roi.height:
        raise ValueError(
            f"AgentView inference crop must be square, got {roi.width}x{roi.height}"
        )
    return roi
