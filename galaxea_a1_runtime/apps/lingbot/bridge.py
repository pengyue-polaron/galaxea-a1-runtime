#!/usr/bin/env python3
# ruff: noqa: E402
"""LingBot-VA -> Galaxea A1 end-effector pose bridge implementation.

This is intentionally dry-run by default. Pass --execute to publish commands to
/a1_ee_target and the staged gripper target consumed by the safe relay.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Keep ROS1 ahead of any ROS2 paths, and expose A1 custom messages.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR, remove_ros2=True)

import numpy as np

import rospy
from galaxea_a1_runtime.apps.eef_bridge import (
    EefCommandPublisher,
)
from galaxea_a1_runtime.apps.lingbot.actions import (
    LingBotActionConfig,
    gripper_stroke_from_norm,
)
from galaxea_a1_runtime.apps.lingbot.episode_state import LingBotEpisodeState
from galaxea_a1_runtime.apps.lingbot.review import LingBotActionReviewer
from galaxea_a1_runtime.apps.lingbot.rollout import LingBotActionChunk
from galaxea_a1_runtime.apps.policy_camera import (
    PolicyCameraSession,
    required_square_front_roi,
)
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String


from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.runtime.relay import RelayMonitor


class A1LingBotEEBridge:
    def __init__(self, args):
        self.args = args
        self.motion_enabled = False
        self.reported_condition_state = False
        self.action_config = self._action_config_from_args(args)
        self.state = LingBotEpisodeState(
            action_config=self.action_config,
            pose_mode=args.action_pose_mode,
            max_feedback_age_s=args.max_feedback_age,
            initial_action8=args.initial_ee_pose,
            frame_chunk_size=args.lingbot_frame_chunk_size,
            action_per_frame=args.lingbot_action_per_frame,
        )
        self.relay = RelayMonitor(args.max_relay_status_age)
        self.reviewer = LingBotActionReviewer(
            state=self.state,
            action_config=self.action_config,
            review_deadband_m=args.review_deadband,
            servo_gain=args.eef_servo_gain,
            orientation_mode=args.orientation_mode,
            execute=args.execute,
        )
        self.front_roi = required_square_front_roi(args)
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
            args.state_pose_topic, PoseStamped, self.state.pose_callback, queue_size=1
        )
        rospy.Subscriber(
            args.state_gripper_topic,
            JointState,
            self.state.gripper_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            args.relay_status_topic, String, self.relay.callback, queue_size=1
        )
        self.pose_keepalive_timer = rospy.Timer(
            rospy.Duration(0.05), self._publish_active_pose_target
        )

        self.cameras = PolicyCameraSession(args, self.front_roi)
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

    def _set_active_pose_target(self, msg: PoseStamped) -> None:
        self.commander.set_active_pose_target_from_msg(msg)

    def _publish_active_pose_target(self, _event=None) -> None:
        self.commander.publish_active_pose_target()

    def _hold_current_pose(self) -> None:
        pose = self.state.current_pose_message()
        if pose is None:
            raise RuntimeError("Cannot hold EE pose before receiving feedback")
        self._set_active_pose_target(pose)
        self._publish_active_pose_target()

    def _enable_motion(self) -> None:
        if self.motion_enabled:
            if self.relay.is_active():
                return
            self.motion_enabled = False
            self.commander.publish_motion_enable(False)
            raise RuntimeError(
                "A1 relay is no longer confirmed ACTIVE; refusing to publish "
                f"more commands. Last relay state: {self.relay.summary()}"
            )
        self.commander.publish_motion_enable(True)
        deadline = time.monotonic() + self.args.relay_enable_timeout
        last_state = "no status"
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status, _ = self.relay.status()
            last_state = self.relay.summary()
            if self.relay.is_active():
                self.motion_enabled = True
                print("[Bridge] Real arm command relay is ACTIVE.")
                return
            if status is not None and status.state == "FAULT":
                break
            time.sleep(0.05)
        self.commander.publish_motion_enable(False)
        raise RuntimeError(f"A1 relay did not become ACTIVE: {last_state}")

    def _read_lingbot_obs(self) -> dict | None:
        obs = self.cameras.read_observation(
            front_key=self.args.cam0_observation_key,
            wrist_key=self.args.cam1_observation_key,
        )
        if obs is None:
            return None
        packet = {"obs": [obs], "prompt": self.args.prompt}
        if self.args.condition_on_ee_state:
            state = self.state.model_condition()
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

    def _wait_for_fresh_feedback(self) -> None:
        deadline = time.monotonic() + self.args.feedback_wait_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if (
                self.state.pose_is_fresh()
                and self.state.gripper_is_fresh()
                and self.state.current_quat() is not None
            ):
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No fresh pose and gripper feedback within "
            f"{self.args.feedback_wait_timeout:.1f}s; check /dev/a1 and the A1 driver"
        )

    def _ensure_episode_origin(self) -> np.ndarray | None:
        existed = self.state.episode_origin is not None
        origin = self.state.ensure_episode_origin()
        if origin is None:
            return None
        if not existed:
            print(
                "[Bridge] Episode EEF origin "
                f"xyz={np.round(origin[:3], 4).tolist()} "
                f"quat={np.round(origin[3:7], 4).tolist()} "
                f"pose_mode={self.args.action_pose_mode}"
            )
        return origin

    def _wait_for_target_tracking(
        self, policy_action8: np.ndarray, started: float
    ) -> float:
        if self.args.eef_servo_settle <= 0:
            return float("nan")
        deadline = time.monotonic() + self.args.eef_servo_settle
        err_norm = float("inf")
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            cur = self.state.current_xyz()
            if cur is not None:
                err = np.asarray(policy_action8[:3], dtype=np.float64) - cur
                err_norm = float(np.linalg.norm(err))
                if err_norm <= self.args.eef_servo_tolerance:
                    break
            time.sleep(0.03)
        cur = self.state.current_xyz()
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
        if not self.state.pose_is_fresh() or not self.state.gripper_is_fresh():
            raise RuntimeError(
                "A1 pose or gripper feedback is missing or stale; refusing to publish"
            )

        started = time.monotonic()
        last_command = self.state.tracker_command(policy_action)

        # Publish the pose target before arming the relay. While locked, this only
        # refreshes the staged tracker command. Delay the gripper target until
        # ACTIVE so enabling cannot forward a target queued before the safety gate.
        self._publish_pose_and_gripper(last_command, publish_gripper=False)
        self._enable_motion()
        self._publish_pose_and_gripper(last_command, publish_gripper=True)

        err_norm = self._wait_for_target_tracking(policy_action, started)
        corrections = max(0, self.args.eef_servo_corrections)
        for correction_i in range(corrections):
            if err_norm <= self.args.eef_servo_tolerance:
                break
            command = self.state.tracker_command(policy_action)
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
            self.state.measured_action(policy_action)
            if self.args.cache_actual_feedback
            else last_command
        )

    @staticmethod
    def _ask_next(prompt: str) -> str:
        try:
            return input(prompt).strip().lower()
        except EOFError:
            return "q"

    def run(self) -> None:
        self._prepare_execution()
        first = True
        call_index = 0
        while not rospy.is_shutdown():
            if (
                self.args.max_model_calls > 0
                and call_index >= self.args.max_model_calls
            ):
                return
            if not self._wait_for_inference_request(call_index):
                return
            if self.args.no_kv_update and not first:
                self.client.reset(self.args.prompt)
            chunk = self._infer_chunk(call_index, first=first)
            if chunk is None:
                return
            stop, key_frames, cache_eligible = self._execute_chunk(call_index, chunk)
            if stop:
                return
            cache_updated = self._sync_kv_cache(
                call_index,
                chunk,
                key_frames=key_frames,
                cache_eligible=cache_eligible,
            )
            if self.args.execute:
                first = not (self.args.no_kv_update or cache_updated)
            print(
                f"\a========== INFERENCE #{call_index + 1} EXECUTION COMPLETE ==========\n"
                "The next READY prompt is the boundary before another new model inference.",
                flush=True,
            )
            call_index += 1

    def _prepare_execution(self) -> None:
        if not self.args.execute:
            return
        self._wait_for_fresh_feedback()
        if self._ensure_episode_origin() is None:
            raise RuntimeError("Cannot establish the episode EEF origin")
        self._hold_current_pose()
        rospy.sleep(1.0)
        if self.args.step_mode:
            print(
                "[Bridge] Holding the current EE pose continuously while waiting for Enter."
            )
            return
        cache_source = (
            "measured-feedback"
            if self.args.cache_actual_feedback
            else "tracker-command"
        )
        print(
            "[Bridge] Continuous execution armed: "
            f"calls={self.args.max_model_calls or 'unbounded'} "
            f"frames_per_call={self.args.execute_frames} "
            f"rate={self.args.exec_rate:.1f}Hz "
            f"cache_action_source={cache_source}"
        )

    def _wait_for_inference_request(self, call_index: int) -> bool:
        if not self.args.step_mode:
            return True
        print(
            f"\n========== INFERENCE #{call_index + 1} READY ==========\n"
            "Press Enter once to run ONE new LingBot action inference.\n"
            "q=quit without running inference."
        )
        command = self._ask_next(
            f"[INFERENCE #{call_index + 1} READY] Enter=infer, q=quit: "
        )
        return command not in {"q", "quit", "exit"}

    def _infer_chunk(
        self, call_index: int, *, first: bool
    ) -> LingBotActionChunk | None:
        observation = None
        while observation is None and not rospy.is_shutdown():
            observation = self._read_lingbot_obs()
            if observation is None:
                time.sleep(0.01)
        if rospy.is_shutdown():
            return None
        print(
            f"[INFERENCE #{call_index + 1} START] "
            "Captured camera observation; model is running...",
            flush=True,
        )
        started = time.monotonic()
        response = self.client.infer(observation)
        elapsed = time.monotonic() - started
        chunk = LingBotActionChunk.from_response(
            response["action"],
            first=first,
            execute_frames=self.args.execute_frames,
        )
        print(
            f"\a[INFERENCE #{call_index + 1} DONE] infer={elapsed:.3f}s "
            f"action_shape={chunk.values.shape} "
            f"obs_frames={len(observation['obs'])}",
            flush=True,
        )
        return chunk

    def _execute_chunk(
        self,
        call_index: int,
        chunk: LingBotActionChunk,
    ) -> tuple[bool, list, bool]:
        if self.args.step_actions:
            print(
                f"[EXECUTION #{call_index + 1}] "
                f"This inference produced {chunk.total_steps} EE steps.\n"
                f"The next {chunk.total_steps} Enter presses publish these existing steps; "
                "they DO NOT run new inference."
            )
        key_frames: list = []
        cache_eligible = True
        for frame_index, step_index, cache_frame_index, raw_action in chunk.steps():
            safe_action = self.state.prepare(
                raw_action, require_orientation=self.args.execute
            )
            chunk.cache_state[:, cache_frame_index, step_index] = (
                self.state.absolute_to_model(safe_action)
            )
            if self.args.print_actions and (step_index == 0 or self.args.step_actions):
                self.reviewer.print_step(
                    call_index=call_index,
                    frame_index=frame_index,
                    step_index=step_index,
                    model_action=raw_action,
                    safe_action=safe_action,
                )
            if self.args.step_actions:
                command = self._ask_next(
                    "       Next=publish this EE step, s=skip, q=quit: "
                )
                if command in {"q", "quit", "exit"}:
                    return True, key_frames, cache_eligible
                if command in {"s", "skip"}:
                    cache_eligible = False
                    continue
            if self.args.execute:
                executed = self._publish_ee_action(safe_action)
                chunk.cache_state[:, cache_frame_index, step_index] = (
                    self.state.absolute_to_model(executed)
                )
            time.sleep(max(0.0, 1.0 / self.args.exec_rate))
            if chunk.needs_observation_after(step_index):
                observation = self._read_lingbot_obs()
                if observation is None:
                    raise RuntimeError(
                        "Camera frame unavailable during KV-cache collection"
                    )
                key_frames.extend(observation["obs"])
        return False, key_frames, cache_eligible

    def _sync_kv_cache(
        self,
        call_index: int,
        chunk: LingBotActionChunk,
        *,
        key_frames: list,
        cache_eligible: bool,
    ) -> bool:
        if (
            self.args.execute
            and key_frames
            and cache_eligible
            and not self.args.no_kv_update
        ):
            try:
                self.client.infer(
                    {
                        "obs": key_frames,
                        "compute_kv_cache": True,
                        "imagine": False,
                        "state": chunk.cache_state,
                    }
                )
                print(
                    f"[CACHE UPDATE #{call_index + 1}] Context synchronized; "
                    "this did NOT generate new actions. "
                    f"rgb_frames={len(key_frames)} "
                    f"action_frames={chunk.cache_state.shape[1]}"
                )
                return True
            except Exception as exc:
                print(f"[Bridge WARNING] compute_kv_cache failed: {exc}")
                self.client.reset(self.args.prompt)
                print("[Bridge] Server reset after KV-cache failure")
        elif self.args.execute and not cache_eligible and not self.args.no_kv_update:
            self.client.reset(self.args.prompt)
            print("[Bridge] Server reset because one or more actions were skipped")
        return False

    def close(self):
        self.commander.publish_motion_enable(False)
        self.pose_keepalive_timer.shutdown()
        self.cameras.close()
        self.client.ws.close()
