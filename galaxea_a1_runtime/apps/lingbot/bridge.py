#!/usr/bin/env python3
# ruff: noqa: E402
"""LingBot-VA -> Galaxea A1 end-effector pose bridge implementation.

Execution and all hardware settings come from the tracked deployment config.
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
    build_action_transform_config,
    gripper_stroke_from_norm,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.lingbot.episode_state import LingBotEpisodeState
from galaxea_a1_runtime.apps.lingbot.review import LingBotActionReviewer
from galaxea_a1_runtime.apps.lingbot.rollout import LingBotActionChunk
from galaxea_a1_runtime.apps.policy_camera import PolicyCameraSession
from galaxea_a1_runtime.console import Tone, info, step, style, success, warning
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String


from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.runtime.relay import RelayMonitor


class A1LingBotEEBridge:
    def __init__(self, config: LingBotConfig):
        self.config = config
        self.system = config.system
        self.execution = config.execution
        self.servo = config.servo
        self.server = config.server
        self.eef = config.system.eef
        self.motion_enabled = False
        self.reported_condition_state = False
        self.pose_keepalive_timer = None
        self.cameras = None
        self.client = None
        self.action_config = build_action_transform_config(config)
        self.state = LingBotEpisodeState(
            action_config=self.action_config,
            pose_mode=config.action.pose_mode,
            max_feedback_age_s=self.eef.max_feedback_age_s,
            initial_action8=self.execution.initial_ee_pose,
            frame_chunk_size=config.policy_server.frame_chunk_size,
            action_per_frame=config.policy_server.action_per_frame,
        )
        self.relay = RelayMonitor(self.system.relay.max_status_age_s)
        self.reviewer = LingBotActionReviewer(
            state=self.state,
            action_config=self.action_config,
            review_deadband_m=self.execution.review_deadband_m,
            servo_gain=self.servo.gain,
            orientation_mode=self.eef.orientation_mode,
            execute=self.execution.execute,
        )
        topics = self.system.topics
        rospy.init_node("lingbot_va_ee_bridge", anonymous=False)
        pose_pub = rospy.Publisher(topics.eef_target, PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher(
            topics.gripper_target, gripper_position_control, queue_size=10
        )
        motion_enable_pub = rospy.Publisher(
            topics.motion_enable, Bool, queue_size=1, latch=True
        )
        self.commander = EefCommandPublisher(
            rospy=rospy,
            pose_pub=pose_pub,
            gripper_pub=gripper_pub,
            motion_enable_pub=motion_enable_pub,
            pose_msg_type=PoseStamped,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            command_frame=self.eef.command_frame,
            gripper_to_stroke=lambda value: gripper_stroke_from_norm(
                value, self.action_config
            ),
            execute=self.execution.execute,
        )
        rospy.Subscriber(
            topics.eef_pose, PoseStamped, self.state.pose_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.gripper_feedback,
            JointState,
            self.state.gripper_callback,
            queue_size=1,
        )
        rospy.Subscriber(topics.relay_status, String, self.relay.callback, queue_size=1)
        try:
            self.pose_keepalive_timer = rospy.Timer(
                rospy.Duration(0.05), self._publish_active_pose_target
            )
            self.cameras = PolicyCameraSession(
                self.system,
                front_key=config.observations.front_key,
                wrist_key=config.observations.wrist_key,
            )
            self.client = LingBotClient(
                self.server.host,
                self.server.port,
                connect_timeout_s=self.server.connect_timeout_s,
                close_timeout_s=self.server.close_timeout_s,
            )
            self.client.reset(self.server.prompt)
        except BaseException as init_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "LingBot bridge initialization and cleanup failed",
                    [init_error, cleanup_error],
                ) from None
            raise

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
        deadline = time.monotonic() + self.system.relay.enable_timeout_s
        last_state = "no status"
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status, _ = self.relay.status()
            last_state = self.relay.summary()
            if self.relay.is_active():
                self.motion_enabled = True
                success("Real arm command relay is ACTIVE.")
                return
            if status is not None and status.state == "FAULT":
                break
            time.sleep(0.05)
        self.commander.publish_motion_enable(False)
        raise RuntimeError(f"A1 relay did not become ACTIVE: {last_state}")

    def _read_lingbot_obs(self) -> dict | None:
        if self.cameras is None:
            raise RuntimeError("LingBot cameras are closed")
        obs = self.cameras.read_observation()
        if obs is None:
            return None
        packet = {"obs": [obs], "prompt": self.server.prompt}
        if self.execution.condition_on_ee_state:
            state = self.state.model_condition()
            if state is not None:
                packet["state"] = state
                if not self.reported_condition_state:
                    first = state[:, 0, 0]
                    info(
                        "Conditioning LingBot on EE state: "
                        f"xyz={np.round(first[:3], 4).tolist()} "
                        f"quat={np.round(first[3:7], 4).tolist()} "
                        f"gripper_norm={first[7]:.3f}"
                    )
                    self.reported_condition_state = True
            elif not self.reported_condition_state:
                warning(
                    "EE state conditioning requested, but no fresh/current or "
                    "initial_ee_pose is available."
                )
                self.reported_condition_state = True
        return packet

    def _wait_for_fresh_feedback(self) -> None:
        deadline = time.monotonic() + self.eef.feedback_wait_timeout_s
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
            f"{self.eef.feedback_wait_timeout_s:.1f}s; check "
            f"{self.system.host.a1_serial} and the A1 driver"
        )

    def _ensure_episode_origin(self) -> np.ndarray | None:
        existed = self.state.episode_origin is not None
        origin = self.state.ensure_episode_origin()
        if origin is None:
            return None
        if not existed:
            info(
                "Episode EEF origin: "
                f"xyz={np.round(origin[:3], 4).tolist()} "
                f"quat={np.round(origin[3:7], 4).tolist()} "
                f"pose_mode={self.config.action.pose_mode}"
            )
        return origin

    def _wait_for_target_tracking(
        self, policy_action8: np.ndarray, started: float
    ) -> float:
        if self.servo.settle_s <= 0:
            return float("nan")
        deadline = time.monotonic() + self.servo.settle_s
        err_norm = float("inf")
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            cur = self.state.current_xyz()
            if cur is not None:
                err = np.asarray(policy_action8[:3], dtype=np.float64) - cur
                err_norm = float(np.linalg.norm(err))
                if err_norm <= self.servo.tolerance_m:
                    break
            time.sleep(0.03)
        cur = self.state.current_xyz()
        if cur is not None:
            err_norm = float(
                np.linalg.norm(np.asarray(policy_action8[:3], dtype=np.float64) - cur)
            )
            info(
                "Tracking: "
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
        corrections = self.servo.corrections
        for correction_i in range(corrections):
            if err_norm <= self.servo.tolerance_m:
                break
            command = self.state.tracker_command(policy_action)
            if np.allclose(command[:3], last_command[:3], atol=1e-4):
                break
            last_command = command
            info(
                "Tracking correction: "
                f"{correction_i + 1}/{corrections} command_xyz={np.round(command[:3], 4).tolist()} "
                f"policy_xyz={np.round(policy_action[:3], 4).tolist()}"
            )
            self._publish_pose_and_gripper(command, publish_gripper=False)
            err_norm = self._wait_for_target_tracking(policy_action, started)

        grip_mm = gripper_stroke_from_norm(float(policy_action[7]), self.action_config)
        info(
            "Published xyz="
            f"{np.round(policy_action[:3], 4).tolist()} "
            f"cmd_xyz={np.round(last_command[:3], 4).tolist()} "
            f"quat={np.round(policy_action[3:7], 4).tolist()} "
            f"gripper_mm={round(float(grip_mm), 3)}"
        )
        return (
            self.state.measured_action(policy_action)
            if self.servo.cache_actual_feedback
            else last_command
        )

    @staticmethod
    def _ask_next(prompt: str) -> str:
        try:
            return input(prompt).strip().lower()
        except EOFError:
            return "q"

    def run(self) -> None:
        if self.client is None or self.cameras is None:
            raise RuntimeError("LingBot bridge is closed")
        self._prepare_execution()
        first = True
        call_index = 0
        while not rospy.is_shutdown():
            if (
                self.execution.max_model_calls > 0
                and call_index >= self.execution.max_model_calls
            ):
                return
            if not self._wait_for_inference_request(call_index):
                return
            if self.execution.no_kv_update and not first:
                self.client.reset(self.server.prompt)
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
            if self.execution.execute:
                first = not (self.execution.no_kv_update or cache_updated)
            success(
                f"Inference #{call_index + 1} execution complete. "
                "The next READY prompt is the boundary before another model call."
            )
            call_index += 1

    def _prepare_execution(self) -> None:
        if not self.execution.execute:
            return
        self._wait_for_fresh_feedback()
        if self._ensure_episode_origin() is None:
            raise RuntimeError("Cannot establish the episode EEF origin")
        self._hold_current_pose()
        rospy.sleep(1.0)
        if self.execution.step_mode:
            info("Holding the current EE pose while waiting for Enter.")
            return
        cache_source = (
            "measured-feedback"
            if self.servo.cache_actual_feedback
            else "tracker-command"
        )
        info(
            "Continuous execution armed: "
            f"calls={self.execution.max_model_calls or 'unbounded'} "
            f"frames_per_call={self.execution.execute_frames} "
            f"rate={self.execution.exec_rate:.1f}Hz "
            f"cache_action_source={cache_source}"
        )

    def _wait_for_inference_request(self, call_index: int) -> bool:
        if not self.execution.step_mode:
            return True
        step(f"Inference #{call_index + 1} ready. Enter=run one model call, q=quit.")
        command = self._ask_next(
            style(
                f"Inference #{call_index + 1} > ",
                Tone.STEP,
            )
        )
        return command not in {"q", "quit", "exit"}

    def _infer_chunk(
        self, call_index: int, *, first: bool
    ) -> LingBotActionChunk | None:
        if self.client is None:
            raise RuntimeError("LingBot client is closed")
        observation = None
        while observation is None and not rospy.is_shutdown():
            observation = self._read_lingbot_obs()
            if observation is None:
                time.sleep(0.01)
        if rospy.is_shutdown():
            return None
        step(f"Inference #{call_index + 1}: camera observation captured; model running")
        started = time.monotonic()
        response = self.client.infer(observation)
        elapsed = time.monotonic() - started
        chunk = LingBotActionChunk.from_response(
            response["action"],
            first=first,
            execute_frames=self.execution.execute_frames,
            observations_per_frame=self.execution.kv_observations_per_frame,
        )
        success(
            f"Inference #{call_index + 1} done: infer={elapsed:.3f}s "
            f"action_shape={chunk.values.shape} "
            f"obs_frames={len(observation['obs'])}"
        )
        return chunk

    def _execute_chunk(
        self,
        call_index: int,
        chunk: LingBotActionChunk,
    ) -> tuple[bool, list, bool]:
        if self.execution.step_actions:
            info(
                f"Execution #{call_index + 1}: "
                f"This inference produced {chunk.total_steps} EE steps.\n"
                f"The next {chunk.total_steps} Enter presses publish these existing steps; "
                "they DO NOT run new inference."
            )
        key_frames: list = []
        cache_eligible = True
        for frame_index, step_index, cache_frame_index, raw_action in chunk.steps():
            safe_action = self.state.prepare(
                raw_action, require_orientation=self.execution.execute
            )
            chunk.cache_state[:, cache_frame_index, step_index] = (
                self.state.absolute_to_model(safe_action)
            )
            if self.execution.print_actions and (
                step_index == 0 or self.execution.step_actions
            ):
                self.reviewer.print_step(
                    call_index=call_index,
                    frame_index=frame_index,
                    step_index=step_index,
                    model_action=raw_action,
                    safe_action=safe_action,
                )
            if self.execution.step_actions:
                command = self._ask_next(
                    "       Next=publish this EE step, s=skip, q=quit: "
                )
                if command in {"q", "quit", "exit"}:
                    return True, key_frames, cache_eligible
                if command in {"s", "skip"}:
                    cache_eligible = False
                    continue
            if self.execution.execute:
                executed = self._publish_ee_action(safe_action)
                chunk.cache_state[:, cache_frame_index, step_index] = (
                    self.state.absolute_to_model(executed)
                )
            time.sleep(1.0 / self.execution.exec_rate)
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
        if self.client is None:
            raise RuntimeError("LingBot client is closed")
        if (
            self.execution.execute
            and key_frames
            and cache_eligible
            and not self.execution.no_kv_update
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
                success(
                    f"Cache update #{call_index + 1}: context synchronized; "
                    "this did NOT generate new actions. "
                    f"rgb_frames={len(key_frames)} "
                    f"action_frames={chunk.cache_state.shape[1]}"
                )
                return True
            except Exception as exc:
                warning(f"compute_kv_cache failed: {exc}")
                self.client.reset(self.server.prompt)
                info("Server reset after KV-cache failure.")
        elif (
            self.execution.execute
            and not cache_eligible
            and not self.execution.no_kv_update
        ):
            self.client.reset(self.server.prompt)
            info("Server reset because one or more actions were skipped.")
        return False

    def close(self) -> None:
        cleanup_errors: list[BaseException] = []
        try:
            self.commander.publish_motion_enable(False)
            self.motion_enabled = False
        except BaseException as exc:  # Cleanup must continue.
            cleanup_errors.append(exc)
        timer, self.pose_keepalive_timer = self.pose_keepalive_timer, None
        if timer is not None:
            try:
                timer.shutdown()
            except BaseException as exc:  # Cleanup must continue.
                cleanup_errors.append(exc)
        cameras, self.cameras = self.cameras, None
        if cameras is not None:
            try:
                cameras.close()
            except BaseException as exc:  # Cleanup must continue.
                cleanup_errors.append(exc)
        client, self.client = self.client, None
        if client is not None:
            try:
                client.close()
            except BaseException as exc:  # Cleanup must continue.
                cleanup_errors.append(exc)
        if cleanup_errors:
            raise BaseExceptionGroup("LingBot bridge cleanup failed", cleanup_errors)
