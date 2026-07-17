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
from galaxea_a1_runtime.apps.eef_policy_actions import (
    build_action_transform_config,
    gripper_stroke_from_norm,
)
from galaxea_a1_runtime.apps.eef_policy_executor import (
    EefPolicyExecutor,
    close_policy_resources,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.eef_policy_state import EefPolicyState
from galaxea_a1_runtime.apps.eef_policy_review import EefActionReviewer
from galaxea_a1_runtime.apps.lingbot.rollout import LingBotActionChunk
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata
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
        self.reported_condition_state = False
        self.pose_keepalive_timer = None
        self.cameras = None
        self.client = None
        self.action_config = build_action_transform_config(
            system=config.system,
            servo_gain=config.servo.gain,
            servo_max_extra_m=config.servo.max_extra_m,
        )
        self.state = EefPolicyState(
            action_config=self.action_config,
            pose_mode=config.action.pose_mode,
            max_feedback_age_s=self.eef.max_feedback_age_s,
            initial_action8=self.execution.initial_ee_pose,
            frame_chunk_size=config.policy_server.frame_chunk_size,
            action_per_frame=config.policy_server.action_per_frame,
        )
        self.relay = RelayMonitor(self.system.relay.max_status_age_s)
        self.reviewer = EefActionReviewer(
            state=self.state,
            action_config=self.action_config,
            review_deadband_m=self.execution.review_deadband_m,
            servo_gain=self.servo.gain,
            orientation_mode=self.eef.orientation_mode,
            execute=self.execution.execute,
            policy_label="LingBot",
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
        self.executor = EefPolicyExecutor(
            state=self.state,
            relay=self.relay,
            commander=self.commander,
            relay_enable_timeout_s=self.system.relay.enable_timeout_s,
            settle_s=self.servo.settle_s,
            tolerance_m=self.servo.tolerance_m,
            corrections=self.servo.corrections,
            is_shutdown=rospy.is_shutdown,
            policy_label="LingBot",
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
                rospy.Duration(0.05), self.executor.publish_active_pose_target
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
                expected_metadata=server_metadata(config),
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

    def _hold_current_pose(self) -> None:
        pose = self.state.current_pose_message()
        if pose is None:
            raise RuntimeError("Cannot hold EE pose before receiving feedback")
        self._set_active_pose_target(pose)
        self.executor.publish_active_pose_target()

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

    def _publish_ee_action(self, policy_action: np.ndarray) -> np.ndarray:
        last_command = self.executor.publish(policy_action)

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
        timer, self.pose_keepalive_timer = self.pose_keepalive_timer, None
        cameras, self.cameras = self.cameras, None
        client, self.client = self.client, None
        close_policy_resources(
            policy_label="LingBot",
            executor=self.executor,
            timer=timer,
            cameras=cameras,
            client=client,
        )
