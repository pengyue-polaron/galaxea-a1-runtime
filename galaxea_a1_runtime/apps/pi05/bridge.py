#!/usr/bin/env python3
# ruff: noqa: E402
"""OpenPI pi0.5 -> Galaxea A1 staged EEF bridge."""

from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR, remove_ros2=True)

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Bool, String

from galaxea_a1_runtime.apps.eef_bridge import EefCommandPublisher
from galaxea_a1_runtime.apps.eef_policy_actions import (
    build_action_transform_config,
    gripper_stroke_from_norm,
)
from galaxea_a1_runtime.apps.eef_policy_executor import (
    EefPolicyExecutor,
    close_policy_resources,
)
from galaxea_a1_runtime.apps.eef_policy_review import EefActionReviewer
from galaxea_a1_runtime.apps.eef_policy_state import EefPolicyState
from galaxea_a1_runtime.apps.pi05.client import Pi05Client
from galaxea_a1_runtime.apps.pi05.config_schema import Pi05Config
from galaxea_a1_runtime.apps.pi05.protocol import server_metadata
from galaxea_a1_runtime.apps.policy_camera import PolicyCameraSession
from galaxea_a1_runtime.console import Tone, info, step, style, success
from galaxea_a1_runtime.runtime.relay import RelayMonitor
from galaxea_a1_runtime.runtime.ros_feedback import A1JointStateCache


class A1Pi05EEBridge:
    def __init__(self, config: Pi05Config) -> None:
        self.config = config
        self.system = config.system
        self.execution = config.execution
        self.servo = config.servo
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
            pose_mode=config.model_contract.pose_mode,
            max_feedback_age_s=config.system.eef.max_feedback_age_s,
            initial_action8=None,
            frame_chunk_size=1,
            action_per_frame=1,
        )
        self.joints = A1JointStateCache(config.system.joint_safety.names)
        self.relay = RelayMonitor(config.system.relay.max_status_age_s)
        self.reviewer = EefActionReviewer(
            state=self.state,
            action_config=self.action_config,
            review_deadband_m=config.execution.review_deadband_m,
            servo_gain=config.servo.gain,
            orientation_mode=config.system.eef.orientation_mode,
            execute=config.execution.execute,
            policy_label="OpenPI pi0.5",
        )

        topics = config.system.topics
        rospy.init_node("openpi_pi05_ee_bridge", anonymous=False)
        self.commander = EefCommandPublisher(
            rospy=rospy,
            pose_pub=rospy.Publisher(topics.eef_target, PoseStamped, queue_size=10),
            gripper_pub=rospy.Publisher(
                topics.gripper_target, gripper_position_control, queue_size=10
            ),
            motion_enable_pub=rospy.Publisher(
                topics.motion_enable, Bool, queue_size=1, latch=True
            ),
            pose_msg_type=PoseStamped,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            command_frame=config.system.eef.command_frame,
            gripper_to_stroke=lambda value: gripper_stroke_from_norm(
                value, self.action_config
            ),
            execute=config.execution.execute,
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
            policy_label="OpenPI pi0.5",
        )
        rospy.Subscriber(
            topics.eef_pose, PoseStamped, self.state.pose_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.joint_states, JointState, self.joints.callback, queue_size=1
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
                config.system,
                front_key=config.observations.front_key,
                wrist_key=config.observations.wrist_key,
            )
            self.client = Pi05Client(
                config.server.host,
                config.server.port,
                connect_timeout_s=config.server.connect_timeout_s,
                close_timeout_s=config.server.close_timeout_s,
                expected_metadata=server_metadata(config),
            )
        except BaseException as init_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "pi0.5 bridge initialization and cleanup failed",
                    [init_error, cleanup_error],
                ) from None
            raise

    def run(self) -> None:
        if self.client is None or self.cameras is None:
            raise RuntimeError("pi0.5 bridge is closed")
        call_index = 0
        while not rospy.is_shutdown():
            if (
                self.execution.max_model_calls
                and call_index >= self.execution.max_model_calls
            ):
                return
            if self.execution.step_mode and not self._wait_for_operator(call_index):
                return
            observation = self._read_observation()
            step(f"Inference #{call_index + 1}: observation ready; pi0.5 running")
            started = time.monotonic()
            response = self.client.infer(observation)
            actions = self._validated_actions(response)
            success(
                f"Inference #{call_index + 1} done: "
                f"infer={time.monotonic() - started:.3f}s action_shape={actions.shape}"
            )
            if self._run_actions(call_index, actions):
                return
            call_index += 1

    def _read_observation(self) -> dict[str, object]:
        self._wait_for_feedback()
        origin_was_missing = self.state.episode_origin is None
        origin = self.state.ensure_episode_origin()
        if origin is None:
            raise RuntimeError("Cannot establish the pi0.5 episode EEF origin")
        if origin_was_missing:
            info(
                "Episode EEF origin: "
                f"xyz={np.round(origin[:3], 4).tolist()} "
                f"quat={np.round(origin[3:7], 4).tolist()} "
                f"pose_mode={self.config.model_contract.pose_mode}"
            )
        current = self.state.current_absolute_action()
        joints = self.joints.positions(
            max_age_s=self.system.joint_safety.max_feedback_age_s
        )
        if current is None or joints is None:
            raise RuntimeError("Fresh EEF, joint, and gripper state is required")
        state = np.asarray((*current[:7], *joints, current[7]), dtype=np.float32)
        if state.shape != (self.config.model_contract.state_dim,):
            raise RuntimeError(f"Unexpected pi0.5 state shape: {state.shape}")
        images = self.cameras.wait_pair(
            timeout_s=self.system.joint_safety.state_timeout_s,
            is_shutdown=rospy.is_shutdown,
        )
        front_bgr, wrist_bgr = images
        return {
            self.config.observations.front_key: front_bgr[..., ::-1].copy(),
            self.config.observations.wrist_key: wrist_bgr[..., ::-1].copy(),
            "observation/state": state,
            "prompt": self.config.server.prompt,
        }

    def _wait_for_feedback(self) -> None:
        timeout_s = max(
            self.system.eef.feedback_wait_timeout_s,
            self.system.joint_safety.state_timeout_s,
        )
        deadline = time.monotonic() + timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            joints = self.joints.positions(
                max_age_s=self.system.joint_safety.max_feedback_age_s
            )
            if (
                joints is not None
                and self.state.pose_is_fresh()
                and self.state.gripper_is_fresh()
                and self.state.current_quat() is not None
            ):
                return
            time.sleep(0.02)
        raise RuntimeError("No fresh EEF, named joint, and gripper feedback for pi0.5")

    def _validated_actions(self, response: dict[str, object]) -> np.ndarray:
        if "actions" not in response:
            raise RuntimeError(f"pi0.5 response is missing actions: {sorted(response)}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        expected = (
            self.config.model_contract.action_horizon,
            self.config.model_contract.source_action_dim,
        )
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(
                f"Invalid pi0.5 action tensor: expected finite {expected}, got {actions.shape}"
            )
        return actions

    def _run_actions(self, call_index: int, actions: np.ndarray) -> bool:
        count = self.execution.execute_actions_per_inference
        for action_index, raw_action in enumerate(actions[:count]):
            safe_action = self.state.prepare(
                raw_action, require_orientation=self.execution.execute
            )
            if self.execution.print_actions:
                self.reviewer.print_step(
                    call_index=call_index,
                    frame_index=0,
                    step_index=action_index,
                    model_action=raw_action,
                    safe_action=safe_action,
                )
            if self.execution.step_actions:
                command = self._ask("Next=publish this EEF action, s=skip, q=quit: ")
                if command in {"q", "quit", "exit"}:
                    return True
                if command in {"s", "skip"}:
                    continue
            if self.execution.execute:
                self._publish_ee_action(safe_action)
            time.sleep(1.0 / self.execution.exec_rate)
        return False

    def _publish_ee_action(self, policy_action: np.ndarray) -> None:
        self.executor.publish(policy_action)

    def _wait_for_operator(self, call_index: int) -> bool:
        step(f"Inference #{call_index + 1} ready. Enter=run, q=quit.")
        return self._ask(style(f"Inference #{call_index + 1} > ", Tone.STEP)) not in {
            "q",
            "quit",
            "exit",
        }

    @staticmethod
    def _ask(prompt: str) -> str:
        try:
            return input(prompt).strip().lower()
        except EOFError:
            return "q"

    def close(self) -> None:
        timer, self.pose_keepalive_timer = self.pose_keepalive_timer, None
        cameras, self.cameras = self.cameras, None
        client, self.client = self.client, None
        close_policy_resources(
            policy_label="pi0.5",
            executor=self.executor,
            timer=timer,
            cameras=cameras,
            client=client,
        )
