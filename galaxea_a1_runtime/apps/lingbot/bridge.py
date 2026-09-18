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

from embodied_ops.operator_panel import announce_input, announce_progress
from galaxea_a1_runtime.hardware.eef_bridge import EefIkCommandPublisher
from galaxea_a1_runtime.policies.eef_actions import (
    build_action_transform_config,
    gripper_stroke_from_norm,
)
from galaxea_a1_runtime.apps.eef_policy_executor import (
    EefPolicyExecutor,
    close_policy_resources,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.lingbot.ik_subgoal import (
    IkSubgoalExecuted,
    intermediate_targets,
    pose_distance,
)
from galaxea_a1_runtime.apps.eef_policy_state import EefPolicyState
from galaxea_a1_runtime.apps.eef_policy_review import EefActionReviewer
from galaxea_a1_runtime.apps.lingbot.rollout import LingBotActionChunk
from galaxea_a1_runtime.apps.lingbot.run_loop import (
    reset_after_ik_rejection,
    run_lingbot_rollout,
)
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata
from galaxea_a1_runtime.apps.lingbot.motion_recording import MotionRecording
from galaxea_a1_runtime.apps.policy_camera import PolicyCameraSession
from embodied_ops import TaskPrompt
from galaxea_a1_runtime.console import (
    LiveStatusLine,
    Tone,
    info,
    step,
    style,
    success,
    warning,
)
from galaxea_a1_runtime.hardware.eef_ik import (
    A1EefIkTargetRejected,
    IkSolution,
    build_eef_ik_solver,
)
from galaxea_a1_runtime.runtime.ros_feedback import (
    A1JointStateCache,
    StagedCommandMonitor,
)
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control
from std_msgs.msg import Bool, String


from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.runtime.relay import RelayMonitor


class A1LingBotEEBridge:
    def __init__(
        self,
        config: LingBotConfig,
        task: TaskPrompt,
        *,
        run_id: str,
        front_video_filename: str,
        wrist_video_filename: str,
    ):
        self.config = config
        self.task = task
        self.run_id = run_id
        self.front_video_filename = front_video_filename
        self.wrist_video_filename = wrist_video_filename
        self.system = config.system
        self.execution = config.execution
        self.server = config.server
        self.eef = config.system.eef
        self.target_keepalive_timer = None
        self.cameras = None
        self.client = None
        self.motion_recording = None
        self.live_status = LiveStatusLine()
        self.panel_progress_phase = ""
        self.actions_executed = 0
        self.last_inference_s = 0.0
        self.action_config = build_action_transform_config(system=config.system)
        self.state = EefPolicyState(
            action_config=self.action_config,
            pose_mode=config.action.pose_mode,
            max_feedback_age_s=self.eef.max_feedback_age_s,
        )
        self.joints = A1JointStateCache(config.system.joint_safety.names)
        self.staged = StagedCommandMonitor()
        self.ik_solver = build_eef_ik_solver(config.system)
        self.relay = RelayMonitor(self.system.relay.max_status_age_s)
        self.reviewer = EefActionReviewer(
            state=self.state,
            action_config=self.action_config,
            review_deadband_m=self.execution.review_deadband_m,
            execute=self.execution.execute,
            policy_label="LingBot",
        )
        topics = self.system.topics
        rospy.init_node("lingbot_va_ee_bridge", anonymous=False)
        gripper_pub = rospy.Publisher(
            topics.gripper_target, gripper_position_control, queue_size=10
        )
        motion_enable_pub = rospy.Publisher(
            topics.motion_enable, Bool, queue_size=1, latch=True
        )
        self.commander = EefIkCommandPublisher(
            rospy=rospy,
            target_pub=rospy.Publisher(topics.joint_target, JointState, queue_size=10),
            gripper_pub=gripper_pub,
            motion_enable_pub=motion_enable_pub,
            joint_state_msg_type=JointState,
            bool_msg_type=Bool,
            gripper_msg_type=gripper_position_control,
            joint_names=config.system.joint_safety.names,
            current_joint_positions=lambda: self.joints.positions(
                max_age_s=self.system.joint_safety.max_feedback_age_s
            ),
            solver=self.ik_solver,
            gripper_to_stroke=lambda value: gripper_stroke_from_norm(
                value, self.action_config
            ),
            execute=self.execution.execute,
            log_solutions=self.execution.print_actions,
        )
        self.executor = EefPolicyExecutor(
            relay=self.relay,
            commander=self.commander,
            staged_monitor=self.staged,
            relay_enable_timeout_s=self.system.relay.enable_timeout_s,
            staged_wait_timeout_s=self.system.startup.topic_timeout_s,
            staged_max_age_s=self.system.relay.max_input_age_s,
            staged_alignment_tolerance_rad=(
                self.system.joint_safety.initial_alignment_tolerance_rad
            ),
            is_shutdown=rospy.is_shutdown,
            policy_label="LingBot",
        )
        rospy.Subscriber(
            topics.eef_pose, PoseStamped, self.state.pose_callback, queue_size=1
        )
        rospy.Subscriber(
            topics.joint_states, JointState, self.joints.callback, queue_size=1
        )
        rospy.Subscriber(
            topics.staged_command, arm_control, self.staged.callback, queue_size=1
        )
        rospy.Subscriber(
            topics.gripper_feedback,
            JointState,
            self.state.gripper_callback,
            queue_size=1,
        )
        rospy.Subscriber(topics.relay_status, String, self.relay.callback, queue_size=1)
        try:
            self.motion_recording = MotionRecording(
                config.recording.output_root / f".{run_id}.motion",
                run_id=run_id,
                joint_names=self.system.joint_safety.names,
            )
            self.motion_recording.snapshot(self.system.eef_ik.urdf, "robot.urdf")
            self.motion_recording.snapshot(self.system.path, "system.toml")
            self.motion_recording.snapshot(config.path, "deployment.toml")
            if self.system.eef_ik.backend == "trac_ik":
                self.motion_recording.snapshot(
                    self.system.eef_ik.trac_ik.binary.with_suffix(".json"),
                    "trac_ik_build.json",
                )
            for topic, message_type, kind in (
                (topics.joint_states, JointState, "measured_joints"),
                (topics.joint_target, JointState, "joint_target"),
                (topics.staged_command, arm_control, "staged_command"),
                (topics.host_command, arm_control, "forwarded_command"),
                (topics.eef_pose, PoseStamped, "measured_eef"),
                (topics.gripper_feedback, JointState, "measured_gripper"),
                (topics.gripper_target, gripper_position_control, "gripper_target"),
                (topics.relay_status, String, "relay_status"),
            ):
                self.motion_recording.subscribe(rospy, topic, message_type, kind)
            self.commander.event_sink = self.motion_recording.record
            info(f"Motion recording armed: {self.motion_recording.directory}")
            self.target_keepalive_timer = rospy.Timer(
                rospy.Duration(0.05), self.executor.publish_active_target
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
            self.client.reset(self.task.prompt)
            if config.recording.camera_videos_enabled:
                front_path, wrist_path = self.cameras.start_camera_recording(
                    output_root=config.recording.output_root,
                    run_id=self.run_id,
                    front_video_filename=self.front_video_filename,
                    wrist_video_filename=self.wrist_video_filename,
                )
                info(
                    "Paired camera recording armed: "
                    f"front={front_path}, wrist={wrist_path}"
                )
        except BaseException as init_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "LingBot bridge initialization and cleanup failed",
                    [init_error, cleanup_error],
                ) from None
            raise

    def _read_lingbot_obs(self) -> dict | None:
        if self.cameras is None:
            raise RuntimeError("LingBot cameras are closed")
        obs = self.cameras.read_observation()
        if obs is None:
            return None
        return {"obs": [obs], "prompt": self.task.prompt}

    def _wait_for_fresh_feedback(self) -> None:
        deadline = time.monotonic() + self.eef.feedback_wait_timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            joints = self.joints.positions(
                max_age_s=self.system.joint_safety.max_feedback_age_s
            )
            if joints is not None and self.state.pose_is_fresh():
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No fresh named joint and pose feedback within "
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

        if self.execution.print_actions:
            grip_mm = gripper_stroke_from_norm(
                float(policy_action[7]), self.action_config
            )
            info(
                "Published xyz="
                f"{np.round(policy_action[:3], 4).tolist()} "
                f"cmd_xyz={np.round(last_command[:3], 4).tolist()} "
                f"quat={np.round(policy_action[3:7], 4).tolist()} "
                f"gripper_mm={round(float(grip_mm), 3)}"
            )
        return last_command

    @staticmethod
    def _ask_next(prompt: str) -> str:
        announce_input(("enter", "quit"))
        try:
            return input(prompt).strip().lower()
        except EOFError:
            return "q"

    def run(self) -> None:
        if self.client is None or self.cameras is None:
            raise RuntimeError("LingBot bridge is closed")
        run_lingbot_rollout(self, is_shutdown=rospy.is_shutdown)

    def _recover_ik_rejection(self) -> None:
        if self.client is None:
            raise RuntimeError("LingBot client is closed")
        self.motion_recording.record(
            "recovery_started", {"episode_origin": self.state.episode_origin.tolist()}
        )
        reset_after_ik_rejection(
            executor=self.executor,
            state=self.state,
            client=self.client,
            prompt=self.task.prompt,
            wait_for_feedback=self._wait_for_fresh_feedback,
        )
        info(
            "IK replan: current-joint hold established; temporal cache reset; "
            f"new episode origin={self.state.episode_origin.tolist()}"
        )
        self.motion_recording.record(
            "recovery_completed", {"episode_origin": self.state.episode_origin.tolist()}
        )

    def _prepare_execution(self) -> None:
        if not self.execution.execute:
            return
        self._wait_for_fresh_feedback()
        self.motion_recording.require_streams(
            ("measured_joints", "staged_command", "measured_eef")
        )
        if self._ensure_episode_origin() is None:
            raise RuntimeError("Cannot establish the episode EEF origin")
        self.executor.activate_current_hold()
        info("Relay activated on a fresh current-joint hold.")
        if self.execution.step_mode:
            info("Holding the current EE pose while waiting for Enter.")
            return
        info(
            "Continuous execution armed: "
            f"calls={self.execution.max_model_calls or 'unbounded'} "
            f"frames_per_call={self.execution.execute_frames} "
            f"rate={self.execution.exec_rate:.1f}Hz "
            f"ik_replan_max_attempts={self.execution.ik_replan_max_attempts} "
            f"ik_subgoal_enabled={self.execution.ik_subgoal.enabled} "
            "cache_action_source=requested-action"
        )
        self._update_live_status(0, phase="READY", force=True)

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
        self.motion_recording.check()
        observation = None
        while observation is None and not rospy.is_shutdown():
            observation = self._read_lingbot_obs()
            if observation is None:
                time.sleep(0.01)
        if rospy.is_shutdown():
            return None
        self._update_live_status(call_index, phase="INFER")
        started = time.monotonic()
        response = self.client.infer(observation)
        elapsed = time.monotonic() - started
        self.last_inference_s = elapsed
        chunk = LingBotActionChunk.from_response(
            response["action"],
            expected_shape=(
                len(self.config.policy_server.action_channel_ids),
                self.config.policy_server.frame_chunk_size,
                self.config.policy_server.action_per_frame,
            ),
            first=first,
            execute_frames=self.execution.execute_frames,
            observations_per_frame=self.execution.kv_observations_per_frame,
        )
        self._update_live_status(call_index, phase="EXECUTE")
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
            validated_action = self.state.validate(raw_action)
            self.motion_recording.record(
                "policy_target",
                {
                    "call_index": call_index,
                    "frame_index": frame_index,
                    "step_index": step_index,
                    "actions_executed": self.actions_executed,
                    "model_action": raw_action.tolist(),
                    "absolute_action": validated_action.tolist(),
                    "episode_origin": (
                        None
                        if self.state.episode_origin is None
                        else self.state.episode_origin.tolist()
                    ),
                },
            )
            chunk.cache_state[:, cache_frame_index, step_index] = (
                self.state.absolute_to_model(validated_action)
            )
            if self.execution.print_actions and (
                step_index == 0 or self.execution.step_actions
            ):
                self.reviewer.print_step(
                    call_index=call_index,
                    frame_index=frame_index,
                    step_index=step_index,
                    model_action=raw_action,
                    validated_action=validated_action,
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
                if not self.executor.motion_enabled:
                    self.live_status.break_line()
                try:
                    executed = self._publish_ee_action(validated_action)
                except A1EefIkTargetRejected:
                    if (
                        self.execution.ik_subgoal.enabled
                        and call_index + 1 < self.execution.max_model_calls
                    ):
                        self._try_ik_subgoal(validated_action)
                    raise
                chunk.cache_state[:, cache_frame_index, step_index] = (
                    self.state.absolute_to_model(executed)
                )
                self.actions_executed += 1
                self._update_live_status(
                    call_index,
                    phase="EXECUTE",
                    target=validated_action[:3],
                )
            time.sleep(1.0 / self.execution.exec_rate)
            if chunk.needs_observation_after(step_index):
                final_step = (
                    frame_index == chunk.end_frame - 1
                    and step_index == chunk.values.shape[2] - 1
                )
                if (
                    self.execution.execute
                    and self.execution.settle.enabled
                    and final_step
                ):
                    observation = self._settled_observation(call_index, executed)
                else:
                    observation = self._read_lingbot_obs()
                if observation is None:
                    raise RuntimeError(
                        "Camera frame unavailable during KV-cache collection"
                    )
                key_frames.extend(observation["obs"])
        return False, key_frames, cache_eligible

    def _settled_observation(self, call_index: int, target: np.ndarray) -> dict:
        """Wait on measured stillness, then capture the final KV history frame."""
        config = self.execution.settle
        started = time.monotonic()
        deadline = started + config.timeout_s
        stable_since = None
        low = high = None
        previous_stamps = (started, started)
        self._update_live_status(call_index, phase="SETTLE", force=True)
        self.motion_recording.record("settle_start", {"call_index": call_index})
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.executor.enable_motion()
            self.motion_recording.check()
            _, joint_stamp = self.joints.cache.snapshot()
            _, gripper_stamp = self.state.gripper_feedback.snapshot()
            joints = self.joints.positions(
                max_age_s=self.system.joint_safety.max_feedback_age_s
            )
            actual = self.state.current_absolute_action()
            if joints is None or actual is None:
                raise RuntimeError(
                    "Feedback became stale while waiting for settled observation"
                )
            if np.any(joints < self.ik_solver.lower_limits) or np.any(
                joints > self.ik_solver.upper_limits
            ):
                raise RuntimeError(
                    "Joint limits exceeded while waiting for settled observation"
                )
            if (
                joint_stamp is not None
                and gripper_stamp is not None
                and joint_stamp > previous_stamps[0]
                and gripper_stamp > previous_stamps[1]
            ):
                previous_stamps = (joint_stamp, gripper_stamp)
                now = time.monotonic()
                values = np.r_[joints, actual[7]]
                if low is None:
                    low = high = values.copy()
                    stable_since = now
                next_low, next_high = np.minimum(low, values), np.maximum(high, values)
                span = next_high - next_low
                if (
                    np.max(span[:-1]) > config.joint_range_rad
                    or span[-1] > config.gripper_range
                ):
                    low = high = values.copy()
                    stable_since = now
                else:
                    low, high = next_low, next_high
                if (
                    now - started >= config.min_wait_s
                    and now - stable_since >= config.stable_window_s
                ):
                    # Camera Bridge timestamps use perf_counter, feedback uses monotonic.
                    barrier = time.perf_counter()
                    self.motion_recording.record(
                        "settle_complete",
                        {
                            "call_index": call_index,
                            "elapsed_s": now - started,
                            "stable_s": now - stable_since,
                            "joint_range_rad": float(np.max((high - low)[:-1])),
                            "gripper_range": float((high - low)[-1]),
                            "actual_action": actual.tolist(),
                            "target_action": target.tolist(),
                            "position_error_m": float(
                                np.linalg.norm(actual[:3] - target[:3])
                            ),
                            "camera_after_perf_counter_s": barrier,
                        },
                    )
                    if self.cameras is None:
                        raise RuntimeError("Cameras closed during settle")
                    while not rospy.is_shutdown() and time.monotonic() < deadline:
                        self.executor.enable_motion()
                        obs = self.cameras.read_observation(after_monotonic_s=barrier)
                        if obs is not None:
                            self.motion_recording.record(
                                "settled_history_observation",
                                {
                                    "call_index": call_index,
                                    "camera_after_perf_counter_s": barrier,
                                    "elapsed_s": time.monotonic() - started,
                                },
                            )
                            return {"obs": [obs], "prompt": self.task.prompt}
                        time.sleep(1.0 / self.execution.exec_rate)
                    break
            time.sleep(1.0 / self.execution.exec_rate)
        raise RuntimeError("Settled history observation timed out or was interrupted")

    def _try_ik_subgoal(self, requested: np.ndarray) -> None:
        config = self.execution.ik_subgoal
        self.live_status.break_line()
        self.executor.hold_for_replan()
        self._wait_for_fresh_feedback()
        joints = self.joints.positions(
            max_age_s=self.system.joint_safety.max_feedback_age_s
        )
        if joints is None:
            raise RuntimeError("IK subgoal requires fresh named joint feedback")
        xyz, quat = self.ik_solver.forward(joints)
        start = np.concatenate([xyz, quat])

        def validate_solution(solution: IkSolution) -> None:
            solved_xyz, _ = self.ik_solver.forward(solution.joint_positions)
            if np.any(solved_xyz < self.eef.xyz_min) or np.any(
                solved_xyz > self.eef.xyz_max
            ):
                raise A1EefIkTargetRejected("IK subgoal solution leaves the workspace")

        for attempt, (fraction, candidate) in enumerate(
            intermediate_targets(xyz, quat, requested, config), start=1
        ):
            distance, rotation = pose_distance(xyz, quat, candidate)
            if (
                distance <= self.ik_solver.position_tolerance_m
                and rotation <= self.ik_solver.orientation_tolerance_rad
            ):
                break  # A numerically successful hold is not progress.
            if np.any(candidate[:3] < self.eef.xyz_min) or np.any(
                candidate[:3] > self.eef.xyz_max
            ):
                continue
            try:
                solution = self.executor.publish_subgoal(
                    candidate,
                    max_joint_delta_rad=config.max_joint_delta_rad,
                    validate_solution=validate_solution,
                )
            except A1EefIkTargetRejected as exc:
                warning(f"IK subgoal {attempt}/{config.max_attempts} rejected: {exc}")
                continue
            published_at = time.monotonic()
            self.motion_recording.record(
                "subgoal_published",
                {
                    "fraction": fraction,
                    "candidate": candidate.tolist(),
                    "requested": requested.tolist(),
                },
            )
            info(
                f"IK subgoal published: fraction={fraction:.5f} "
                f"translation_m={distance:.5f} rotation_rad={rotation:.5f} "
                f"max_joint_delta_rad={solution.max_joint_delta_rad:.5f}; "
                "gripper unchanged, awaiting measured progress"
            )
            self.actions_executed += 1
            self._wait_for_subgoal_feedback(candidate, start, requested, published_at)
            self.motion_recording.record(
                "subgoal_confirmed", {"candidate": candidate.tolist()}
            )
            raise IkSubgoalExecuted(
                "intermediate pose reached; discarding the remaining chunk and cache"
            )
        warning("No converged intermediate target with measurable progress; holding")

    def _wait_for_subgoal_feedback(
        self,
        candidate: np.ndarray,
        start: np.ndarray,
        requested: np.ndarray,
        published_at: float,
    ) -> None:
        deadline = published_at + self.execution.ik_subgoal.feedback_timeout_s
        ik = self.ik_solver
        initial_error = pose_distance(start[:3], start[3:7], requested)

        def score(errors: tuple[float, float]) -> float:
            return (errors[0] / ik.position_tolerance_m) ** 2 + (
                errors[1] / ik.orientation_tolerance_rad
            ) ** 2

        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.executor.enable_motion()
            _, updated_at = self.joints.cache.snapshot()
            joints = self.joints.positions(
                max_age_s=self.system.joint_safety.max_feedback_age_s
            )
            if joints is None:
                raise RuntimeError("Joint feedback became stale during IK subgoal")
            if np.any(joints < ik.lower_limits) or np.any(joints > ik.upper_limits):
                raise RuntimeError("Joint feedback exceeded limits during IK subgoal")
            if updated_at is not None and updated_at > published_at:
                xyz, quat = ik.forward(joints)
                error = pose_distance(xyz, quat, candidate)
                moved = pose_distance(xyz, quat, start)
                remaining = pose_distance(xyz, quat, requested)
                if (
                    error[0] <= ik.position_tolerance_m
                    and error[1] <= ik.orientation_tolerance_rad
                    and (
                        moved[0] > ik.position_tolerance_m
                        or moved[1] > ik.orientation_tolerance_rad
                    )
                    and score(remaining) < score(initial_error)
                ):
                    info(
                        "IK subgoal feedback confirmed: "
                        f"xyz={xyz.tolist()} quat={quat.tolist()} "
                        f"remaining_position_m={remaining[0]:.5f} "
                        f"remaining_orientation_rad={remaining[1]:.5f}"
                    )
                    return
            time.sleep(1.0 / self.execution.exec_rate)
        raise RuntimeError(
            "IK subgoal did not produce confirmed progress before timeout"
        )

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
        if self.execution.execute and key_frames and cache_eligible:
            try:
                self._update_live_status(call_index, phase="CACHE")
                self.client.infer(
                    {
                        "obs": key_frames,
                        "compute_kv_cache": True,
                        "imagine": False,
                        "state": chunk.cache_state,
                    }
                )
                self._update_live_status(call_index, phase="CACHE-SYNCED")
                return True
            except Exception as exc:
                self.live_status.break_line()
                warning(f"compute_kv_cache failed: {exc}")
                self.client.reset(self.task.prompt)
                info("Server reset after KV-cache failure.")
        elif self.execution.execute and not cache_eligible:
            self.client.reset(self.task.prompt)
            info("Server reset because one or more actions were skipped.")
        return False

    def _expected_action_steps(self) -> int | None:
        calls = self.execution.max_model_calls
        if calls <= 0:
            return None
        frames = self.config.policy_server.frame_chunk_size
        actions = self.config.policy_server.action_per_frame
        first_frames = max(0, min(frames, 1 + self.execution.execute_frames) - 1)
        later_frames = min(frames, self.execution.execute_frames)
        return (first_frames + max(0, calls - 1) * later_frames) * actions

    def _update_live_status(
        self,
        call_index: int,
        *,
        phase: str,
        target: np.ndarray | None = None,
        force: bool = False,
    ) -> None:
        call_total = self.execution.max_model_calls or "∞"
        action_total = self._expected_action_steps()
        action_label = (
            str(self.actions_executed)
            if action_total is None
            else f"{self.actions_executed}/{action_total}"
        )
        parts = [
            f"LingBot {phase}",
            f"call {call_index + 1}/{call_total}",
            f"action {action_label}",
        ]
        current = self.state.current_xyz()
        if current is not None:
            parts.append(
                "eef=" + ",".join(f"{value:.3f}" for value in current.tolist())
            )
        if target is not None and current is not None:
            delta_cm = float(np.linalg.norm(np.asarray(target) - current) * 100.0)
            parts.append(f"target_delta={delta_cm:.1f}cm")
        if self.last_inference_s > 0:
            parts.append(f"infer={self.last_inference_s:.2f}s")
        if self.cameras is not None:
            progress = self.cameras.recording_progress()
            if progress is not None:
                frames, elapsed = progress
                parts.append(f"REC={elapsed:05.1f}s/{frames}f")
        message = " | ".join(parts)
        panel_mode = announce_progress(
            "inference",
            "LingBot inference",
            call_index + 1,
            self.execution.max_model_calls or None,
            phase=phase,
            detail=" | ".join(parts[1:]),
            force=force or phase != self.panel_progress_phase,
        )
        self.panel_progress_phase = phase
        if not panel_mode:
            self.live_status.update(message, force=force)

    def close(self) -> None:
        self.live_status.close()
        timer, self.target_keepalive_timer = self.target_keepalive_timer, None
        cameras, self.cameras = self.cameras, None
        client, self.client = self.client, None
        try:
            close_policy_resources(
                policy_label="LingBot",
                executor=self.executor,
                ik_solver=self.ik_solver,
                timer=timer,
                cameras=cameras,
                client=client,
            )
        finally:
            if self.motion_recording is not None:
                self.motion_recording.close()
            result = None if cameras is None else cameras.recording_result
            if result is not None:
                success(
                    "Paired camera videos saved: "
                    f"front={result.front_path}, wrist={result.wrist_path} "
                    f"({result.frames} frames, {result.elapsed_s:.1f}s)"
                )
