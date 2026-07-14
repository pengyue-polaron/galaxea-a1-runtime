#!/usr/bin/env python3
# ruff: noqa: E402
"""ACT joint-state policy bridge for Galaxea A1.

The bridge is dry-run by default. When --execute is enabled it publishes only
to the safe joint target topic, then relies on the isolated jointTracker and
safe relay to reach /arm_joint_command_host.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

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
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
    str(_ROS1_OVERLAY),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import cv2
import numpy as np
import torch

import rospy
from galaxea_a1_runtime.apps.eef_bridge import (
    RelayStatus,
    decode_relay_status,
    relay_state_summary,
    relay_status_is_fresh,
)
from galaxea_a1_runtime.hardware.cameras import (
    LatestCameraReader,
    RealSenseColorCamera,
    RealSenseFrameSet,
    open_color_camera,
)
from galaxea_a1_runtime.hardware.image_geometry import ImageRoi, crop_image
from galaxea_a1_runtime.hardware.web_preview import (
    CameraWebPreview,
    add_web_preview_arguments,
    color_from_bgr,
    color_from_frameset,
    web_preview_config_from_args,
)
from galaxea_a1_runtime.gripper import denormalize_stroke, normalize_stroke
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control, gripper_position_control
from std_msgs.msg import Bool, String


def log(message: str) -> None:
    print(message, flush=True)


class LatestCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._value: Any | None = None
        self._updated_monotonic: float | None = None

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._updated_monotonic = time.monotonic()

    def get(self) -> tuple[Any | None, float | None]:
        with self._lock:
            return self._value, self._updated_monotonic


class A1JointStateCache:
    def __init__(self, ordered_names: tuple[str, ...]):
        self.ordered_names = ordered_names
        self.cache = LatestCache()

    def callback(self, msg: JointState) -> None:
        self.cache.set(msg)

    def positions(self, *, max_age_s: float) -> tuple[float, ...] | None:
        msg, updated = self.cache.get()
        if msg is None or updated is None or time.monotonic() - updated > max_age_s:
            return None
        names = list(getattr(msg, "name", []))
        values = list(getattr(msg, "position", []))
        if len(values) < len(self.ordered_names):
            return None
        name_to_idx = {name: index for index, name in enumerate(names)}
        if names and all(name in name_to_idx for name in self.ordered_names):
            indices = [name_to_idx[name] for name in self.ordered_names]
            if all(index < len(values) for index in indices):
                return tuple(float(values[index]) for index in indices)
        return tuple(float(value) for value in values[: len(self.ordered_names)])

class GripperFeedbackCache:
    def __init__(self):
        self.cache = LatestCache()

    def callback(self, msg: JointState) -> None:
        self.cache.set(msg)

    def normalized(
        self,
        *,
        max_age_s: float,
        stroke_min_mm: float,
        stroke_max_mm: float,
    ) -> float | None:
        msg, updated = self.cache.get()
        if msg is None or updated is None or time.monotonic() - updated > max_age_s:
            return None
        values = list(getattr(msg, "position", []))
        if not values:
            return None
        return normalize_stroke(
            float(values[0]),
            stroke_min_mm=stroke_min_mm,
            stroke_max_mm=stroke_max_mm,
        )


class RelayMonitor:
    def __init__(self, max_status_age_s: float):
        self.max_status_age_s = max_status_age_s
        self.cache = LatestCache()

    def callback(self, msg: String) -> None:
        self.cache.set(decode_relay_status(msg.data))

    def status(self) -> tuple[RelayStatus | None, float | None]:
        value, updated = self.cache.get()
        return value, updated

    def summary(self) -> str:
        status, updated = self.status()
        return relay_state_summary(status, updated, max_age_s=self.max_status_age_s)

    def is_active(self) -> bool:
        status, updated = self.status()
        return (
            relay_status_is_fresh(updated, max_age_s=self.max_status_age_s)
            and (status or RelayStatus("UNKNOWN")).state == "ACTIVE"
        )


class StagedCommandMonitor:
    def __init__(self):
        self.cache = LatestCache()

    def callback(self, msg: arm_control) -> None:
        self.cache.set(msg)

    def max_error(self, target: Sequence[float], dof: int) -> float | None:
        msg, _ = self.cache.get()
        if msg is None or len(getattr(msg, "p_des", ())) < dof:
            return None
        staged = tuple(float(value) for value in msg.p_des[:dof])
        return max(abs(staged[index] - float(target[index])) for index in range(dof))


class ActPolicyRunner:
    def __init__(self, args: argparse.Namespace):
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"ACT checkpoint not found: {checkpoint}")
        log(f"[ACT] Loading checkpoint: {checkpoint}")
        cfg = PreTrainedConfig.from_pretrained(
            checkpoint,
            cli_overrides=[f"--device={args.device}"],
        )
        if args.disable_backbone_download and hasattr(cfg, "pretrained_backbone_weights"):
            cfg.pretrained_backbone_weights = None
        self.front_width, self.front_height = _policy_image_hw(
            cfg, "observation.images.front"
        )
        self.wrist_width, self.wrist_height = _policy_image_hw(
            cfg, "observation.images.wrist"
        )
        configured_front = (args.cam0_crop_width, args.cam0_crop_height)
        configured_wrist = (args.cam_width, args.cam_height)
        if (self.front_width, self.front_height) != configured_front:
            raise RuntimeError(
                "ACT checkpoint front image contract is "
                f"{self.front_width}x{self.front_height}, but configured AgentView crop is "
                f"{configured_front[0]}x{configured_front[1]}; retrain/register a matching checkpoint"
            )
        if (self.wrist_width, self.wrist_height) != configured_wrist:
            raise RuntimeError(
                "ACT checkpoint wrist image contract is "
                f"{self.wrist_width}x{self.wrist_height}, but configured Wrist source is "
                f"{configured_wrist[0]}x{configured_wrist[1]}; retrain/register a matching checkpoint"
            )
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(checkpoint, config=cfg, local_files_only=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.device = str(cfg.device)
        self.use_amp = bool(getattr(cfg, "use_amp", False))
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        log(
            "[ACT] Ready: "
            f"device={self.device} chunk={getattr(cfg, 'chunk_size', '?')} "
            f"n_action_steps={getattr(cfg, 'n_action_steps', '?')}"
        )

    def predict_chunk(
        self,
        *,
        front_bgr: np.ndarray,
        wrist_bgr: np.ndarray,
        state7: Sequence[float],
    ) -> np.ndarray:
        obs = {
            "observation.images.front": _bgr_to_chw_tensor(
                front_bgr, width=self.front_width, height=self.front_height
            ),
            "observation.images.wrist": _bgr_to_chw_tensor(
                wrist_bgr, width=self.wrist_width, height=self.wrist_height
            ),
            "observation.state": torch.tensor(tuple(float(v) for v in state7), dtype=torch.float32),
        }
        batch = self.preprocessor(obs)
        amp_context = (
            torch.autocast(device_type="cuda")
            if self.use_amp and self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[-1] != 7:
            raise RuntimeError(f"ACT returned unexpected action shape: {tuple(chunk.shape)}")
        return chunk[0].detach().cpu().numpy().astype(np.float64, copy=False)


class ActJointBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.target_names = tuple(args.target_joint_names)
        self.lower_limits = np.asarray(args.lower_limits, dtype=np.float64)
        self.upper_limits = np.asarray(args.upper_limits, dtype=np.float64)
        self.motion_enabled = False
        self.front_roi = _front_roi_from_args(args)

        rospy.init_node("act_joint_policy_bridge", anonymous=False, disable_signals=True)
        self.joints = A1JointStateCache(self.target_names)
        self.gripper_feedback = GripperFeedbackCache()
        self.relay = RelayMonitor(args.max_relay_status_age)
        self.staged = StagedCommandMonitor()

        rospy.Subscriber(args.joint_states_topic, JointState, self.joints.callback, queue_size=1)
        rospy.Subscriber(args.gripper_feedback_topic, JointState, self.gripper_feedback.callback, queue_size=1)
        rospy.Subscriber(args.relay_status_topic, String, self.relay.callback, queue_size=1)
        rospy.Subscriber(args.staged_command_topic, arm_control, self.staged.callback, queue_size=1)
        self.target_pub = rospy.Publisher(args.target_topic, JointState, queue_size=10)
        self.gripper_pub = rospy.Publisher(args.gripper_command_topic, gripper_position_control, queue_size=10)
        self.motion_enable_pub = rospy.Publisher(args.motion_enable_topic, Bool, queue_size=1, latch=True)

        self.policy = ActPolicyRunner(args)
        self.cam_front = RealSenseColorCamera(
            args.cam0_serial,
            args.cam_width,
            args.cam_height,
            args.cam_fps,
            auto_exposure=args.cam0_auto_exposure,
            exposure=args.cam0_exposure,
            gain=args.cam0_gain,
            auto_white_balance=args.cam0_auto_white_balance,
            white_balance=args.cam0_white_balance,
            warmup_frames=args.camera_warmup_frames,
        )
        self.cam_wrist = open_color_camera(
            args.cam1_backend,
            serial=args.cam1_serial,
            device=args.cam1_device,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.cam_fps,
            backend_api=args.cam1_backend_api,
            pixel_format=args.cam1_pixel_format,
            warmup_frames=args.camera_warmup_frames,
        )
        self.front_reader = LatestCameraReader("front", self.cam_front.read_frameset)
        self.wrist_reader = LatestCameraReader("wrist", self.cam_wrist.read_bgr)
        self.front_reader.start()
        self.wrist_reader.start()
        self.web_preview: CameraWebPreview | None = None
        preview_config = web_preview_config_from_args(args)
        if preview_config.enabled:
            self.web_preview = CameraWebPreview(preview_config)
            self.web_preview.register_reader(
                "agent",
                self.front_reader,
                extract=color_from_frameset,
                source=self.cam_front.label,
                overlay_roi=self.front_roi,
                overlay_label=(
                    f"POLICY INPUT {self.front_roi.width}x{self.front_roi.height}"
                ),
            )
            self.web_preview.register_reader(
                "wrist", self.wrist_reader, extract=color_from_bgr, source=self.cam_wrist.label
            )
            self.web_preview.start()

    def close(self) -> None:
        if self.motion_enabled:
            self.motion_enable_pub.publish(Bool(data=False))
            self.motion_enabled = False
        if getattr(self, "web_preview", None) is not None:
            self.web_preview.close()
        for reader in (getattr(self, "front_reader", None), getattr(self, "wrist_reader", None)):
            if reader is not None:
                reader.stop()
        for camera in (getattr(self, "cam_wrist", None), getattr(self, "cam_front", None)):
            if camera is not None:
                camera.close()

    def run(self) -> None:
        mode = "EXECUTE" if self.args.execute else "DRY-RUN"
        log(f"[ACT] Bridge started in {mode}. step_mode={self.args.step_mode}")
        model_calls = 0
        while not rospy.is_shutdown():
            if self.args.max_model_calls and model_calls >= self.args.max_model_calls:
                log("[ACT] max_model_calls reached; exiting.")
                return
            if self.args.step_mode and not self._wait_for_operator(model_calls + 1):
                return

            front_bgr, wrist_bgr, state7, current_joints = self._read_observation()
            chunk = self.policy.predict_chunk(
                front_bgr=front_bgr,
                wrist_bgr=wrist_bgr,
                state7=state7,
            )
            model_calls += 1
            self._print_preview(model_calls, chunk, current_joints)

            if not self.args.execute:
                if not self.args.step_mode:
                    time.sleep(1.0)
                continue

            try:
                steps = self._validated_execution_steps(chunk, current_joints)
            except RuntimeError as exc:
                self._skip_execution(str(exc))
                continue
            if not self.motion_enabled:
                self._wait_for_staged_alignment(current_joints)
                self._enable_motion()
            self._execute_steps(steps)

    def _wait_for_operator(self, call_index: int) -> bool:
        log(
            f"\n[ACT] INFERENCE #{call_index} READY. "
            "Press Enter to run one new model inference; q=quit."
        )
        try:
            value = input().strip().lower()
        except EOFError:
            value = "q"
        return value not in {"q", "quit", "exit"}

    def _read_observation(self) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], tuple[float, ...]]:
        current_joints = self._wait_for_joints()
        gripper = self._gripper_normalized()
        front_bgr, wrist_bgr = self._wait_for_cameras()
        state7 = (*current_joints, gripper)
        return front_bgr, wrist_bgr, state7, current_joints

    def _wait_for_joints(self) -> tuple[float, ...]:
        deadline = time.monotonic() + self.args.state_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            positions = self.joints.positions(max_age_s=self.args.max_feedback_age)
            if positions is not None and all(np.isfinite(positions)):
                return positions
            time.sleep(0.02)
        raise RuntimeError(f"No fresh usable joint feedback on {self.args.joint_states_topic}")

    def _gripper_normalized(self) -> float:
        feedback = self.gripper_feedback.normalized(
            max_age_s=self.args.max_feedback_age,
            stroke_min_mm=self.args.gripper_stroke_min,
            stroke_max_mm=self.args.gripper_stroke_max,
        )
        if feedback is not None:
            return feedback
        raise RuntimeError(f"No fresh gripper feedback on {self.args.gripper_feedback_topic}")

    def _wait_for_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + self.args.state_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            for reader in (self.front_reader, self.wrist_reader):
                exc = reader.exception()
                if exc is not None:
                    raise RuntimeError(f"{reader.name} camera reader failed") from exc
            front = self.front_reader.latest()
            wrist = self.wrist_reader.latest()
            now = time.perf_counter()
            if (
                front is not None
                and wrist is not None
                and now - front.monotonic_s <= self.args.max_camera_age
                and now - wrist.monotonic_s <= self.args.max_camera_age
            ):
                frameset = front.value
                if not isinstance(frameset, RealSenseFrameSet):
                    raise RuntimeError("front camera did not return a RealSenseFrameSet")
                return (
                    crop_image(frameset.color_bgr, self.front_roi, label="AgentView inference frame"),
                    wrist.value,
                )
            time.sleep(0.02)
        raise RuntimeError("No fresh camera pair within timeout")

    def _validated_execution_steps(self, chunk: np.ndarray, current_joints: tuple[float, ...]) -> np.ndarray:
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise RuntimeError(f"invalid chunk shape: {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError("ACT chunk contains non-finite values")
        count = min(int(self.args.execute_steps_per_inference), len(chunk))
        steps = chunk[:count].copy()
        current = np.asarray(current_joints, dtype=np.float64)
        if self.args.action_step_guard_enabled:
            first_delta = float(np.max(np.abs(steps[0, :6] - current)))
            if first_delta > self.args.max_first_target_delta_rad:
                raise RuntimeError(
                    f"first ACT target jumps {first_delta:.4f} rad from feedback; "
                    f"limit={self.args.max_first_target_delta_rad:.4f}"
                )
        previous = current
        for index, row in enumerate(steps):
            joints = row[:6]
            violations = self._joint_limit_violations(joints)
            if violations:
                detail = "; ".join(violations)
                raise RuntimeError(f"ACT target {index} violates joint limits: {detail}")
            if self.args.action_step_guard_enabled:
                step = float(np.max(np.abs(joints - previous)))
                limit = (
                    self.args.max_first_target_delta_rad
                    if index == 0
                    else self.args.max_joint_action_step_rad
                )
                if step > limit:
                    raise RuntimeError(
                        f"ACT target {index} step={step:.4f} rad exceeds limit={limit:.4f}"
                    )
            previous = joints
        return steps

    def _joint_limit_violations(self, joints: np.ndarray) -> list[str]:
        out: list[str] = []
        for name, value, lo, hi in zip(self.target_names, joints, self.lower_limits, self.upper_limits, strict=True):
            value = float(value)
            if value < float(lo) or value > float(hi):
                out.append(
                    f"{name}={value:.4f} outside [{float(lo):.4f}, {float(hi):.4f}] "
                    f"(target={np.round(joints, 4).tolist()})"
                )
        return out

    def _skip_execution(self, reason: str) -> None:
        self.motion_enable_pub.publish(Bool(data=False))
        self.motion_enabled = False
        log(f"[ACT safety] {reason}")
        log("[ACT safety] Skipping this action; relay is locked/disabled. Press Enter to infer again, q=quit.")

    def _print_preview(self, call_index: int, chunk: np.ndarray, current_joints: tuple[float, ...]) -> None:
        if not self.args.print_actions:
            return
        count = min(int(self.args.preview_steps), len(chunk))
        first_delta = float(np.max(np.abs(chunk[0, :6] - np.asarray(current_joints, dtype=np.float64))))
        adjacent = np.diff(chunk[: max(count, 2), :6], axis=0)
        max_step = float(np.max(np.abs(adjacent))) if adjacent.size else 0.0
        log(
            f"[ACT #{call_index}] first_delta={first_delta:.4f} max_preview_step={max_step:.4f} "
            f"gripper0={chunk[0, 6]:.3f}"
        )
        for idx in range(count):
            row = chunk[idx]
            log(
                f"  step {idx:02d}: joints={np.round(row[:6], 4).tolist()} "
                f"gripper_norm={row[6]:.3f}"
            )

    def _wait_for_staged_alignment(self, target: tuple[float, ...]) -> None:
        rate = rospy.Rate(min(float(self.args.control_hz), 30.0))
        deadline = time.monotonic() + self.args.state_timeout
        last_error: float | None = None
        log("[ACT] Aligning jointTracker staged output with current feedback before relay enable...")
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._publish_joint_target(target)
            last_error = self.staged.max_error(target, len(target))
            if last_error is not None and last_error <= self.args.initial_alignment_tolerance:
                log(f"[ACT] Tracker aligned; staged max error={last_error:.4f} rad")
                return
            rate.sleep()
        detail = "no staged command" if last_error is None else f"last max error {last_error:.4f} rad"
        raise RuntimeError(
            "jointTracker staged output did not align with current target "
            f"({detail}, tolerance={self.args.initial_alignment_tolerance:.4f})"
        )

    def _enable_motion(self) -> None:
        self.motion_enable_pub.publish(Bool(data=True))
        deadline = time.monotonic() + self.args.relay_enable_timeout
        last = self.relay.summary()
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            last = self.relay.summary()
            if self.relay.is_active():
                self.motion_enabled = True
                log("[ACT] Relay ACTIVE; joint targets may move the arm.")
                return
            status, _ = self.relay.status()
            if status is not None and status.state == "FAULT":
                break
            time.sleep(0.05)
        self.motion_enable_pub.publish(Bool(data=False))
        raise RuntimeError(f"A1 relay did not become ACTIVE: {last}")

    def _require_relay_active(self) -> None:
        if self.relay.is_active():
            return
        self.motion_enable_pub.publish(Bool(data=False))
        self.motion_enabled = False
        raise RuntimeError(f"A1 relay is not ACTIVE; refusing to publish. Last state: {self.relay.summary()}")

    def _execute_steps(self, steps: np.ndarray) -> None:
        rate = rospy.Rate(float(self.args.control_hz))
        for index, row in enumerate(steps):
            self._require_relay_active()
            stamp = self._publish_joint_target(tuple(float(v) for v in row[:6]))
            gripper_mm = self._publish_gripper(float(row[6]), stamp)
            if self.args.print_actions:
                log(
                    f"[ACT execute] step={index + 1}/{len(steps)} "
                    f"target={np.round(row[:6], 4).tolist()} "
                    f"gripper={row[6]:.3f} gripper_mm={gripper_mm:.1f}"
                )
            rate.sleep()

    def _publish_joint_target(self, target: Sequence[float]) -> Any:
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(self.target_names)
        msg.position = [float(value) for value in target]
        self.target_pub.publish(msg)
        return msg.header.stamp

    def _publish_gripper(self, value: float, stamp: Any) -> float:
        msg = gripper_position_control()
        msg.header.stamp = stamp
        stroke = denormalize_stroke(
            value,
            stroke_min_mm=self.args.gripper_stroke_min,
            stroke_max_mm=self.args.gripper_stroke_max,
        )
        msg.gripper_stroke = stroke
        self.gripper_pub.publish(msg)
        return stroke


def _bgr_to_chw_tensor(image: np.ndarray, *, width: int, height: int) -> torch.Tensor:
    if image.shape[:2] != (height, width):
        raise RuntimeError(f"camera image shape {image.shape[:2]} does not match expected {(height, width)}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)


def _policy_image_hw(config: Any, key: str) -> tuple[int, int]:
    features = getattr(config, "input_features", None)
    feature = features.get(key) if isinstance(features, dict) else None
    shape = tuple(int(value) for value in getattr(feature, "shape", ()))
    if len(shape) != 3 or shape[0] != 3 or shape[1] <= 0 or shape[2] <= 0:
        raise RuntimeError(
            f"ACT checkpoint is missing a valid CHW input feature for {key!r}: {shape!r}"
        )
    return shape[2], shape[1]


def _front_roi_from_args(args: argparse.Namespace) -> ImageRoi:
    if not args.cam0_crop_enabled:
        raise ValueError("--cam0-crop-enabled is required for the inference input contract")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-backbone-download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--step-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute-steps-per-inference", type=int, default=8)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--max-model-calls", type=int, default=0)
    parser.add_argument("--print-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-steps", type=int, default=5)
    parser.add_argument("--joint-states-topic", default="/joint_states_host")
    parser.add_argument("--target-topic", default="/arm_joint_target_position")
    parser.add_argument("--staged-command-topic", default="/arm_joint_command_a1_staged")
    parser.add_argument("--motion-enable-topic", default="/a1_arm_motion_enable")
    parser.add_argument("--relay-status-topic", default="/a1_arm_relay_status")
    parser.add_argument("--gripper-command-topic", default="/gripper_position_control_host")
    parser.add_argument("--gripper-feedback-topic", default="/gripper_stroke_host")
    parser.add_argument("--relay-enable-timeout", type=float, default=2.0)
    parser.add_argument("--max-relay-status-age", type=float, default=1.0)
    parser.add_argument("--target-joint-names", nargs=6, required=True)
    parser.add_argument("--lower-limits", nargs=6, type=float, required=True)
    parser.add_argument("--upper-limits", nargs=6, type=float, required=True)
    parser.add_argument(
        "--action-step-guard-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-joint-action-step-rad", type=float, default=0.25)
    parser.add_argument("--max-first-target-delta-rad", type=float, default=0.25)
    parser.add_argument("--initial-alignment-tolerance", type=float, default=0.05)
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--max-feedback-age", type=float, default=0.5)
    parser.add_argument("--max-camera-age", type=float, default=0.5)
    parser.add_argument("--gripper-stroke-min", type=float, default=0.0)
    parser.add_argument("--gripper-stroke-max", type=float, default=200.0)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=20)
    parser.add_argument("--cam0-serial", default="")
    parser.add_argument("--cam0-auto-exposure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cam0-exposure", type=int, default=140)
    parser.add_argument("--cam0-gain", type=int, default=32)
    parser.add_argument("--cam0-auto-white-balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cam0-white-balance", type=int, default=4600)
    parser.add_argument("--cam0-crop-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cam0-crop-x", type=int, default=0)
    parser.add_argument("--cam0-crop-y", type=int, default=0)
    parser.add_argument("--cam0-crop-width", type=int, default=640)
    parser.add_argument("--cam0-crop-height", type=int, default=480)
    parser.add_argument("--cam1-device", default="auto")
    parser.add_argument("--cam1-backend", choices=("realsense", "v4l2"), default="v4l2")
    parser.add_argument("--cam1-serial", default="")
    parser.add_argument("--cam1-backend-api", default="v4l2")
    parser.add_argument("--cam1-pixel-format", default="YUYV")
    add_web_preview_arguments(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "execute_steps_per_inference",
        "preview_steps",
        "cam_width",
        "cam_height",
        "cam_fps",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "control_hz",
        "relay_enable_timeout",
        "max_relay_status_age",
        "initial_alignment_tolerance",
        "state_timeout",
        "max_feedback_age",
        "max_camera_age",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.action_step_guard_enabled:
        for name in ("max_joint_action_step_rad", "max_first_target_delta_rad"):
            if float(getattr(args, name)) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive when the guard is enabled")
    if args.max_model_calls < 0:
        raise ValueError("--max-model-calls must be >= 0")
    if args.gripper_stroke_max <= args.gripper_stroke_min:
        raise ValueError("--gripper-stroke-max must be greater than --gripper-stroke-min")
    lower = np.asarray(args.lower_limits, dtype=np.float64)
    upper = np.asarray(args.upper_limits, dtype=np.float64)
    if np.any(lower >= upper):
        raise ValueError("--lower-limits must be below --upper-limits")
    _front_roi_from_args(args)


def main() -> int:
    args = parse_args()
    validate_args(args)
    bridge: ActJointBridge | None = None
    try:
        bridge = ActJointBridge(args)
        bridge.run()
        return 0
    finally:
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
