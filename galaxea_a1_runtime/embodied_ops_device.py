"""Operational device hosted by the supervised Galaxea A1 RPC service."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from embodied_ops import (
    Capability,
    DeviceManifest,
    FeatureSpec,
    HealthReport,
    HealthStatus,
    LifecycleError,
    validate_feature_values,
)

from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.configuration.system import SystemConfig
from galaxea_a1_runtime.gripper import denormalize_stroke
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES_RAD

JOINT_FEATURE_KEYS = JOINT_ACTION_NAMES_RAD[:-1]
GRIPPER_FEATURE_KEY = JOINT_ACTION_NAMES_RAD[-1]


class _Session(Protocol):
    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def observe(self) -> Mapping[str, object]: ...

    def command(self, action: Mapping[str, object]) -> Mapping[str, object]: ...

    def health(self) -> HealthReport: ...

    def disconnect(self) -> None: ...


SessionFactory = Callable[[SystemConfig, float], _Session]


class A1RuntimeDevice:
    """Strict device whose construction performs no ROS or hardware access."""

    def __init__(
        self,
        *,
        system: SystemConfig,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.system = system
        self.device_connect_timeout_s = system.embodied_ops.device_connect_timeout_s
        self._session_factory = session_factory or _RosA1Session
        self._session: _Session | None = None
        self._manifest = _manifest_from_system(self.system)

    @property
    def manifest(self) -> DeviceManifest:
        return self._manifest

    @property
    def is_connected(self) -> bool:
        return self._session is not None and self._session.is_connected

    def connect(self) -> None:
        if self.is_connected:
            raise LifecycleError("Galaxea A1 runtime device is already connected")
        session = self._session_factory(self.system, self.device_connect_timeout_s)
        try:
            session.connect()
        except BaseException:
            session.disconnect()
            raise
        self._session = session

    def observe(self) -> Mapping[str, object]:
        session = self._require_session()
        return validate_feature_values(
            session.observe(),
            self.manifest.observation_features,
        )

    def command(self, action: Mapping[str, object]) -> Mapping[str, object]:
        session = self._require_session()
        requested = validate_feature_values(action, self.manifest.action_features)
        sent = session.command(requested)
        return validate_feature_values(sent, self.manifest.action_features)

    def health(self) -> HealthReport:
        if not self.is_connected:
            return HealthReport(
                HealthStatus.UNKNOWN, "A1 runtime backend is disconnected"
            )
        return self._require_session().health()

    def disconnect(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.disconnect()

    def _require_session(self) -> _Session:
        if not self.is_connected or self._session is None:
            raise LifecycleError("Galaxea A1 runtime device is not connected")
        return self._session


def _manifest_from_system(system: SystemConfig) -> DeviceManifest:
    observations = tuple(
        FeatureSpec(name, unit="rad") for name in JOINT_FEATURE_KEYS
    ) + (FeatureSpec(GRIPPER_FEATURE_KEY, minimum=0.0, maximum=1.0),)
    actions = tuple(
        FeatureSpec(name, unit="rad", minimum=lower, maximum=upper)
        for name, lower, upper in zip(
            JOINT_FEATURE_KEYS,
            system.joint_safety.lower_limits,
            system.joint_safety.upper_limits,
            strict=True,
        )
    ) + (FeatureSpec(GRIPPER_FEATURE_KEY, minimum=0.0, maximum=1.0),)
    return DeviceManifest(
        identifier="galaxea-a1",
        capabilities=(Capability.OBSERVE, Capability.COMMAND, Capability.HEALTH),
        observation_features=observations,
        action_features=actions,
        metadata={
            "robot_type": "galaxea_a1",
            "control_path": "joint_target->staged_tracker->locked_relay->host_driver",
        },
    )


class _RosA1Session:
    """Lazy ROS adapter; the supervised runtime must already own driver and relay."""

    def __init__(self, system: SystemConfig, connect_timeout_s: float) -> None:
        self.system = system
        self.connect_timeout_s = connect_timeout_s
        self._connected = False
        self._resources_open = False
        self._subscribers: list[Any] = []
        self._publishers: list[Any] = []
        self._timer: Any | None = None
        self._gate: Any | None = None
        self._commander: _JointCommandPublisher | None = None
        self._rospy: Any | None = None
        self._joints: Any | None = None
        self._gripper: Any | None = None
        self._relay: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self._resources_open:
            raise LifecycleError("ROS A1 session is already open")

        repo_root = discover_repo_root(self.system.path)
        from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

        configure_ros1_python(repo_root)

        import rospy
        from sensor_msgs.msg import JointState
        from signal_arm.msg import arm_control, gripper_position_control
        from std_msgs.msg import Bool, String

        from galaxea_a1_runtime.runtime.relay import (
            RelayMonitor,
            relay_status_is_fresh,
        )
        from galaxea_a1_runtime.runtime.ros_feedback import (
            A1JointStateCache,
            GripperFeedbackCache,
            StagedCommandMonitor,
        )
        from galaxea_a1_runtime.runtime.staged_motion import StagedMotionGate

        if not rospy.core.is_initialized():
            rospy.init_node(
                "embodied_ops_galaxea_a1",
                anonymous=True,
                disable_signals=True,
            )

        topics = self.system.topics
        self._rospy = rospy
        self._joints = A1JointStateCache(self.system.joint_safety.names)
        self._gripper = GripperFeedbackCache()
        self._relay = RelayMonitor(self.system.relay.max_status_age_s)
        staged = StagedCommandMonitor()
        self._subscribers = [
            rospy.Subscriber(
                topics.joint_states, JointState, self._joints.callback, queue_size=10
            ),
            rospy.Subscriber(
                topics.gripper_feedback,
                JointState,
                self._gripper.callback,
                queue_size=10,
            ),
            rospy.Subscriber(
                topics.relay_status, String, self._relay.callback, queue_size=10
            ),
            rospy.Subscriber(
                topics.staged_command, arm_control, staged.callback, queue_size=10
            ),
        ]
        target_pub = rospy.Publisher(topics.joint_target, JointState, queue_size=10)
        motion_pub = rospy.Publisher(
            topics.motion_enable, Bool, queue_size=1, latch=True
        )
        gripper_pub = rospy.Publisher(
            topics.gripper_target,
            gripper_position_control,
            queue_size=10,
        )
        self._publishers = [target_pub, motion_pub, gripper_pub]
        self._commander = _JointCommandPublisher(
            rospy=rospy,
            joint_state_type=JointState,
            bool_type=Bool,
            gripper_type=gripper_position_control,
            target_pub=target_pub,
            motion_pub=motion_pub,
            gripper_pub=gripper_pub,
            system=self.system,
            current_joints=self._current_joints,
        )
        self._gate = StagedMotionGate(
            relay=self._relay,
            commander=self._commander,
            staged_monitor=staged,
            relay_enable_timeout_s=self.system.relay.enable_timeout_s,
            staged_wait_timeout_s=self.connect_timeout_s,
            staged_max_age_s=self.system.relay.max_input_age_s,
            staged_alignment_tolerance_rad=(
                self.system.joint_safety.initial_alignment_tolerance_rad
            ),
            is_shutdown=rospy.is_shutdown,
            owner_label="LeRobot Galaxea A1",
        )
        self._timer = rospy.Timer(
            rospy.Duration(0.05), self._gate.publish_active_target
        )
        self._resources_open = True

        try:
            self._wait_ready(relay_status_is_fresh)
        except BaseException:
            self.disconnect()
            raise
        self._connected = True

    def observe(self) -> Mapping[str, object]:
        joints = self._current_joints()
        gripper = self._current_gripper()
        if joints is None or gripper is None:
            raise LifecycleError(
                "A1 observation is missing fresh joint or gripper feedback"
            )
        return {
            **dict(zip(JOINT_FEATURE_KEYS, joints, strict=True)),
            GRIPPER_FEATURE_KEY: gripper,
        }

    def command(self, action: Mapping[str, object]) -> Mapping[str, object]:
        if not self._connected or self._gate is None or self._commander is None:
            raise LifecycleError("ROS A1 session is not connected")
        try:
            if not self._gate.motion_enabled:
                self._gate.activate_current_hold()
            else:
                self._gate.enable_motion()
            joints = tuple(float(action[name]) for name in JOINT_FEATURE_KEYS)
            gripper = float(action[GRIPPER_FEATURE_KEY])
            self._commander.publish_joint_target(joints)
            self._commander.publish_gripper(gripper)
        except BaseException:
            if self._gate.motion_requested:
                self._gate.disable_motion()
            raise
        return dict(action)

    def health(self) -> HealthReport:
        if not self._connected or self._relay is None:
            return HealthReport(HealthStatus.UNKNOWN, "ROS A1 session is disconnected")
        joints_ready = self._current_joints() is not None
        gripper_ready = self._current_gripper() is not None
        status, _ = self._relay.status()
        if status is not None and status.state == "FAULT":
            return HealthReport(HealthStatus.FAULT, self._relay.summary())
        if not joints_ready or not gripper_ready:
            return HealthReport(
                HealthStatus.DEGRADED,
                "A1 feedback is stale",
                {"joints_fresh": joints_ready, "gripper_fresh": gripper_ready},
            )
        return HealthReport(HealthStatus.HEALTHY, self._relay.summary())

    def disconnect(self) -> None:
        self._connected = False
        errors: list[BaseException] = []
        if self._gate is not None and self._gate.motion_requested:
            try:
                self._gate.disable_motion()
                time.sleep(0.1)
            except BaseException as exc:
                errors.append(exc)
        if self._timer is not None:
            try:
                self._timer.shutdown()
            except BaseException as exc:
                errors.append(exc)
        for resource in [*self._subscribers, *self._publishers]:
            try:
                resource.unregister()
            except BaseException as exc:
                errors.append(exc)
        self._subscribers = []
        self._publishers = []
        self._timer = None
        self._gate = None
        self._commander = None
        self._resources_open = False
        if errors:
            raise BaseExceptionGroup("ROS A1 session cleanup failed", errors)

    def _wait_ready(self, relay_status_is_fresh: Callable[..., bool]) -> None:
        assert self._rospy is not None and self._relay is not None
        deadline = time.monotonic() + self.connect_timeout_s
        last_summary = "no relay status"
        while not self._rospy.is_shutdown() and time.monotonic() < deadline:
            joints = self._current_joints()
            gripper = self._current_gripper()
            status, updated = self._relay.status()
            last_summary = self._relay.summary()
            if status is not None and relay_status_is_fresh(
                updated,
                max_age_s=self.system.relay.max_status_age_s,
            ):
                if status.state == "FAULT":
                    raise LifecycleError(f"A1 relay is faulted: {last_summary}")
                if status.state in {"ACTIVE", "ARMING"}:
                    raise LifecycleError(
                        "A1 relay already has an active/arming owner; refuse competing control"
                    )
                if (
                    status.state == "LOCKED"
                    and joints is not None
                    and gripper is not None
                ):
                    return
            time.sleep(0.05)
        raise LifecycleError(
            f"A1 runtime did not provide fresh locked feedback within "
            f"{self.connect_timeout_s:.1f}s: {last_summary}"
        )

    def _current_joints(self) -> tuple[float, ...] | None:
        if self._joints is None:
            return None
        return self._joints.positions(
            max_age_s=self.system.joint_safety.max_feedback_age_s
        )

    def _current_gripper(self) -> float | None:
        if self._gripper is None:
            return None
        return self._gripper.normalized(
            max_age_s=self.system.joint_safety.max_feedback_age_s,
            stroke_min_mm=self.system.gripper.stroke_min_mm,
            stroke_max_mm=self.system.gripper.stroke_max_mm,
        )


class _JointCommandPublisher:
    """Named joint and gripper target publisher for the staged runtime path."""

    def __init__(
        self,
        *,
        rospy: Any,
        joint_state_type: Any,
        bool_type: Any,
        gripper_type: Any,
        target_pub: Any,
        motion_pub: Any,
        gripper_pub: Any,
        system: SystemConfig,
        current_joints: Callable[[], Sequence[float] | None],
    ) -> None:
        self.rospy = rospy
        self.joint_state_type = joint_state_type
        self.bool_type = bool_type
        self.gripper_type = gripper_type
        self.target_pub = target_pub
        self.motion_pub = motion_pub
        self.gripper_pub = gripper_pub
        self.system = system
        self.current_joints = current_joints
        self._active_target: tuple[float, ...] | None = None
        self._lock = threading.Lock()

    def publish_motion_enable(self, enabled: bool) -> None:
        self.motion_pub.publish(self.bool_type(data=bool(enabled)))

    def hold_current_target(self) -> tuple[float, ...]:
        current = self.current_joints()
        if current is None:
            raise LifecycleError("cannot stage an A1 hold without fresh joint feedback")
        return self._set_active_target(current)

    def publish_joint_target(self, target: Sequence[float]) -> None:
        self._set_active_target(target)
        self.publish_active_target()

    def publish_active_target(self) -> None:
        with self._lock:
            target = self._active_target
        if target is None:
            return
        message = self.joint_state_type()
        message.header.stamp = self.rospy.Time.now()
        message.name = list(self.system.joint_safety.names)
        message.position = list(target)
        self.target_pub.publish(message)

    def publish_gripper(self, normalized: float) -> None:
        stroke = denormalize_stroke(
            normalized,
            stroke_min_mm=self.system.gripper.stroke_min_mm,
            stroke_max_mm=self.system.gripper.stroke_max_mm,
        )
        message = self.gripper_type()
        message.header.stamp = self.rospy.Time.now()
        message.gripper_stroke = float(stroke)
        self.gripper_pub.publish(message)

    def _set_active_target(self, target: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in target)
        expected = len(self.system.joint_safety.names)
        if len(values) != expected or not all(math.isfinite(value) for value in values):
            raise ValueError(f"A1 joint target must contain {expected} finite values")
        for name, value, lower, upper in zip(
            self.system.joint_safety.names,
            values,
            self.system.joint_safety.lower_limits,
            self.system.joint_safety.upper_limits,
            strict=True,
        ):
            if not lower <= value <= upper:
                raise ValueError(
                    f"A1 joint target {name}={value:g} is outside [{lower:g}, {upper:g}]"
                )
        with self._lock:
            self._active_target = values
        return values
