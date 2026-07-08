"""ROS1 hardware adapter for the Galaxea A1 safe runtime.

ROS imports are lazy so static tests can import this module without a ROS
installation. The adapter is scoped to the safe EEF command path and never
publishes directly to `/arm_joint_command_host`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from galaxea_a1_runtime.config import RuntimeConfig
from galaxea_a1_runtime.constants import EEF_FEEDBACK_TOPIC, JOINT_FEEDBACK_TOPIC
from galaxea_a1_runtime.hardware.eef import EefPose, action_to_eef_target
from galaxea_a1_runtime.hardware.io import A1Observation
from galaxea_a1_runtime.policies.actions import RuntimeAction
from galaxea_a1_runtime.schema import ActionMode, DEFAULT_STATE_NAMES


@dataclass(frozen=True)
class Ros1AdapterTopics:
    ee_target: str
    relay_enable: str
    relay_status: str
    gripper_command: str
    eef_feedback: str
    joint_feedback: str


class Ros1A1HardwareIO:
    """Safe-path ROS1 adapter for LeRobot-style A1 control.

    EEF actions are synthesized from live `/end_effector_pose` feedback and
    published to `/a1_ee_target`. This adapter never publishes host joint
    commands directly.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._connected = False
        self._rospy: Any | None = None
        self._ee_pub: Any | None = None
        self._motion_enable_pub: Any | None = None
        self._gripper_pub: Any | None = None
        self._eef_sub: Any | None = None
        self._pose_stamped_type: Any | None = None
        self._bool_type: Any | None = None
        self._gripper_type: Any | None = None
        self._latest_eef_pose: EefPose | None = None
        self._latest_observation = A1Observation(state=(0.0,) * len(DEFAULT_STATE_NAMES))

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def topics(self) -> Ros1AdapterTopics:
        return Ros1AdapterTopics(
            ee_target=self.config.topics.ee_target,
            relay_enable=self.config.topics.relay_enable,
            relay_status=self.config.topics.relay_status,
            gripper_command=self.config.topics.gripper_command,
            eef_feedback=EEF_FEEDBACK_TOPIC,
            joint_feedback=JOINT_FEEDBACK_TOPIC,
        )

    def connect(self) -> None:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Bool
        from signal_arm.msg import gripper_position_control

        self._rospy = rospy
        self._pose_stamped_type = PoseStamped
        self._bool_type = Bool
        self._gripper_type = gripper_position_control
        if not rospy.core.is_initialized():
            rospy.init_node("galaxea_a1_runtime_adapter", anonymous=True, disable_signals=True)
        self._ee_pub = rospy.Publisher(self.config.topics.ee_target, PoseStamped, queue_size=1)
        self._motion_enable_pub = rospy.Publisher(
            self.config.topics.relay_enable,
            Bool,
            queue_size=1,
        )
        self._gripper_pub = rospy.Publisher(
            self.config.topics.gripper_command,
            gripper_position_control,
            queue_size=1,
        )
        self._eef_sub = rospy.Subscriber(
            EEF_FEEDBACK_TOPIC,
            PoseStamped,
            self._eef_pose_cb,
            queue_size=1,
        )
        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            self._publish_motion_enable(False)
        if self._eef_sub is not None:
            unregister = getattr(self._eef_sub, "unregister", None)
            if callable(unregister):
                unregister()
            self._eef_sub = None
        self._connected = False

    def get_observation(self) -> A1Observation:
        if not self._connected:
            raise RuntimeError("ROS1 A1 adapter is not connected")
        return self._latest_observation

    def send_runtime_action(self, action: RuntimeAction) -> RuntimeAction:
        if not self._connected:
            raise RuntimeError("ROS1 A1 adapter is not connected")
        if self._rospy is None:
            raise RuntimeError("rospy is not initialized")
        target = self._target_from_action(action)
        if target is not None:
            self._publish_eef_target(target)
            self._publish_motion_enable(True)
            self._publish_eef_target(target)
        self._publish_gripper_if_present(action)
        return action

    def _target_from_action(self, action: RuntimeAction) -> EefPose | None:
        if action.mode == ActionMode.JOINT_ABSOLUTE:
            raise ValueError("ROS1 safe adapter only supports EEF action modes")
        if self._latest_eef_pose is None:
            target = action_to_eef_target(EefPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), action)
            if target is None:
                return None
            raise RuntimeError(
                f"No {EEF_FEEDBACK_TOPIC} feedback has been received; cannot synthesize a safe "
                "/a1_ee_target for arm motion"
            )
        return action_to_eef_target(self._latest_eef_pose, action)

    def _publish_eef_target(self, target: EefPose) -> None:
        if self._ee_pub is None:
            raise RuntimeError("ROS1 A1 adapter EE publisher is not initialized")
        msg = self._pose_to_msg(target)
        self._ee_pub.publish(msg)

    def _publish_motion_enable(self, enabled: bool) -> None:
        if self._motion_enable_pub is None or self._bool_type is None:
            return
        msg = self._bool_type(bool(enabled))
        self._motion_enable_pub.publish(msg)

    def _publish_gripper_if_present(self, action: RuntimeAction) -> None:
        if "gripper" not in action.names or self._gripper_pub is None or self._rospy is None:
            return
        if self._gripper_type is None:
            raise RuntimeError("ROS1 A1 adapter gripper message type is not initialized")
        msg = self._gripper_type()
        msg.header.stamp = self._rospy.Time.now()
        msg.gripper_stroke = _normalized_gripper_to_stroke_mm(action.as_dict()["gripper"])
        self._gripper_pub.publish(msg)

    def _pose_to_msg(self, target: EefPose) -> Any:
        if self._pose_stamped_type is None or self._rospy is None:
            raise RuntimeError("ROS1 A1 adapter pose message type is not initialized")
        pose = target.normalized()
        msg = self._pose_stamped_type()
        msg.header.stamp = self._rospy.Time.now()
        msg.header.frame_id = pose.frame_id
        msg.pose.position.x = pose.xyz[0]
        msg.pose.position.y = pose.xyz[1]
        msg.pose.position.z = pose.xyz[2]
        msg.pose.orientation.x = pose.quat_xyzw[0]
        msg.pose.orientation.y = pose.quat_xyzw[1]
        msg.pose.orientation.z = pose.quat_xyzw[2]
        msg.pose.orientation.w = pose.quat_xyzw[3]
        return msg

    def _eef_pose_cb(self, msg: Any) -> None:
        pose = EefPose(
            xyz=(
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ),
            quat_xyzw=(
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
                float(msg.pose.orientation.w),
            ),
            frame_id=getattr(msg.header, "frame_id", "") or "world",
        ).normalized()
        self._latest_eef_pose = pose
        previous = self._latest_observation.state
        state = pose.xyz + pose.quat_xyzw + previous[7:]
        self._latest_observation = A1Observation(
            state=state,
            images=self._latest_observation.images,
            timestamp=_stamp_to_seconds(getattr(msg.header, "stamp", None)),
        )


def build_ros1_safe_adapter(config: RuntimeConfig) -> Ros1A1HardwareIO:
    if not config.touches_hardware:
        raise ValueError("ROS1 adapter requires a hardware-touching runtime profile")
    return Ros1A1HardwareIO(config=config)


def _normalized_gripper_to_stroke_mm(value: float) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"gripper action must be normalized to [0, 1], got {value!r}")
    return normalized * 60.0


def _stamp_to_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    to_sec = getattr(stamp, "to_sec", None)
    if callable(to_sec):
        return float(to_sec())
    secs = getattr(stamp, "secs", None)
    nsecs = getattr(stamp, "nsecs", None)
    if secs is not None and nsecs is not None:
        return float(secs) + float(nsecs) * 1e-9
    return None
