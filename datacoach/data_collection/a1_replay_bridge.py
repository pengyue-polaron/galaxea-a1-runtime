import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    import zmq
except Exception as exc:
    raise RuntimeError(
        "Failed to import pyzmq. Install dependency in DataCoach environment, e.g. `pip install pyzmq`."
    ) from exc


def _extend_ros_python_paths():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
        str(repo_root / "third_party" / "A1_SDK" / "install" / "lib" / "python3" / "dist-packages"),
    ]
    a1_sdk_root = os.environ.get("A1_SDK_ROOT")
    if a1_sdk_root:
        candidates.append(str(Path(a1_sdk_root) / "install" / "lib" / "python3" / "dist-packages"))
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


_extend_ros_python_paths()

try:
    import rospy
    import rosgraph
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from signal_arm.msg import gripper_joint_command, gripper_position_control
except Exception as exc:
    raise RuntimeError(
        "Failed to import ROS/A1 message modules. "
        "Please source ROS + A1_SDK setup first, e.g.:\n"
        "source /opt/ros/noetic/setup.bash\n"
        "source third_party/A1_SDK/install/setup.bash\n"
        f"Original error: {exc}"
    ) from exc

from datacoach.constants import ROBOT_FPS, ZMQ_CMD_PORT, ZMQ_STATE_PORT


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _pose_to_dict(msg: PoseStamped) -> dict[str, tuple[float, ...]]:
    return {
        "pos": (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ),
        "ori": (
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        ),
    }


class A1ReplayBridge:
    """Bridge ROS replay topics to DataCoach ZMQ state/cmd streams."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.lock = threading.Lock()

        self.publish_rate_hz = float(_cfg_get(cfg, "publish_rate_hz", ROBOT_FPS))
        self.fallback_gripper = float(_cfg_get(cfg, "fallback_gripper", 0.0))
        self.require_command_pose = bool(_cfg_get(cfg, "require_command_pose", True))
        self.publish_debug = bool(_cfg_get(cfg, "publish_debug", False))

        self.state_pose_topic = str(_cfg_get(cfg, "state_pose_topic", "/end_effector_pose"))
        self.state_gripper_topic = str(_cfg_get(cfg, "state_gripper_topic", "/gripper_stroke_host"))
        self.cmd_pose_topic = str(_cfg_get(cfg, "cmd_pose_topic", "/a1_ee_target"))
        self.cmd_gripper_position_topic = str(
            _cfg_get(cfg, "cmd_gripper_position_topic", "/gripper_position_control_host")
        )
        self.cmd_gripper_force_topic = str(
            _cfg_get(cfg, "cmd_gripper_force_topic", "/gripper_force_control_host")
        )

        state_bind = str(_cfg_get(cfg, "zmq_state_bind", f"tcp://127.0.0.1:{ZMQ_STATE_PORT}"))
        cmd_bind = str(_cfg_get(cfg, "zmq_cmd_bind", f"tcp://127.0.0.1:{ZMQ_CMD_PORT}"))

        self.context = zmq.Context()
        self.zmq_state_pub = self.context.socket(zmq.PUB)
        self.zmq_cmd_pub = self.context.socket(zmq.PUB)
        self.zmq_state_pub.bind(state_bind)
        self.zmq_cmd_pub.bind(cmd_bind)

        self.latest_state_pos: Optional[tuple[float, float, float]] = None
        self.latest_state_ori: Optional[tuple[float, float, float, float]] = None
        self.latest_state_gripper: Optional[float] = None

        self.latest_cmd_pos: Optional[tuple[float, float, float]] = None
        self.latest_cmd_ori: Optional[tuple[float, float, float, float]] = None
        self.latest_cmd_gripper: Optional[float] = None
        self.latest_cmd_gripper_source = "fallback"

        rospy.Subscriber(self.state_pose_topic, PoseStamped, self._state_pose_cb, queue_size=100)
        rospy.Subscriber(self.state_gripper_topic, JointState, self._state_gripper_cb, queue_size=100)
        rospy.Subscriber(self.cmd_pose_topic, PoseStamped, self._cmd_pose_cb, queue_size=100)
        rospy.Subscriber(
            self.cmd_gripper_position_topic,
            gripper_position_control,
            self._cmd_gripper_position_cb,
            queue_size=100,
        )
        rospy.Subscriber(
            self.cmd_gripper_force_topic,
            gripper_joint_command,
            self._cmd_gripper_force_cb,
            queue_size=100,
        )

        print(f"[ReplayBridge] State pose topic: {self.state_pose_topic}")
        print(f"[ReplayBridge] State gripper topic: {self.state_gripper_topic}")
        print(f"[ReplayBridge] Command pose topic: {self.cmd_pose_topic}")
        print(f"[ReplayBridge] Command gripper position topic: {self.cmd_gripper_position_topic}")
        print(f"[ReplayBridge] Command gripper force topic: {self.cmd_gripper_force_topic}")
        print(f"[ReplayBridge] ZMQ state PUB: {state_bind}")
        print(f"[ReplayBridge] ZMQ cmd PUB: {cmd_bind}")

    def _state_pose_cb(self, msg: PoseStamped):
        pose = _pose_to_dict(msg)
        with self.lock:
            self.latest_state_pos = pose["pos"]
            self.latest_state_ori = pose["ori"]

    def _state_gripper_cb(self, msg: JointState):
        if not msg.position:
            return
        with self.lock:
            self.latest_state_gripper = float(msg.position[0])

    def _cmd_pose_cb(self, msg: PoseStamped):
        pose = _pose_to_dict(msg)
        with self.lock:
            self.latest_cmd_pos = pose["pos"]
            self.latest_cmd_ori = pose["ori"]

    def _cmd_gripper_position_cb(self, msg: gripper_position_control):
        with self.lock:
            self.latest_cmd_gripper = float(msg.gripper_stroke)
            self.latest_cmd_gripper_source = "position"

    def _cmd_gripper_force_cb(self, msg: gripper_joint_command):
        # Keep the latest force value as a fallback scalar channel.
        with self.lock:
            self.latest_cmd_gripper = float(msg.gripper_force)
            self.latest_cmd_gripper_source = "force"

    def _effective_gripper(self, cmd_value: Optional[float], state_value: Optional[float]) -> float:
        if cmd_value is not None:
            return float(cmd_value)
        if state_value is not None:
            return float(state_value)
        return self.fallback_gripper

    def _publish_once(self):
        now = time.time()
        with self.lock:
            state_pos = self.latest_state_pos
            state_ori = self.latest_state_ori
            state_gripper = self.latest_state_gripper

            cmd_pos = self.latest_cmd_pos
            cmd_ori = self.latest_cmd_ori
            cmd_gripper = self.latest_cmd_gripper
            cmd_gripper_source = self.latest_cmd_gripper_source

        if state_pos is not None and state_ori is not None:
            state_payload = {
                "timestamp": now,
                "pos": state_pos,
                "ori": state_ori,
                "gripper": self._effective_gripper(None, state_gripper),
            }
            self.zmq_state_pub.send_json(state_payload)

        should_publish_cmd = cmd_pos is not None and cmd_ori is not None
        if not self.require_command_pose:
            should_publish_cmd = should_publish_cmd or (cmd_gripper is not None)

        if should_publish_cmd:
            cmd_payload = {
                "timestamp": now,
                "pos": cmd_pos if cmd_pos is not None else state_pos,
                "ori": cmd_ori if cmd_ori is not None else state_ori,
                "gripper": self._effective_gripper(cmd_gripper, state_gripper),
                "gripper_source": cmd_gripper_source,
            }
            self.zmq_cmd_pub.send_json(cmd_payload)

            if self.publish_debug:
                print(f"[ReplayBridge] cmd: {cmd_payload}")

    def run(self, stop_event=None):
        rate = rospy.Rate(max(self.publish_rate_hz, 1.0))
        while not rospy.is_shutdown():
            if stop_event is not None and stop_event.is_set():
                break
            self._publish_once()
            rate.sleep()

    def close(self):
        self.zmq_state_pub.close(0)
        self.zmq_cmd_pub.close(0)
        self.context.term()


def main(cfg=None, stop_event=None):
    node_name = str(_cfg_get(cfg, "node_name", "a1_replay_bridge"))
    anonymous = bool(_cfg_get(cfg, "anonymous", False))
    disable_ros_signals = bool(_cfg_get(cfg, "disable_ros_signals", False))

    if not rosgraph.is_master_online():
        raise RuntimeError("ROS master is not online. Start roscore/launch files first.")

    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=anonymous, disable_signals=disable_ros_signals)

    bridge = A1ReplayBridge(cfg)
    try:
        bridge.run(stop_event=stop_event)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
