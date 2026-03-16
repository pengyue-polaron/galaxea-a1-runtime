from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Header

from collections import deque
import socket
import threading
import time

import rospy
import zmq

from datacoach.constants import (
    LEROBOT_PORT,
    OFFSET,
    ROBOT_FPS,
    SCALE,
    ZMQ_CMD_PORT,
    ZMQ_POLICY_ACTION_PORT,
    ZMQ_STATE_PORT,
)


class A1Server:
    """Bridge leader UDP / policy actions and A1 ROS topics."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.leader_udp_host = str(self._cfg_get("leader_udp_host", "127.0.0.1"))
        self.leader_udp_port = int(self._cfg_get("leader_udp_port", LEROBOT_PORT))
        self.scale = [float(v) for v in self._cfg_get("scale", SCALE)]
        self.offset = [float(v) for v in self._cfg_get("offset", OFFSET)]
        self.publish_rate_hz = float(self._cfg_get("publish_rate_hz", ROBOT_FPS))
        # Drop stale ROS feedback instead of replaying frozen state forever.
        self.state_stale_timeout_s = float(self._cfg_get("state_stale_timeout_s", 0.3))

        self.state_joint_topic = str(self._cfg_get("state_joint_topic", "/joint_states_host"))
        self.cmd_joint_topic = str(self._cfg_get("cmd_joint_topic", "/arm_joint_target_position"))
        # Kept for the leader/drag path (ros_publisher).
        self.cmd_pose_topic = str(self._cfg_get("cmd_pose_topic", "/a1_ee_target"))
        self.cmd_gripper_position_topic = str(
            self._cfg_get("cmd_gripper_position_topic", "/gripper_position_control_host")
        )

        self.zmq_cmd_bind = str(self._cfg_get("zmq_cmd_bind", f"tcp://127.0.0.1:{ZMQ_CMD_PORT}"))
        self.zmq_state_bind = str(self._cfg_get("zmq_state_bind", f"tcp://127.0.0.1:{ZMQ_STATE_PORT}"))
        self.zmq_policy_action_connect = str(
            self._cfg_get("zmq_policy_action_connect", f"tcp://127.0.0.1:{ZMQ_POLICY_ACTION_PORT}")
        )
        self.feedback_lock = threading.Lock()
        # Keep only the freshest leader command to prevent stale command replay.
        self._leader_queue = deque(maxlen=1)
        self._leader_lock = threading.Lock()

    def _cfg_get(self, key, default=None):
        if self.cfg is None:
            return default
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        if hasattr(self.cfg, "get"):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def leader_data_receiver(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.leader_udp_host, self.leader_udp_port))
        print(
            "[A1Server] Listening for leader data on "
            f"UDP {self.leader_udp_host}:{self.leader_udp_port} ..."
        )

        while not rospy.is_shutdown():
            data, _ = sock.recvfrom(4096)
            parts = data.decode().split(",")

            try:
                x, y, z = map(float, parts[:3])
            except Exception:
                continue

            tx = x * self.scale[0] + self.offset[0]
            ty = y * self.scale[1] + self.offset[1]
            tz = z * self.scale[2] + self.offset[2]

            pose_data = {
                "pos": (tx, ty, tz),
                "ori": tuple(map(float, parts[3:7])) if len(parts) >= 7 else None,
                "gripper": float(parts[7]) if len(parts) >= 8 else None,
            }

            with self._leader_lock:
                self._leader_queue.append(pose_data)
            time.sleep(1 / max(self.publish_rate_hz, 1.0))

    def ros_publisher(self):
        """
        Publish processed leader data to A1 ROS topics and ZMQ commanded-state.
        """
        pub = rospy.Publisher(self.cmd_pose_topic, PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher(self.cmd_gripper_position_topic, gripper_position_control, queue_size=10)
        rate = rospy.Rate(max(self.publish_rate_hz, 1.0))

        context = zmq.Context()
        zmq_pub_cmd = context.socket(zmq.PUB)
        zmq_pub_cmd.bind(self.zmq_cmd_bind)
        print(f"[A1Server] ZMQ publisher (commanded_state) bound to {self.zmq_cmd_bind}")

        while not rospy.is_shutdown():
            pose_data = None
            with self._leader_lock:
                if self._leader_queue:
                    pose_data = self._leader_queue.popleft()

            if pose_data is not None:

                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = pose_data[
                    "pos"
                ]

                if pose_data["ori"]:
                    (
                        pose_msg.pose.orientation.x,
                        pose_msg.pose.orientation.y,
                        pose_msg.pose.orientation.z,
                        pose_msg.pose.orientation.w,
                    ) = pose_data["ori"]
                pub.publish(pose_msg)

                if pose_data["gripper"] is not None:
                    grip_msg = gripper_position_control()
                    grip_msg.header = Header()
                    grip_msg.header.stamp = rospy.Time.now()
                    grip_msg.gripper_stroke = pose_data["gripper"]
                    gripper_pub.publish(grip_msg)

                zmq_pub_cmd.send_json(
                    {
                        "timestamp": time.time(),
                        "pos": pose_data["pos"],
                        "ori": pose_data["ori"],
                        "gripper": pose_data["gripper"],
                    }
                )
            rate.sleep()

    def ros_subscriber(self):
        """
        Subscribe joint state feedback and publish to ZMQ state stream.
        State vector: 7D = [joint1..joint6, gripper_joint]
        """
        context = zmq.Context()
        zmq_pub_state = context.socket(zmq.PUB)
        zmq_pub_state.bind(self.zmq_state_bind)
        print(f"[A1Server] ZMQ PUB bound to {self.zmq_state_bind}")
        time.sleep(0.2)

        self.feedback_lock = threading.Lock()
        self.joint_data = None
        self.joint_data_ts = None

        def _stamp_to_sec(stamp):
            try:
                ts = float(stamp.to_sec())
            except Exception:
                return time.time()
            if ts <= 0.0:
                return time.time()
            return ts

        _JOINT_NAMES = [
            "arm_joint1", "arm_joint2", "arm_joint3",
            "arm_joint4", "arm_joint5", "arm_joint6", "gripper",
        ]

        def joint_callback(msg):
            if not msg.position or not msg.name:
                return
            name_to_pos = dict(zip(msg.name, msg.position))
            if not all(n in name_to_pos for n in _JOINT_NAMES):
                return
            with self.feedback_lock:
                self.joint_data = tuple(float(name_to_pos[n]) for n in _JOINT_NAMES)
                self.joint_data_ts = _stamp_to_sec(msg.header.stamp)

        rospy.Subscriber(self.state_joint_topic, JointState, joint_callback)

        rate = rospy.Rate(max(self.publish_rate_hz, 1.0))
        stale_state_drop_count = 0
        while not rospy.is_shutdown():
            now_s = time.time()
            with self.feedback_lock:
                joints = self.joint_data
                joint_ts = self.joint_data_ts

            if joints is not None and joint_ts is not None:
                joint_age = now_s - float(joint_ts)
                if joint_age > self.state_stale_timeout_s:
                    stale_state_drop_count += 1
                    if stale_state_drop_count % 100 == 1:
                        print(
                            "[A1Server] Dropping stale ROS feedback "
                            f"(joint_age={joint_age:.3f}s, "
                            f"timeout={self.state_stale_timeout_s:.3f}s, count={stale_state_drop_count})"
                        )
                    rate.sleep()
                    continue

                zmq_pub_state.send_json(
                    {
                        "timestamp": float(joint_ts),
                        "joints": list(joints),
                    }
                )
            rate.sleep()

    def policy_action_subscriber(self):
        """
        Subscribe joint-angle policy actions from ZMQ and forward to ROS
        as sensor_msgs/JointState on cmd_joint_topic (/arm_joint_target_position).
        Action vector: 7D = [joint1..joint6, gripper_joint] (gripper ignored).
        One ZMQ message → one JointState published → robot moves to that target.
        """
        pub = rospy.Publisher(self.cmd_joint_topic, JointState, queue_size=1)

        context = zmq.Context()
        zmq_sub = context.socket(zmq.SUB)
        zmq_sub.connect(self.zmq_policy_action_connect)
        zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[A1Server] ZMQ SUB connected to {self.zmq_policy_action_connect}")

        while not rospy.is_shutdown():
            try:
                action = zmq_sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue

            if not isinstance(action, dict):
                continue

            joints = action.get("joints")
            if not isinstance(joints, (list, tuple)):
                continue

            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.name = [
                "arm_joint1", "arm_joint2", "arm_joint3",
                "arm_joint4", "arm_joint5", "arm_joint6",
            ]
            msg.position = [float(j) for j in joints[:6]]
            msg.velocity = []
            msg.effort = []
            pub.publish(msg)
