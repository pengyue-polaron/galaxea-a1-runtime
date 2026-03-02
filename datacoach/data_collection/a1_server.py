from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Header

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

        self.state_pose_topic = str(self._cfg_get("state_pose_topic", "/end_effector_pose"))
        self.state_gripper_topic = str(self._cfg_get("state_gripper_topic", "/gripper_stroke_host"))
        self.cmd_pose_topic = str(self._cfg_get("cmd_pose_topic", "/a1_ee_target"))
        self.cmd_gripper_position_topic = str(
            self._cfg_get("cmd_gripper_position_topic", "/gripper_position_control_host")
        )

        self.zmq_cmd_bind = str(self._cfg_get("zmq_cmd_bind", f"tcp://127.0.0.1:{ZMQ_CMD_PORT}"))
        self.zmq_state_bind = str(self._cfg_get("zmq_state_bind", f"tcp://127.0.0.1:{ZMQ_STATE_PORT}"))
        self.zmq_policy_action_connect = str(
            self._cfg_get("zmq_policy_action_connect", f"tcp://127.0.0.1:{ZMQ_POLICY_ACTION_PORT}")
        )

        self.current_feedback = {
            "x": None,
            "y": None,
            "z": None,
            "qx": None,
            "qy": None,
            "qz": None,
            "qw": None,
            "gripper": None,
        }
        self.feedback_lock = threading.Lock()
        self.processed_data_queue = []

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

            self.processed_data_queue.append(pose_data)
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
            if self.processed_data_queue:
                pose_data = self.processed_data_queue.pop(0)

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
        Subscribe A1 ROS feedback (pose/gripper) and publish unified state to ZMQ.
        """
        context = zmq.Context()
        zmq_pub_state = context.socket(zmq.PUB)
        zmq_pub_state.bind(self.zmq_state_bind)
        print(f"[A1Server] ZMQ PUB bound to {self.zmq_state_bind}")
        time.sleep(0.2)

        self.feedback_lock = threading.Lock()
        self.pose_data = {"pos": None, "ori": None}
        self.gripper_data = None

        def pose_callback(msg):
            with self.feedback_lock:
                self.pose_data["pos"] = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                )
                self.pose_data["ori"] = (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                )

        def gripper_callback(msg):
            with self.feedback_lock:
                self.gripper_data = msg.position[0]

        rospy.Subscriber(self.state_pose_topic, PoseStamped, pose_callback)
        rospy.Subscriber(self.state_gripper_topic, JointState, gripper_callback)

        rate = rospy.Rate(max(self.publish_rate_hz, 1.0))
        while not rospy.is_shutdown():
            with self.feedback_lock:
                pos = self.pose_data["pos"]
                ori = self.pose_data["ori"]
                gripper = self.gripper_data

            if pos is not None and ori is not None and gripper is not None:
                zmq_pub_state.send_json(
                    {
                        "timestamp": time.time(),
                        "pos": pos,
                        "ori": ori,
                        "gripper": gripper,
                    }
                )
            rate.sleep()

    def policy_action_subscriber(self):
        """
        Subscribe policy actions from ZMQ and forward to A1 ROS control topics.
        """
        pub = rospy.Publisher(self.cmd_pose_topic, PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher(self.cmd_gripper_position_topic, gripper_position_control, queue_size=10)

        context = zmq.Context()
        zmq_sub = context.socket(zmq.SUB)
        zmq_sub.connect(self.zmq_policy_action_connect)
        zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[A1Server] ZMQ SUB connected to {self.zmq_policy_action_connect}")

        rate = rospy.Rate(max(self.publish_rate_hz, 1.0))
        while not rospy.is_shutdown():
            try:
                action = zmq_sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                rate.sleep()
                continue

            try:
                pos = action["pos"]
                ori = action["ori"]
                gripper = action.get("gripper", None)
            except Exception:
                rate.sleep()
                continue

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "world"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])
            pose_msg.pose.orientation.x = float(ori[0])
            pose_msg.pose.orientation.y = float(ori[1])
            pose_msg.pose.orientation.z = float(ori[2])
            pose_msg.pose.orientation.w = float(ori[3])
            pub.publish(pose_msg)

            if gripper is not None:
                grip_msg = gripper_position_control()
                grip_msg.header = Header()
                grip_msg.header.stamp = rospy.Time.now()
                grip_msg.gripper_stroke = float(gripper)
                gripper_pub.publish(grip_msg)

            rate.sleep()
