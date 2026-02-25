from geometry_msgs.msg import PoseStamped
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
import threading
import zmq
import time
import socket
from datacoach.constants import *
import rospy

class A1Server:
    def __init__(self):
        self.current_feedback = {
            'x': None, 'y': None, 'z': None,
            'qx': None, 'qy': None, 'qz': None, 'qw': None,
            'gripper': None
        }
        self.feedback_lock = threading.Lock()
        self.processed_data_queue = []


    def leader_data_receiver(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", LEROBOT_PORT))
        print(f"[A1Server] Listening for leader data on UDP 127.0.0.1:{LEROBOT_PORT} ...")

        while not rospy.is_shutdown():
            data, _ = sock.recvfrom(4096)
            parts = data.decode().split(",")

            try:
                x, y, z = map(float, parts[:3])
            except Exception:
                continue

            tx = x * SCALE[0] + OFFSET[0]
            ty = y * SCALE[1] + OFFSET[1]
            tz = z * SCALE[2] + OFFSET[2]

            pose_data = {
                'pos': (tx, ty, tz),
                'ori': tuple(map(float, parts[3:7])) if len(parts) >= 7 else None,
                'gripper': float(parts[7]) if len(parts) >= 8 else None
            }

            self.processed_data_queue.append(pose_data)
            time.sleep(1/ROBOT_FPS)  # 50Hz



    def ros_publisher(self):
        ''' 
        publish processed leader data to A1 via ROS topics 
        and publish commanded_state via ZMQ
        '''
        pub = rospy.Publisher('/a1_ee_target', PoseStamped, queue_size=10)
        gripper_pub = rospy.Publisher('/gripper_position_control_host', gripper_position_control, queue_size=10)
        rate = rospy.Rate(ROBOT_FPS) 

        # ZMQ publisher for commanded_state
        context = zmq.Context()
        zmq_pub_cmd = context.socket(zmq.PUB)
        zmq_pub_cmd.bind(f"tcp://127.0.0.1:{ZMQ_CMD_PORT}")
        print(f"[A1Server] ZMQ publisher (commanded_state) bound to tcp://127.0.0.1:{ZMQ_CMD_PORT}")

        while not rospy.is_shutdown():
            if self.processed_data_queue:
                pose_data = self.processed_data_queue.pop(0)

                pose_msg = PoseStamped()
                pose_msg.header.frame_id = 'world'
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = pose_data['pos']

                if pose_data['ori']:
                    pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = pose_data['ori']
                pub.publish(pose_msg)

                # gripper publish
                if pose_data['gripper'] is not None:
                    grip_msg = gripper_position_control()
                    grip_msg.header = Header()
                    grip_msg.header.stamp = rospy.Time.now()
                    grip_msg.gripper_stroke = pose_data['gripper']
                    gripper_pub.publish(grip_msg)

                zmq_pub_cmd.send_json({
                    'timestamp': time.time(),
                    'pos': pose_data['pos'],
                    'ori': pose_data['ori'],
                    'gripper': pose_data['gripper']
                })
                print("send")
                print({
                    'timestamp': time.time(),
                    'pos': pose_data['pos'],
                    'ori': pose_data['ori'],
                    'gripper': pose_data['gripper']
                })
            rate.sleep()


    def ros_subscriber(self):
        """
        Subscribe pose & gripper feedback via ROS
        Fuse them into one state per timestep
        Publish via ZMQ for data collection
        """

        # ================== ZMQ ==================
        context = zmq.Context()
        zmq_pub_cmd = context.socket(zmq.PUB)
        zmq_pub_cmd.bind(f"tcp://127.0.0.1:{ZMQ_STATE_PORT}")
        print(f"[A1Server] ZMQ PUB bound to tcp://127.0.0.1:{ZMQ_STATE_PORT}")

        # Avoid dropping the first few messages
        time.sleep(0.2)

        # ================== Shared State ==================
        self.feedback_lock = threading.Lock()

        self.pose_data = {
            'pos': None,   # (x, y, z)
            'ori': None    # (qx, qy, qz, qw)
        }
        self.gripper_data = None

        # ================== ROS Callbacks ==================
        def pose_callback(msg):
            with self.feedback_lock:
                self.pose_data['pos'] = (
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z
                )
                self.pose_data['ori'] = (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w
                )


        def gripper_callback(msg):
            with self.feedback_lock:
                self.gripper_data = msg.position[0]

        # ================== ROS Subscribers ==================
        rospy.Subscriber('/end_effector_pose', PoseStamped, pose_callback)
        rospy.Subscriber('/gripper_stroke_host',
                        JointState,
                        gripper_callback)

        # ================== ZMQ Publish Loop ==================
        rate = rospy.Rate(ROBOT_FPS)  # dataset timestep (Hz)

        while not rospy.is_shutdown():
            with self.feedback_lock:
                pos = self.pose_data['pos']
                ori = self.pose_data['ori']
                gripper = self.gripper_data

            # Only publish once all fields are available
            if pos is not None and ori is not None and gripper is not None:
                zmq_pub_cmd.send_json({
                    'timestamp': time.time(),
                    'pos': pos,
                    'ori': ori,
                    'gripper': gripper
                })
                print("reveive")
                print({
                    'timestamp': time.time(),
                    'pos': pos,
                    'ori': ori,
                    'gripper': gripper
                })
            print("-----")
            rate.sleep()
