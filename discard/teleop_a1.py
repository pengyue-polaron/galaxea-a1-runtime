import rospy
from geometry_msgs.msg import PoseStamped
from signal_arm.msg import gripper_position_control
from std_msgs.msg import Header
import socket
import math

def main():
    rospy.init_node('ee_target_relay')
    pub = rospy.Publisher('/a1_ee_target', PoseStamped, queue_size=10)
    gripper_pub = rospy.Publisher('/gripper_position_control_host', gripper_position_control, queue_size=10)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 7005))
    pose = PoseStamped()
    pose.header.frame_id = 'world'
    pose.pose.orientation.x = 0.5
    pose.pose.orientation.y = 0.5
    pose.pose.orientation.z = 0.5
    pose.pose.orientation.w = 0.5

    # 保存当前机械臂的上行反馈
    current_feedback = {'x': None, 'y': None, 'z': None, 'qx': None, 'qy': None, 'qz': None, 'qw': None}

    # subscribe A1 feedback
    def feedback_callback(msg):
        current_feedback['x'] = msg.pose.position.x
        current_feedback['y'] = msg.pose.position.y
        current_feedback['z'] = msg.pose.position.z
        current_feedback['qx'] = msg.pose.orientation.x
        current_feedback['qy'] = msg.pose.orientation.y
        current_feedback['qz'] = msg.pose.orientation.z
        current_feedback['qw'] = msg.pose.orientation.w
    rospy.Subscriber('/a1_ee_target', PoseStamped, feedback_callback)

    import threading
    feedback_lock = threading.Lock()

    for _ in range(6):
        print("____")
    while not rospy.is_shutdown():
        data, _ = sock.recvfrom(4096)
        parts = data.decode().split(",")
        x, y, z = map(float, parts[:3])
        pose.header.stamp = rospy.Time.now()
        scale = [2,2,1.5]  # 缩放
        offset = [0,0,0.05]  # 偏移
        tx = x * scale[0] + offset[0]
        ty = y * scale[1] + offset[1]
        tz = z * scale[2] + offset[2]
        pose.pose.position.x = tx
        pose.pose.position.y = ty
        pose.pose.position.z = tz
        if len(parts) >= 7:
            pose.pose.orientation.x = float(parts[3])
            pose.pose.orientation.y = float(parts[4])
            pose.pose.orientation.z = float(parts[5])
            pose.pose.orientation.w = float(parts[6])
            
        pub.publish(pose)
        # 解析夹爪角度（第8个字段），但不再核对和打印夹爪
        gripper_angle = None
        if len(parts) >= 8:
            try:
                gripper_angle = float(parts[7])
                print(gripper_angle)
            except Exception:
                gripper_angle = None
        # 发布夹爪控制消息（仿照keyboardcontrol.py）
        if gripper_angle is not None:
            gripper_msg = gripper_position_control()
            gripper_msg.header = Header()
            gripper_msg.header.stamp = rospy.Time.now()
            gripper_msg.header.frame_id = ''
            gripper_msg.gripper_stroke = gripper_angle
            gripper_pub.publish(gripper_msg)
        

        # 打印5行：目标位置、目标方向、上行位置、上行方向、是否一致
        tgt_pos = f"[目标位置] x={tx:.3f}, y={ty:.3f}, z={tz:.3f}"
        if len(parts) >= 7:
            tgt_ori = f"[目标方向] x={float(parts[3]):.3f}, y={float(parts[4]):.3f}, z={float(parts[5]):.3f}, w={float(parts[6]):.3f}"
        else:
            tgt_ori = "[目标方向] N/A"
        with feedback_lock:
            if current_feedback['x'] is not None:
                fb_pos = f"[上行位置] x={current_feedback['x']:.3f}, y={current_feedback['y']:.3f}, z={current_feedback['z']:.3f}"
                fb_ori = f"[上行方向] x={current_feedback['qx']:.3f}, y={current_feedback['qy']:.3f}, z={current_feedback['qz']:.3f}, w={current_feedback['qw']:.3f}"
                # 判断是否完全一致（允许微小误差1e-3），目标位置需加offset后再比较
                pos_equal = abs(tx-current_feedback['x'])<1e-3 and abs(ty-current_feedback['y'])<1e-3 and abs(tz-current_feedback['z'])<1e-3
                if len(parts) >= 7:
                    ori_equal = all(abs(float(parts[i+3])-current_feedback[k])<1e-3 for i,k in enumerate(['qx','qy','qz','qw']))
                else:
                    ori_equal = False
                all_equal = pos_equal and ori_equal
                eq_line = f"[是否一致] {'是Yes ###' if all_equal else '否No ---'}"
            else:
                fb_pos = "[上行位置] N/A"
                fb_ori = "[上行方向] N/A"
                eq_line = "[是否一致] N/A"
        # 覆盖式打印（每次刷新5行）
        output_lines = [tgt_pos, tgt_ori, fb_pos, fb_ori, eq_line]
        print("\033[F" * 5, end='')  # 光标上移5行
        for line in output_lines:
            # print(line)
            pass

if __name__ == "__main__":
    main()


# sudo -E bash -c "source /opt/ros/noetic/setup.bash && source ~/Projects/A1_SDK-galaxea-main/install/setup.bash && python3 /home/lewis/Projects/A1_SDK-galaxea-main/receiver.py"