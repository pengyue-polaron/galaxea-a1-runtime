"""Relay node: /arm_joint_target_position (JointState) → /arm_joint_command_host (arm_control).

Subscribes to JointState target positions published by the inference stack and
republishes them as signal_arm/arm_control messages that iarm_node_single_arm accepts.

Usage:
    just joint-relay
"""

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import arm_control

KP = [20.0, 20.0, 16.0, 12.0, 8.0, 8.0]
KD = [1.0,  1.0,  0.8,  0.6,  0.4, 0.4]


def main():
    rospy.init_node("joint_target_relay")

    target_topic = rospy.get_param("~target_topic", "/arm_joint_target_position")
    command_topic = rospy.get_param("~command_topic", "/arm_joint_command_host")
    kp = rospy.get_param("~kp", KP)
    kd = rospy.get_param("~kd", KD)

    pub = rospy.Publisher(command_topic, arm_control, queue_size=10)

    def callback(msg: JointState):
        if len(msg.position) < 6:
            rospy.logwarn_throttle(5.0, f"[Relay] Expected 6 joints, got {len(msg.position)}")
            return
        cmd = arm_control()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "world"
        cmd.p_des = [float(v) for v in msg.position[:6]]
        cmd.v_des = [0.0] * 6
        cmd.kp = kp
        cmd.kd = kd
        cmd.t_ff = [0.0] * 6
        cmd.mode = 1
        pub.publish(cmd)

    rospy.Subscriber(target_topic, JointState, callback, queue_size=1)

    rospy.loginfo(f"[Relay] {target_topic} → {command_topic} (kp={kp[0]}, kd={kd[0]})")
    rospy.spin()


if __name__ == "__main__":
    main()
