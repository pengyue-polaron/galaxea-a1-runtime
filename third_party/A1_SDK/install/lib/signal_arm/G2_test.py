#!/usr/bin/env python3
import rospy
from std_msgs.msg import Header
from signal_arm.msg import gripper_position_control

def gripper_publisher():
    pub = rospy.Publisher('/gripper_position_control_host', gripper_position_control, queue_size=10)
    rospy.init_node('gripper_publisher', anonymous=True)
    rate = rospy.Rate(2)  # 0.5 seconds

    gripper_stroke = 0.0
    increment = 10.0
    max_stroke = 100.0
    min_stroke = 0.0

    while not rospy.is_shutdown():
        # Create the message
        msg = gripper_position_control()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.gripper_stroke = gripper_stroke

        # Publish the message
        rospy.loginfo("Publishing gripper_stroke: %f", gripper_stroke)
        pub.publish(msg)

        # Update the gripper_stroke value
        gripper_stroke += increment
        if gripper_stroke > max_stroke or gripper_stroke < min_stroke:
            increment = -increment  # Reverse the direction

        # Sleep for 0.5 seconds
        rate.sleep()

if __name__ == '__main__':
    try:
        gripper_publisher()
    except rospy.ROSInterruptException:
        pass
