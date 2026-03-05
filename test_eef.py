import rospy
from geometry_msgs.msg import Pose, PoseArray


def create_pose(x, y, z, w, ox, oy, oz):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = w
    pose.orientation.x = ox
    pose.orientation.y = oy
    pose.orientation.z = oz
    return pose


def main():
    rospy.init_node("pose_array_publisher")
    pub = rospy.Publisher("/arm_target_trajectory", PoseArray, queue_size=10)

    # Wait for subscribers to connect
    rate = rospy.Rate(10)
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rate.sleep()

    msg = PoseArray()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"

    msg.poses.append(create_pose(0.08, 0.0, 0.30, 0.5, 0.5, 0.5, 0.5))
    msg.poses.append(create_pose(0.08, 0.0, 0.40, 0.5, 0.5, 0.5, 0.5))
    msg.poses.append(create_pose(0.08, 0.0, 0.54, 0.5, 0.5, 0.5, 0.5))

    pub.publish(msg)
    rospy.loginfo("Published PoseArray with 3 poses")


if __name__ == "__main__":
    main()