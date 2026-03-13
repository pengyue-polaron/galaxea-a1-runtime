import rospy
from sensor_msgs.msg import JointState

def publish_joint_state():
    rospy.init_node('joint_state_publisher', anonymous=True)
    pub = rospy.Publisher('/arm_joint_target_position', JointState, queue_size=10)
    rate = rospy.Rate(30) # 10 Hz
    joint_state = JointState()
    joint_state.header.seq = 0
    joint_state.header.stamp = rospy.Time.now()
    joint_state.header.frame_id = 'world'
    joint_state.name = ['arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5', 'arm_joint6']
    joint_state.velocity = []
    joint_state.effort = []
    # Initialize positions to zeros
    joint_state.position = [0, 0, 0, 0, 0, 0]
    steps = 100 # Number of steps

    #to reach the target position
    target_position = [5., 0, 0, 0, 0, 0]
    step_increment = [(target - current) / steps for target, current in zip(target_position, joint_state.position)]
    print(step_increment)
    for step in range(steps):
        print(joint_state)
        joint_state.header.stamp = rospy.Time.now() # Update the timestamp
        joint_state.position = [current + increment for current, increment in zip(joint_state.position, step_increment)]
        pub.publish(joint_state)
        rate.sleep()
    rospy.loginfo("Published JointState message to /arm_joint_target_position")

if __name__ == '__main__':
    try:
        publish_joint_state()
    except rospy.ROSInterruptException:
        pass