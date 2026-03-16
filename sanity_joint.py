import rospy                                                                                                                                                                                                        
from signal_arm.msg import arm_control                                                                                                                                                                              
from sensor_msgs.msg import JointState                                                                                                                                                                              
                                                                                                                                                                                                                    
def get_current_joints():
    msg = rospy.wait_for_message('/joint_states_host', JointState, timeout=5.0)
    name_to_pos = dict(zip(msg.name, msg.position))
    names = ['arm_joint1','arm_joint2','arm_joint3','arm_joint4','arm_joint5','arm_joint6']
    return [name_to_pos[n] for n in names]

def publish_joint_command():
    rospy.init_node('joint_sanity_check', anonymous=True)
    pub = rospy.Publisher('/arm_joint_command_host', arm_control, queue_size=10)
    rospy.sleep(0.5)  # let publisher register

    current = get_current_joints()
    rospy.loginfo(f"Current joints: {[f'{v:.3f}' for v in current]}")

    target = current[:]
    target[0] += 0.1  # nudge joint1 by +0.1 rad only

    steps = 50
    rate = rospy.Rate(50)  # 50 Hz, matches robot control rate

    for step in range(steps):
        alpha = (step + 1) / steps
        interp = [c + alpha * (t - c) for c, t in zip(current, target)]

        msg = arm_control()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.p_des = interp
        msg.v_des = [0.0] * 6
        msg.kp = [200.0] * 6
        msg.kd = [10.0] * 6
        msg.t_ff = [0.0] * 6
        msg.mode = 1
        pub.publish(msg)
        rate.sleep()

    rospy.loginfo("Done. Joint1 moved +0.1 rad.")

if __name__ == '__main__':
    try:
        publish_joint_command()
    except rospy.ROSInterruptException:
        pass
