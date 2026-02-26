#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

# Let script run even if user forgets to source ROS env.
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

try:
    import rospy
    from sensor_msgs.msg import JointState
    from signal_arm.msg import arm_control
    from signal_arm.msg import gripper_joint_command
    from signal_arm.msg import gripper_position_control
except Exception as exc:
    print(
        f"ROS import failed: {exc}\n"
        "Tip: source install/setup.bash (or /opt/ros/noetic/setup.bash) first.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_gain_list(text: str, n: int, name: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"{name} must have 1 or {n} values, got {len(vals)}")
    return vals


class DragComplianceNode:
    def __init__(self, args):
        self.args = args
        self.latest_joint_pos = None
        self.latest_joint_names = None
        self.latest_gripper_stroke = None
        self.publisher = rospy.Publisher(args.cmd_topic, arm_control, queue_size=10)
        self.gripper_force_pub = rospy.Publisher(
            args.gripper_force_topic, gripper_joint_command, queue_size=10
        )
        self.gripper_position_pub = rospy.Publisher(
            args.gripper_position_topic, gripper_position_control, queue_size=10
        )
        self.subscriber = rospy.Subscriber(
            args.joint_topic, JointState, self.joint_cb, queue_size=20
        )
        self.gripper_subscriber = rospy.Subscriber(
            args.gripper_feedback_topic, JointState, self.gripper_cb, queue_size=20
        )
        self.rate = rospy.Rate(args.rate)

    def joint_cb(self, msg: JointState):
        if len(msg.position) < self.args.dof:
            return
        self.latest_joint_pos = list(msg.position[: self.args.dof])
        self.latest_joint_names = list(msg.name[: self.args.dof])

    def gripper_cb(self, msg: JointState):
        if len(msg.position) < 1:
            return
        self.latest_gripper_stroke = float(msg.position[0])

    def make_command(self, p_des: List[float], kp: List[float], kd: List[float]) -> arm_control:
        cmd = arm_control()
        cmd.header.stamp = rospy.Time.now()
        cmd.p_des = p_des
        cmd.v_des = [0.0] * self.args.dof
        cmd.t_ff = [0.0] * self.args.dof
        cmd.kp = kp
        cmd.kd = kd
        cmd.mode = self.args.mode
        return cmd

    def run(self):
        rospy.loginfo("Waiting for joint state from %s ...", self.args.joint_topic)
        timeout_t = rospy.Time.now().to_sec() + self.args.wait_timeout if self.args.wait_timeout > 0 else None
        while not rospy.is_shutdown() and self.latest_joint_pos is None:
            if timeout_t is not None and rospy.Time.now().to_sec() > timeout_t:
                rospy.logerr("No joint state received in %.1f sec. Exit.", self.args.wait_timeout)
                return 1
            self.rate.sleep()

        kp = parse_gain_list(self.args.kp, self.args.dof, "kp")
        kd = parse_gain_list(self.args.kd, self.args.dof, "kd")
        rospy.logwarn(
            "Drag compliance active. mode=%d kp=%s kd=%s cmd_topic=%s hold_gripper_position=%s gripper_position_topic=%s",
            self.args.mode,
            ",".join(f"{x:.3f}" for x in kp),
            ",".join(f"{x:.3f}" for x in kd),
            self.args.cmd_topic,
            str(self.args.hold_gripper_position),
            self.args.gripper_position_topic,
        )

        while not rospy.is_shutdown():
            if self.latest_joint_pos is None:
                self.rate.sleep()
                continue

            # Hold desired position at current measured position, with low gains.
            cmd = self.make_command(self.latest_joint_pos, kp, kd)
            self.publisher.publish(cmd)

            if self.args.hold_gripper_position:
                gpcmd = gripper_position_control()
                gpcmd.header.stamp = rospy.Time.now()
                if self.latest_gripper_stroke is not None:
                    gpcmd.gripper_stroke = self.latest_gripper_stroke
                else:
                    gpcmd.gripper_stroke = self.args.default_gripper_stroke
                self.gripper_position_pub.publish(gpcmd)

            if self.args.zero_gripper_force:
                gcmd = gripper_joint_command()
                gcmd.header.stamp = rospy.Time.now()
                gcmd.gripper_force = 0.0
                self.gripper_force_pub.publish(gcmd)
            self.rate.sleep()
        return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Low-stiffness drag mode for A1 by publishing arm_control continuously."
    )
    parser.add_argument("--joint-topic", default="/joint_states_host")
    parser.add_argument("--cmd-topic", default="/arm_joint_command_host")
    parser.add_argument("--dof", type=int, default=6)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--mode", type=int, default=0, help="Driver mode (default: 0, MIT)")
    parser.add_argument(
        "--kp",
        default="1.5,1.5,1.2,0.8,0.6,0.4",
        help="1 or DOF comma-separated values",
    )
    parser.add_argument(
        "--kd",
        default="0.08,0.08,0.06,0.05,0.04,0.03",
        help="1 or DOF comma-separated values",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for first joint state; <=0 means wait forever.",
    )
    parser.add_argument("--gripper-force-topic", default="/gripper_force_control_host")
    parser.add_argument("--gripper-position-topic", default="/gripper_position_control_host")
    parser.add_argument("--gripper-feedback-topic", default="/gripper_stroke_host")
    parser.add_argument(
        "--hold-gripper-position",
        action="store_true",
        help="Continuously publish current gripper_stroke to gripper_position topic.",
    )
    parser.add_argument(
        "--default-gripper-stroke",
        type=float,
        default=0.0,
        help="Fallback gripper_stroke when no feedback is received.",
    )
    parser.add_argument(
        "--zero-gripper-force",
        action="store_true",
        help="Continuously publish gripper_force=0.0 during drag mode.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node("a1_drag_compliance", anonymous=False)

    try:
        node = DragComplianceNode(args)
        code = node.run()
    except ValueError as exc:
        rospy.logerr(str(exc))
        code = 2
    except rospy.ROSInterruptException:
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
