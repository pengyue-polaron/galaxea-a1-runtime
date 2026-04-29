#!/usr/bin/env python3
"""Gripper stroke test: publish a stroke value to /gripper_position_control_host
and visually verify what the gripper does. Usage:

    python3 scripts/inference/test_gripper.py 0     # try stroke = 0 mm
    python3 scripts/inference/test_gripper.py 60    # try stroke = 60 mm
    python3 scripts/inference/test_gripper.py 200   # try stroke = 200 mm
    python3 scripts/inference/test_gripper.py -60   # try negative stroke

Run roscore + driver first (the bridge does NOT need to be running).
"""
import argparse
import os
import sys
import time

sys.path = [p for p in sys.path if "/opt/ros/humble" not in p]
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/home/nyu/A1-Research/third_party/A1_SDK/install/lib/python3/dist-packages")

import rospy
from signal_arm.msg import gripper_position_control


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stroke", type=float, help="Stroke value to send (mm)")
    p.add_argument("--rate", type=float, default=10.0, help="Hz")
    p.add_argument("--duration", type=float, default=3.0, help="Seconds")
    args = p.parse_args()

    rospy.init_node("test_gripper", anonymous=True)
    pub = rospy.Publisher("/gripper_position_control_host", gripper_position_control, queue_size=10)
    time.sleep(0.5)  # let publisher register

    print(f"[GripperTest] Sending stroke={args.stroke} mm at {args.rate} Hz for {args.duration}s")
    print("[GripperTest] Watch the gripper — does it match expectation?")

    rate = rospy.Rate(args.rate)
    end = time.time() + args.duration
    while not rospy.is_shutdown() and time.time() < end:
        msg = gripper_position_control()
        msg.header.stamp = rospy.Time.now()
        msg.gripper_stroke = float(args.stroke)
        pub.publish(msg)
        rate.sleep()
    print("[GripperTest] Done.")


if __name__ == "__main__":
    main()
