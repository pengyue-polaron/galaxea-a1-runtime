#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Bridge LeRobot SO leader joint angles to A1 jointTracker target topic.

Usage example:
    lerobot-a1-jointtracker-bridge \
      --leader-port /dev/ttyACM0 \
      --leader-id my_leader \
      --hz 60
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

# Allow using rospy in conda python by appending common ROS Python paths.
import pathlib as _pathlib

_A1_SDK = _pathlib.Path(__file__).parents[4] / "A1_SDK" / "install"

for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control

from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig


def _parse_csv_floats(text: str, expected_len: int, name: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(values)}: {text}")
    return values


def _parse_csv_strings(text: str, expected_len: int, name: str) -> list[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(f"{name} expects {expected_len} values, got {len(values)}: {text}")
    return values


def _detect_leader_joint_keys(action: dict[str, float], dof: int) -> list[str]:
    v2 = [f"joint{i}.pos" for i in range(dof)]
    if all(k in action for k in v2):
        return v2

    legacy = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ][:dof]
    if all(k in action for k in legacy):
        return legacy

    fallback = sorted([k for k in action if k.endswith(".pos") and "gripper" not in k])[:dof]
    if len(fallback) == dof:
        return fallback

    raise RuntimeError(f"Could not detect {dof} leader joint keys from action keys: {sorted(action.keys())}")


@dataclass
class ArmStateCache:
    msg: JointState | None = None

    def cb(self, msg: JointState):
        self.msg = msg

    def get_positions(self, ordered_names: list[str]) -> list[float]:
        if self.msg is None:
            raise RuntimeError("No arm joint state received yet.")

        msg = self.msg
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        if all(name in name_to_idx for name in ordered_names):
            indices = [name_to_idx[n] for n in ordered_names]
            if all(0 <= idx < len(msg.position) for idx in indices):
                return [float(msg.position[idx]) for idx in indices]
            raise RuntimeError(
                "JointState name/position mismatch on /joint_states_host: "
                f"{len(msg.name)} names but {len(msg.position)} positions."
            )

        if len(msg.position) >= len(ordered_names):
            return [float(x) for x in msg.position[: len(ordered_names)]]

        raise RuntimeError(
            "JointState on /joint_states_host is incomplete: "
            f"{len(msg.position)} positions for {len(ordered_names)} requested joints."
        )


def main():
    parser = argparse.ArgumentParser(description="Bridge LeRobot leader joints -> A1 jointTracker topic.")
    parser.add_argument("--leader-port", required=True, help="Serial port of SO leader, e.g. /dev/ttyACM0")
    parser.add_argument("--leader-id", default="my_leader")
    parser.add_argument("--leader-use-degrees", action="store_true", default=True)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--dof", type=int, default=6)

    parser.add_argument("--joint-states-topic", default="/joint_states_host")
    parser.add_argument("--target-topic", default="/arm_joint_target_position")
    parser.add_argument("--target-joint-names", default="arm_joint1,arm_joint2,arm_joint3,arm_joint4,arm_joint5,arm_joint6")

    # Relative mode avoids sudden jumps: target = a1_start + mapped(leader - leader_start)
    parser.add_argument("--relative", action="store_true", default=True)
    parser.add_argument("--input-degrees", action="store_true", default=True)
    parser.add_argument("--scale", default="1,1,1,1,1,1", help="Per-joint scale in relative mode")
    parser.add_argument("--sign", default="-1,1,1,-1,1,-1", help="Per-joint sign (+1/-1) in relative mode")
    parser.add_argument("--bias-rad", default="0,0,0,0,0,0", help="Per-joint additive bias in radians")
    parser.add_argument("--lower-limits", default="-2.8798,0,-3.3161,-2.8798,-1.6581,-2.8798")
    parser.add_argument("--upper-limits", default="2.8798,3.2289,0,2.8798,1.6581,2.8798")
    parser.add_argument("--gripper-enabled", action="store_true", default=True, help="Publish A1 gripper position command")
    parser.add_argument("--gripper-source-key", default="gripper.pos", help="Leader action key for gripper")
    parser.add_argument("--gripper-position-topic", default="/gripper_position_control_host")
    parser.add_argument("--gripper-min-stroke-mm", type=float, default=0.0, help="A1 gripper closed stroke (mm)")
    parser.add_argument("--gripper-max-stroke-mm", type=float, default=60.0, help="A1 gripper open stroke (mm)")
    parser.add_argument("--gripper-invert", action="store_true", default=False, help="Invert leader gripper percentage")
    args = parser.parse_args()

    if args.hz <= 0:
        raise ValueError(f"--hz must be > 0, got {args.hz}")

    dof = args.dof
    target_names = _parse_csv_strings(args.target_joint_names, dof, "--target-joint-names")
    scale = _parse_csv_floats(args.scale, dof, "--scale")
    sign = _parse_csv_floats(args.sign, dof, "--sign")
    bias_rad = _parse_csv_floats(args.bias_rad, dof, "--bias-rad")
    lower = _parse_csv_floats(args.lower_limits, dof, "--lower-limits")
    upper = _parse_csv_floats(args.upper_limits, dof, "--upper-limits")

    rospy.init_node("lerobot_a1_jointtracker_bridge", anonymous=False)
    state_cache = ArmStateCache()
    _ = rospy.Subscriber(args.joint_states_topic, JointState, state_cache.cb, queue_size=10)
    pub = rospy.Publisher(args.target_topic, JointState, queue_size=10)
    gripper_pub = rospy.Publisher(args.gripper_position_topic, gripper_position_control, queue_size=10)

    leader = SOLeader(
        SOLeaderTeleopConfig(
            id=args.leader_id,
            port=args.leader_port,
            use_degrees=args.leader_use_degrees,
        )
    )
    leader.connect(calibrate=False)

    try:
        rospy.loginfo("Waiting for first leader action and A1 joint state...")
        rate = rospy.Rate(args.hz)
        action0 = leader.get_action()
        leader_keys = _detect_leader_joint_keys(action0, dof)
        if args.gripper_enabled and args.gripper_source_key not in action0:
            raise RuntimeError(
                f"Gripper source key '{args.gripper_source_key}' not found in leader action keys: {sorted(action0.keys())}"
            )
        leader_start = [float(action0[k]) for k in leader_keys]

        while not rospy.is_shutdown():
            try:
                a1_start = state_cache.get_positions(target_names)
                break
            except RuntimeError:
                rate.sleep()

        rospy.loginfo("Leader keys: %s", leader_keys)
        rospy.loginfo("A1 joint names: %s", target_names)
        rospy.loginfo("Bridge running at %.1f Hz -> %s", args.hz, args.target_topic)

        while not rospy.is_shutdown():
            action = leader.get_action()
            leader_now = [float(action[k]) for k in leader_keys]

            if args.relative:
                delta = [leader_now[i] - leader_start[i] for i in range(dof)]
                if args.input_degrees:
                    delta = [math.radians(x) for x in delta]
                target = [a1_start[i] + sign[i] * scale[i] * delta[i] + bias_rad[i] for i in range(dof)]
            else:
                target = leader_now
                if args.input_degrees:
                    target = [math.radians(x) for x in target]
                target = [sign[i] * scale[i] * target[i] + bias_rad[i] for i in range(dof)]

            target = [min(upper[i], max(lower[i], target[i])) for i in range(dof)]

            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = target_names
            msg.position = target
            pub.publish(msg)

            if args.gripper_enabled:
                gripper_pct = float(action[args.gripper_source_key])
                gripper_pct = max(0.0, min(100.0, gripper_pct))
                if args.gripper_invert:
                    gripper_pct = 100.0 - gripper_pct
                stroke = args.gripper_min_stroke_mm + (args.gripper_max_stroke_mm - args.gripper_min_stroke_mm) * (
                    gripper_pct / 100.0
                )

                gmsg = gripper_position_control()
                gmsg.header.stamp = msg.header.stamp
                gmsg.gripper_stroke = float(stroke)
                gripper_pub.publish(gmsg)
            rate.sleep()
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
