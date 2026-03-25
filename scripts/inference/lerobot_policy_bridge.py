#!/usr/bin/env python3
"""Policy bridge — replaces lerobot-a1-jointtracker-bridge in just teleop.

Receiver side is UNCHANGED from just teleop (/home/eric/lerobot/justfile):
    roscore          (from /home/eric/A1_SDK)
    single_arm_node  (from /home/eric/A1_SDK)
    joint_tracker    (from /home/eric/A1_SDK)

This script is the sender side only. It reads ground-truth observations from
a LeRobot dataset (teacher-forcing), calls the policy, and publishes each
action in the returned chunk at 60 Hz — exactly as the SO leader bridge does:

    /arm_joint_target_position     sensor_msgs/JointState  (6 arm joints, radians)
    /gripper_position_control_host signal_arm/gripper_position_control (stroke mm)

Usage:
    python lerobot_policy_bridge.py --episode 0
    python lerobot_policy_bridge.py --episode 0 --dry-run
    python lerobot_policy_bridge.py --episode 0 --no-step-mode
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ── ROS path (Eric's A1 SDK, same as just teleop) ──────────────────────────
_SDK = "/home/eric/A1_SDK/install"
for _p in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    f"{_SDK}/lib/python3/dist-packages",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

# ── openpi client ───────────────────────────────────────────────────────────
_OPENPI = "/home/eric/openpi/packages/openpi-client/src"
if os.path.isdir(_OPENPI) and _OPENPI not in sys.path:
    sys.path.insert(0, _OPENPI)

from openpi_client import websocket_client_policy as _ws

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control

# Joint names — identical to the bridge
_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "arm_joint6",
]

_DATASET_ROOT = "/home/eric/lerobot/data/a1_v21_old"


# ── Publish one action — identical format to the bridge ────────────────────

def _publish(joint_pub, gripper_pub, action7: np.ndarray, dry_run: bool,
             gripper_scale: float = 1.0, gripper_offset: float = 0.0):
    if dry_run:
        return
    now = rospy.Time.now()

    js = JointState()
    js.header.stamp = now
    js.name = _JOINT_NAMES
    js.position = [float(v) for v in action7[:6]]
    joint_pub.publish(js)

    gm = gripper_position_control()
    gm.header.stamp = now
    gm.gripper_stroke = float(action7[6]) * gripper_scale + gripper_offset
    gripper_pub.publish(gm)


# ── Dataset helpers ─────────────────────────────────────────────────────────

def _load_episode(dataset_root: str, episode_index: int):
    import pandas as pd
    chunk = episode_index // 1000
    path = f"{dataset_root}/data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _load_image(dataset_root: str, episode_index: int, cam: str, frame_index: int) -> np.ndarray:
    chunk = episode_index // 1000
    p = f"{dataset_root}/images/chunk-{chunk:03d}/episode_{episode_index:06d}/{cam}/{frame_index:06d}.jpg"
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Main execution loop ─────────────────────────────────────────────────────

def run(args):
    rospy.init_node("lerobot_policy_bridge", anonymous=False)

    joint_pub   = rospy.Publisher("/arm_joint_target_position",     JointState,               queue_size=10)
    gripper_pub = rospy.Publisher("/gripper_position_control_host", gripper_position_control,  queue_size=10)
    time.sleep(0.5)  # let publishers register, same as bridge startup

    print(f"[Bridge] Connecting to policy at ws://{args.host}:{args.port} ...")
    policy = _ws.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[Bridge] Connected. {policy.get_server_metadata()}")

    df = _load_episode(args.dataset, args.episode)
    n = len(df) if args.max_steps <= 0 else min(len(df), args.max_steps)
    rate = rospy.Rate(args.hz)

    print(f"[Bridge] episode={args.episode}  frames={n}  hz={args.hz}  "
          f"step_mode={args.step_mode}  dry_run={args.dry_run}")

    t = 0
    chunk_idx = 0
    while t < n and not rospy.is_shutdown():
        row = df.iloc[t]
        frame_idx = int(row["frame_index"])

        frame0 = _load_image(args.dataset, args.episode, "cam0", frame_idx)
        frame1 = _load_image(args.dataset, args.episode, "cam1", frame_idx)

        state7 = np.asarray(row["observation.state"], dtype=np.float32)[:7]

        obs = {
            "observation/image":       frame0,
            "observation/wrist_image": frame1,
            "observation/state":       state7,
            "prompt":                  args.prompt,
        }

        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        horizon = actions.shape[0]
        if args.chunk_size > 0:
            actions = actions[:args.chunk_size]
        exec_len = actions.shape[0]
        print(f"  chunk {chunk_idx}  horizon={horizon}  exec={exec_len}  "
              f"actions[0]={np.round(actions[0, :6], 3).tolist()}")

        # Publish each action at args.hz — same as bridge publish loop
        for action in actions:
            if rospy.is_shutdown():
                return
            _publish(joint_pub, gripper_pub, action[:7], args.dry_run,
                     args.gripper_scale, args.gripper_offset)
            rate.sleep()

        t += exec_len
        chunk_idx += 1

        if args.num_chunks > 0 and chunk_idx >= args.num_chunks:
            print(f"[Bridge] Reached {args.num_chunks} chunk(s), stopping.")
            break

        if args.step_mode:
            try:
                cmd = input(f"  chunk {chunk_idx} done. Enter=next, q=quit: ").strip().lower()
            except EOFError:
                cmd = "q"
            if cmd in {"q", "quit"}:
                break

    print(f"[Bridge] Done. {chunk_idx} chunks executed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default=_DATASET_ROOT)
    parser.add_argument("--episode",   type=int, default=0)
    parser.add_argument("--host",      default="127.0.0.1")
    parser.add_argument("--port",      type=int, default=8001)
    parser.add_argument("--prompt",    default="pick up the marker and place it into the red plate")
    parser.add_argument("--hz",        type=float, default=60.0)
    parser.add_argument("--max-steps", type=int,   default=0)
    parser.add_argument("--step-mode",    action="store_true",  default=True)
    parser.add_argument("--no-step-mode", dest="step_mode", action="store_false")
    parser.add_argument("--num-chunks",  type=int, default=0,
                        help="Stop after this many chunks (0 = run all)")
    parser.add_argument("--chunk-size",  type=int, default=0,
                        help="Execute only first N actions per chunk (0 = full horizon)")
    parser.add_argument("--gripper-scale",  type=float, default=-58.14,
                        help="Multiply policy gripper output by this to get stroke mm")
    parser.add_argument("--gripper-offset", type=float, default=-18.84,
                        help="Add this to (gripper * scale) to get stroke mm")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
