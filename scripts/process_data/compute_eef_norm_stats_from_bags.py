#!/usr/bin/env python3
"""Compute Galaxea A1 EE-pose quantile norm stats from ROS bag files.

Run inside the A1 Noetic Docker environment so rosbag and tf are available.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import rosbag


def expand_bags(patterns: list[str]) -> list[str]:
    bags: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            bags.extend(matched)
        elif os.path.isfile(pattern):
            bags.append(pattern)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for bag in bags:
        real = os.path.realpath(bag)
        if real not in seen:
            seen.add(real)
            out.append(bag)
    return out


def read_bag(bag_path: str) -> tuple[np.ndarray, list[float], list[float], dict]:
    xyzquat: list[list[float]] = []
    stroke_host: list[float] = []
    cmd_stroke: list[float] = []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _stamp in bag.read_messages(
            topics=["/end_effector_pose", "/gripper_stroke_host", "/gripper_position_control_host"]
        ):
            if topic == "/end_effector_pose":
                p = msg.pose.position
                q = msg.pose.orientation
                xyzquat.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            elif topic == "/gripper_stroke_host" and getattr(msg, "position", None):
                stroke_host.append(float(msg.position[0]))
            elif topic == "/gripper_position_control_host" and hasattr(msg, "gripper_stroke"):
                cmd_stroke.append(float(msg.gripper_stroke))
    arr = np.asarray(xyzquat, dtype=np.float64)
    summary = {
        "bag": os.path.basename(bag_path),
        "path": bag_path,
        "has_end_effector_pose": bool(len(arr)),
        "n_pose": int(len(arr)),
        "n_gripper_stroke_host": len(stroke_host),
        "n_gripper_position_control_host": len(cmd_stroke),
    }
    if len(arr):
        summary.update(
            xyz_min=np.nanmin(arr[:, :3], axis=0).round(8).tolist(),
            xyz_max=np.nanmax(arr[:, :3], axis=0).round(8).tolist(),
            quat_min=np.nanmin(arr[:, 3:7], axis=0).round(8).tolist(),
            quat_max=np.nanmax(arr[:, 3:7], axis=0).round(8).tolist(),
        )
    return arr, stroke_host, cmd_stroke, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute EE-pose q01/q99 from A1 ROS bags.")
    parser.add_argument(
        "--bag-glob",
        action="append",
        default=[],
        help="Bag path or glob. Can be passed multiple times.",
    )
    parser.add_argument("--out", default="/workspace/data/diagnostics/a1_eef_norm_stats_latest.json")
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    parser.add_argument(
        "--gripper-scale-mm",
        type=float,
        default=60.0,
        help="Bridge maps normalized gripper * this scale to mm; q stats divide command stroke by this value.",
    )
    parser.add_argument(
        "--default-globs",
        action="store_true",
        help="Also include the standard tracked A1_SDK records folder.",
    )
    args = parser.parse_args()

    patterns = list(args.bag_glob)
    if args.default_globs or not patterns:
        patterns.extend(
            [
                "/workspace/third_party/A1_SDK/data/records/*.bag",
            ]
        )
    bags = expand_bags(patterns)
    if not bags:
        raise SystemExit(f"No bag files matched: {patterns}")

    all_pose: list[np.ndarray] = []
    all_stroke_host: list[float] = []
    all_cmd_stroke: list[float] = []
    per_bag = []
    for bag_path in bags:
        arr, stroke_host, cmd_stroke, summary = read_bag(bag_path)
        per_bag.append(summary)
        if len(arr):
            all_pose.append(arr)
        all_stroke_host.extend(stroke_host)
        all_cmd_stroke.extend(cmd_stroke)

    if not all_pose:
        raise SystemExit("Matched bags contain no /end_effector_pose messages.")

    pose = np.concatenate(all_pose, axis=0)
    q_low = np.nanquantile(pose, args.q_low, axis=0)
    q_high = np.nanquantile(pose, args.q_high, axis=0)

    # Prefer commanded gripper strokes because the bridge publishes normalized * scale to this topic.
    if all_cmd_stroke:
        grip_vals = np.asarray(all_cmd_stroke, dtype=np.float64) / float(args.gripper_scale_mm)
        grip_source = "/gripper_position_control_host"
    elif all_stroke_host:
        grip_vals = np.asarray(all_stroke_host, dtype=np.float64) / float(args.gripper_scale_mm)
        grip_source = "/gripper_stroke_host"
    else:
        grip_vals = np.asarray([0.0, 1.0], dtype=np.float64)
        grip_source = "fallback_[0,1]"

    grip_low = float(np.nanquantile(grip_vals, args.q_low))
    grip_high = float(np.nanquantile(grip_vals, args.q_high))

    # Canonical LingBot layout: xyz + quaternion in 0..6, gripper in 28.
    norm_q01 = q_low.tolist() + [0.0] * 21 + [grip_low] + [0.0]
    norm_q99 = q_high.tolist() + [0.0] * 21 + [grip_high] + [0.0]

    out = {
        "bags": bags,
        "num_bags": len(bags),
        "pose_samples": int(len(pose)),
        "channels": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_normalized"],
        "q_low": args.q_low,
        "q_high": args.q_high,
        "xyzquat_q_low": q_low.tolist(),
        "xyzquat_q_high": q_high.tolist(),
        "xyzquat_min": np.nanmin(pose, axis=0).tolist(),
        "xyzquat_max": np.nanmax(pose, axis=0).tolist(),
        "gripper_source": grip_source,
        "gripper_q_low": grip_low,
        "gripper_q_high": grip_high,
        "lingbot_norm_stat": {"q01": norm_q01, "q99": norm_q99},
        "per_bag": per_bag,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("\nUse the lingbot_norm_stat JSON field from this output in the policy server config you own.")


if __name__ == "__main__":
    main()
