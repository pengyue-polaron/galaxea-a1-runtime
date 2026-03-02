#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
import time
from pathlib import Path


def _extend_ros_python_paths():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
        str(repo_root / "third_party" / "A1_SDK" / "install" / "lib" / "python3" / "dist-packages"),
    ]
    a1_sdk_root = os.environ.get("A1_SDK_ROOT")
    if a1_sdk_root:
        candidates.append(str(Path(a1_sdk_root) / "install" / "lib" / "python3" / "dist-packages"))
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


_extend_ros_python_paths()

try:
    import rospy
    from geometry_msgs.msg import PoseStamped
    from signal_arm.msg import gripper_position_control
    from std_msgs.msg import Header
except Exception as exc:
    print(
        f"ROS import failed: {exc}\n"
        "Tip: source /opt/ros/noetic/setup.bash and third_party/A1_SDK/install/setup.bash first.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _is_finite_seq(values, expected_len: int) -> bool:
    if not isinstance(values, (list, tuple)) or len(values) != expected_len:
        return False
    for v in values:
        try:
            fv = float(v)
        except Exception:
            return False
        if not math.isfinite(fv):
            return False
    return True


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _load_csv(path: Path):
    points = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = ("x", "y", "z", "qx", "qy", "qz", "qw")
        for col in required:
            if col not in reader.fieldnames:
                raise RuntimeError(f"CSV missing required column: {col}")

        for row in reader:
            try:
                pos = [float(row["x"]), float(row["y"]), float(row["z"])]
                ori = [float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])]
            except Exception:
                continue
            if not _is_finite_seq(pos, 3) or not _is_finite_seq(ori, 4):
                continue

            ts = None
            if "timestamp" in row and row["timestamp"] not in (None, ""):
                try:
                    ts = float(row["timestamp"])
                except Exception:
                    ts = None

            grip = None
            if "gripper" in row and row["gripper"] not in (None, ""):
                try:
                    grip = float(row["gripper"])
                except Exception:
                    grip = None

            points.append({"timestamp": ts, "pos": pos, "ori": ori, "gripper": grip})
    return points


def _load_pkl(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise RuntimeError("PKL must be a list")

    points = []
    for item in data:
        if not isinstance(item, dict):
            continue

        ts = item.get("timestamp")
        if isinstance(ts, (int, float)):
            ts = float(ts)
        else:
            ts = None

        payload = item.get("data") if isinstance(item.get("data"), dict) else item
        pos = payload.get("pos")
        ori = payload.get("ori")
        grip = payload.get("gripper")

        if not _is_finite_seq(pos, 3) or not _is_finite_seq(ori, 4):
            continue

        if grip is not None:
            try:
                grip = float(grip)
            except Exception:
                grip = None

        points.append(
            {
                "timestamp": ts,
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "ori": [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])],
                "gripper": grip,
            }
        )
    return points


def _normalize_quat(q):
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(v) / norm for v in q]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay inferred EEF trajectory to A1 ROS topics."
    )
    parser.add_argument("--input", required=True, help="Path to eef_actions.csv or states.pkl")
    parser.add_argument(
        "--source",
        choices=("auto", "csv", "pkl"),
        default="auto",
        help="Input format (default: auto by file suffix)",
    )
    parser.add_argument("--pose-topic", default="/a1_ee_target")
    parser.add_argument("--gripper-topic", default="/gripper_position_control_host")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument(
        "--rate",
        type=float,
        default=15.0,
        help="Fallback fixed publish rate (Hz) when timestamps unavailable",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (>1 faster, <1 slower)",
    )
    parser.add_argument(
        "--ignore-timestamps",
        action="store_true",
        help="Ignore per-point timestamps and use fixed --rate",
    )
    parser.add_argument("--loop", action="store_true", help="Loop replay until Ctrl+C")
    parser.add_argument("--no-gripper", action="store_true", help="Do not publish gripper command")
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=60.0)
    parser.add_argument("--startup-delay", type=float, default=0.2)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(rospy.myargv()[1:])


def _sleep_until(target_monotonic: float):
    while True:
        now = time.monotonic()
        dt = target_monotonic - now
        if dt <= 0:
            return
        time.sleep(min(dt, 0.01))


def _has_valid_timestamps(points) -> bool:
    if len(points) < 2:
        return False
    ts = [p["timestamp"] for p in points]
    if any(v is None for v in ts):
        return False
    for i in range(1, len(ts)):
        if float(ts[i]) <= float(ts[i - 1]):
            return False
    return True


def _load_points(input_path: Path, source: str):
    use_source = source
    if use_source == "auto":
        if input_path.suffix.lower() == ".csv":
            use_source = "csv"
        elif input_path.suffix.lower() in (".pkl", ".pickle"):
            use_source = "pkl"
        else:
            raise RuntimeError(f"Cannot auto-detect format from suffix: {input_path.suffix}")

    if use_source == "csv":
        return _load_csv(input_path)
    if use_source == "pkl":
        return _load_pkl(input_path)
    raise RuntimeError(f"Unsupported source: {use_source}")


def _publish_point(pose_pub, grip_pub, frame_id: str, point, enable_gripper: bool, grip_min: float, grip_max: float):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = float(point["pos"][0])
    pose.pose.position.y = float(point["pos"][1])
    pose.pose.position.z = float(point["pos"][2])
    qx, qy, qz, qw = _normalize_quat(point["ori"])
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    pose_pub.publish(pose)

    if enable_gripper and point["gripper"] is not None:
        gmsg = gripper_position_control()
        gmsg.header = Header()
        gmsg.header.stamp = rospy.Time.now()
        gmsg.gripper_stroke = _clamp(float(point["gripper"]), grip_min, grip_max)
        grip_pub.publish(gmsg)


def main():
    args = parse_args()
    if args.rate <= 0:
        raise RuntimeError("--rate must be > 0")
    if args.speed <= 0:
        raise RuntimeError("--speed must be > 0")
    if args.gripper_min > args.gripper_max:
        raise RuntimeError("--gripper-min must be <= --gripper-max")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    points = _load_points(input_path, args.source)
    if not points:
        raise RuntimeError(f"No valid points loaded from {input_path}")

    use_ts = (not args.ignore_timestamps) and _has_valid_timestamps(points)
    duration_s = 0.0
    if use_ts:
        duration_s = float(points[-1]["timestamp"]) - float(points[0]["timestamp"])

    rospy.init_node("replay_inferred_trajectory", anonymous=True)
    pose_pub = rospy.Publisher(args.pose_topic, PoseStamped, queue_size=10)
    grip_pub = rospy.Publisher(args.gripper_topic, gripper_position_control, queue_size=10)

    if args.startup_delay > 0:
        time.sleep(args.startup_delay)

    print(f"[ReplayInfer] Input: {input_path}")
    print(f"[ReplayInfer] Points: {len(points)}")
    print(
        f"[ReplayInfer] Mode: {'timestamped' if use_ts else 'fixed-rate'} "
        f"(speed={args.speed}, loop={args.loop})"
    )
    if use_ts:
        print(f"[ReplayInfer] Source duration: {duration_s:.3f}s")
    else:
        print(f"[ReplayInfer] Fixed rate: {args.rate:.2f} Hz")
    print(f"[ReplayInfer] Pose topic: {args.pose_topic}")
    if args.no_gripper:
        print("[ReplayInfer] Gripper publish: disabled")
    else:
        print(f"[ReplayInfer] Gripper topic: {args.gripper_topic}")

    round_idx = 0
    fixed_rate = rospy.Rate(max(args.rate * args.speed, 1e-3))
    while not rospy.is_shutdown():
        round_idx += 1
        print(f"[ReplayInfer] Start round #{round_idx}")

        start_monotonic = time.monotonic()
        t0 = float(points[0]["timestamp"]) if use_ts else None
        for i, point in enumerate(points):
            if rospy.is_shutdown():
                break

            if use_ts:
                rel_t = (float(point["timestamp"]) - t0) / args.speed
                _sleep_until(start_monotonic + rel_t)
            else:
                if i > 0:
                    fixed_rate.sleep()

            _publish_point(
                pose_pub=pose_pub,
                grip_pub=grip_pub,
                frame_id=args.frame_id,
                point=point,
                enable_gripper=not args.no_gripper,
                grip_min=args.gripper_min,
                grip_max=args.gripper_max,
            )

            if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or (i + 1) == len(points)):
                print(f"[ReplayInfer] round={round_idx} progress={i + 1}/{len(points)}")

        if not args.loop:
            break

    print("[ReplayInfer] Done")


if __name__ == "__main__":
    main()
