#!/usr/bin/env python3
"""Teacher-forcing execution via ROS — same receiver pipeline as just teleop.

Feeds ground-truth observations from a recorded demo to the policy, sends
predicted joint targets to the arm exactly as the SO leader bridge does.

Each action in the predicted chunk is published sequentially at 50 Hz
(matching ROBOT_FPS), then the next inference is called.

    /arm_joint_target_position      (sensor_msgs/JointState, 6 arm joints)
    /gripper_position_control_host  (signal_arm/gripper_position_control)

Required services:
    just launch roscore
    just launch driver
    just joint-tracker
    just joint-relay
    just policy

Usage (LeRobot v2.1 mode):
    just tf-exec-ros-lerobot /path/to/dataset "prompt" -- --episode 0
    just tf-exec-ros-lerobot /path/to/dataset "prompt" -- --episode 0 --dry-run

Usage (processed-data mode):
    just tf-exec-ros demo_0
    just tf-exec-ros demo_0 -- --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time

import cv2
import numpy as np

# ROS path setup
_A1_SDK = pathlib.Path(__file__).parents[3] / "third_party" / "A1_SDK" / "install"
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openpi_client import websocket_client_policy as _ws_policy

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control

_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "arm_joint6",
]


# ---------------------------------------------------------------------------
# ROS publisher
# ---------------------------------------------------------------------------

class RosPublisher:
    def __init__(self, gripper_scale: float, gripper_offset: float, dry_run: bool):
        self._joint_pub = rospy.Publisher(
            "/arm_joint_target_position", JointState, queue_size=10
        )
        self._gripper_pub = rospy.Publisher(
            "/gripper_position_control_host", gripper_position_control, queue_size=10
        )
        self._gripper_scale = gripper_scale
        self._gripper_offset = gripper_offset
        self.dry_run = dry_run
        time.sleep(0.5)
        print(f"[Pub] ROS publishers ready (dry_run={dry_run})")

    def publish(self, action7: np.ndarray):
        if self.dry_run:
            return
        now = rospy.Time.now()

        js = JointState()
        js.header.stamp = now
        js.name = _JOINT_NAMES
        js.position = [float(v) for v in action7[:6]]
        self._joint_pub.publish(js)

        stroke_mm = float(action7[6]) * self._gripper_scale + self._gripper_offset
        gm = gripper_position_control()
        gm.header.stamp = now
        gm.gripper_stroke = stroke_mm
        self._gripper_pub.publish(gm)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _parse_state7(entry) -> tuple[np.ndarray | None, float]:
    data = entry.get("data", entry)
    joints = data.get("joint", data.get("joints", None))
    if joints is None:
        return None, 0.0
    return np.asarray(joints, dtype=np.float32)[:7], float(entry.get("timestamp", 0.0))


def _read_frame_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _demo_sort_key(path: pathlib.Path):
    for p in path.name.split("_"):
        if p.isdigit():
            return int(p)
    return path.name


# ---------------------------------------------------------------------------
# Execute one chunk: publish each action at publish_hz
# ---------------------------------------------------------------------------

def _exec_chunk(actions, pub, rate, dry_run, chunk_idx, step_mode):
    tag = "DRY" if dry_run else "EXEC"
    print(f"  [{tag}] chunk {chunk_idx}  horizon={actions.shape[0]}  "
          f"actions[0]={np.round(actions[0, :6], 3).tolist()}")

    for action in actions:
        if rospy.is_shutdown():
            raise StopIteration
        t0 = time.monotonic()
        pub.publish(action[:7])
        rate.sleep()

    if step_mode:
        try:
            cmd = input(f"  [Step] chunk {chunk_idx} done. Enter=next, q=quit: ").strip().lower()
        except EOFError:
            cmd = "q"
        if cmd in {"q", "quit", "exit"}:
            raise StopIteration("user quit")


# ---------------------------------------------------------------------------
# LeRobot v2.1 execution loop
# ---------------------------------------------------------------------------

def _load_lerobot_episode(dataset_root, episode_index):
    import pandas as pd
    chunk_ds = episode_index // 1000
    parquet = dataset_root / "data" / f"chunk-{chunk_ds:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet.exists():
        parquet = dataset_root / "data" / f"chunk-{chunk_ds:03d}" / f"file-{chunk_ds:03d}.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"No parquet for episode {episode_index} in {dataset_root}")
        df = pd.read_parquet(parquet)
        df = df[df["episode_index"] == episode_index].reset_index(drop=True)
    else:
        df = pd.read_parquet(parquet)
    return df


def exec_demo_lerobot(*, dataset_root, episode_index, policy, pub, prompt,
                      max_steps, step_mode, publish_hz):
    df = _load_lerobot_episode(dataset_root, episode_index)

    chunk_ds = episode_index // 1000
    images_dir = dataset_root / "images" / f"chunk-{chunk_ds:03d}" / f"episode_{episode_index:06d}"
    cam0_dir = images_dir / "cam0"
    cam1_dir = images_dir / "cam1"
    if not cam0_dir.exists() or not cam1_dir.exists():
        raise FileNotFoundError(f"Image dirs not found: {images_dir}")

    n_steps = len(df) if max_steps <= 0 else min(len(df), max_steps)
    rate = rospy.Rate(publish_hz)
    print(f"[Exec] episode_{episode_index:06d}  steps={n_steps}  "
          f"publish_hz={publish_hz}  step_mode={step_mode}  dry_run={pub.dry_run}")

    records, t, chunk_idx = [], 0, 0
    while t < n_steps and not rospy.is_shutdown():
        row = df.iloc[t]
        frame_idx = int(row["frame_index"])

        img0 = cv2.imread(str(cam0_dir / f"{frame_idx:06d}.jpg"))
        img1 = cv2.imread(str(cam1_dir / f"{frame_idx:06d}.jpg"))
        if img0 is None or img1 is None:
            print(f"[Exec] Missing image at frame {frame_idx}")
            break
        frame0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        state7  = np.asarray(row["observation.state"], dtype=np.float32)[:7]
        target7 = np.asarray(row["action"],            dtype=np.float32)[:7]

        obs = {
            "observation/image":       frame0,
            "observation/wrist_image": frame1,
            "observation/state":       state7,
            "prompt":                  prompt,
        }

        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        horizon = actions.shape[0]
        pred7 = actions[0, :7]

        records.append({
            "step": t, "chunk_idx": chunk_idx,
            "gt_state":    state7.tolist(),
            "gt_target":   target7.tolist(),
            "pred_action": pred7.tolist(),
            "delta_arm":   float(np.linalg.norm(pred7[:6] - target7[:6])),
            "published":   not pub.dry_run,
        })

        try:
            _exec_chunk(actions[:, :7], pub, rate, pub.dry_run, chunk_idx, step_mode)
        except StopIteration:
            print("[Exec] Stopped.")
            break

        t += horizon
        chunk_idx += 1

    print(f"[Exec] Done. {chunk_idx} chunks.")
    return records


# ---------------------------------------------------------------------------
# Processed-data execution loop
# ---------------------------------------------------------------------------

def exec_demo(*, demo_dir, policy, pub, prompt, max_steps, step_mode, publish_hz):
    cmd_states = _load_pickle(demo_dir / "commanded_states.pkl")
    n_valid = len(cmd_states) - 1
    n_steps = n_valid if max_steps <= 0 else min(n_valid, max_steps)

    cap0 = cv2.VideoCapture(str(demo_dir / "cam_0_rgb_video.mp4"))
    cap1 = cv2.VideoCapture(str(demo_dir / "cam_1_rgb_video.mp4"))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"Cannot open videos in {demo_dir}")

    rate = rospy.Rate(publish_hz)
    print(f"[Exec] {demo_dir.name}  steps={n_steps}  "
          f"publish_hz={publish_hz}  step_mode={step_mode}  dry_run={pub.dry_run}")

    records, t, chunk_idx = [], 0, 0
    while t < n_steps and not rospy.is_shutdown():
        frame0 = _read_frame_rgb(cap0)
        frame1 = _read_frame_rgb(cap1)
        if frame0 is None or frame1 is None:
            break

        state7, ts = _parse_state7(cmd_states[t])
        if state7 is None:
            t += 1
            continue

        target7, _ = _parse_state7(cmd_states[min(t + 1, len(cmd_states) - 1)])

        obs = {
            "observation/image":       frame0,
            "observation/wrist_image": frame1,
            "observation/state":       state7,
            "prompt":                  prompt,
        }

        result = policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        horizon = actions.shape[0]
        pred7 = actions[0, :7]

        if target7 is not None:
            records.append({
                "step": t, "chunk_idx": chunk_idx,
                "gt_state":    state7.tolist(),
                "gt_target":   target7.tolist(),
                "pred_action": pred7.tolist(),
                "delta_arm":   float(np.linalg.norm(pred7[:6] - target7[:6])),
                "published":   not pub.dry_run,
            })

        try:
            _exec_chunk(actions[:, :7], pub, rate, pub.dry_run, chunk_idx, step_mode)
        except StopIteration:
            print("[Exec] Stopped.")
            break

        skip = horizon - 1
        for _ in range(skip):
            cap0.read(); cap1.read()

        t += horizon
        chunk_idx += 1

    cap0.release()
    cap1.release()
    print(f"[Exec] Done. {chunk_idx} chunks.")
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root",
                        default=str(ROOT_DIR / "data" / "processed_data" / "pick_twice"))
    parser.add_argument("--demo", default=None)
    parser.add_argument("--lerobot-root", default=None)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   type=int, default=8001)
    parser.add_argument("--prompt", default="swap the position of the marker and the yellow block")
    parser.add_argument("--step-mode", action="store_true", default=True)
    parser.add_argument("--no-step-mode", dest="step_mode", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--output-dir",
                        default=str(ROOT_DIR / "data" / "tf_exec_ros"))
    parser.add_argument("--gripper-scale",  type=float, default=1.0)
    parser.add_argument("--gripper-offset", type=float, default=0.0)
    args = parser.parse_args()

    rospy.init_node("tf_exec_ros", anonymous=False)

    print(f"[Exec] Connecting to policy at ws://{args.host}:{args.port} ...")
    policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[Exec] Connected. Metadata: {policy.get_server_metadata()}")

    pub = RosPublisher(
        gripper_scale=args.gripper_scale,
        gripper_offset=args.gripper_offset,
        dry_run=args.dry_run,
    )

    out_root = pathlib.Path(args.output_dir).expanduser().resolve()

    if args.lerobot_root:
        dataset_root = pathlib.Path(args.lerobot_root).expanduser().resolve()
        if args.episode is not None:
            episodes = [args.episode]
        else:
            parquets = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
            episodes = [int(p.stem.split("_")[1]) for p in parquets if p.stem.split("_")[1].isdigit()]

        for ep_idx in episodes:
            print(f"\n[Exec] === episode_{ep_idx:06d} ===")
            records = exec_demo_lerobot(
                dataset_root=dataset_root, episode_index=ep_idx,
                policy=policy, pub=pub, prompt=args.prompt,
                max_steps=args.max_steps, step_mode=args.step_mode,
                publish_hz=args.publish_hz,
            )
            out_dir = out_root / f"episode_{ep_idx:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "trajectory.json").write_text(json.dumps(records, indent=2))
            print(f"[Exec] Saved: {out_dir}/trajectory.json")

        print(f"\n[Exec] All done. Results: {out_root}")
        return

    data_root = pathlib.Path(args.processed_root).expanduser().resolve()
    if args.demo:
        demo_dirs = [data_root / args.demo]
    else:
        demo_dirs = sorted(
            [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("demo_")],
            key=_demo_sort_key,
        )

    for demo_dir in demo_dirs:
        print(f"\n[Exec] === {demo_dir.name} ===")
        records = exec_demo(
            demo_dir=demo_dir, policy=policy, pub=pub, prompt=args.prompt,
            max_steps=args.max_steps, step_mode=args.step_mode,
            publish_hz=args.publish_hz,
        )
        out_dir = out_root / demo_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trajectory.json").write_text(json.dumps(records, indent=2))
        print(f"[Exec] Saved: {out_dir}/trajectory.json")

    print(f"\n[Exec] All done. Results: {out_root}")


if __name__ == "__main__":
    main()
