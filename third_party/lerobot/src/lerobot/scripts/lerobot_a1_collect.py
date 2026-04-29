#!/usr/bin/env python
"""Multi-episode A1 teleop data collection.

Data layout:
    data/raw/{experiment}/
        task.txt                          # task prompt (created once)
        episode_000_20260418_120000/      # episode directories
        episode_001_20260418_120030/
        ...

Usage:
    just collect pick_block
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import numpy as np

# ROS Python path setup
import pathlib as _pathlib
_A1_SDK = _pathlib.Path(__file__).parents[4] / "A1_SDK" / "install"
for _p in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class JointSnapshot:
    ros_stamp_s: float
    names: list[str]
    positions: list[float]


class JointStateCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._msg: Any | None = None

    def callback(self, msg: Any) -> None:
        with self._lock:
            self._msg = msg

    def snapshot(self) -> JointSnapshot | None:
        with self._lock:
            msg = self._msg
        if msg is None:
            return None
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        if not names or not positions:
            return None
        n = min(len(names), len(positions))
        h = getattr(msg, "header", None)
        s = getattr(h, "stamp", None) if h else None
        fn = getattr(s, "to_sec", None) if s else None
        if callable(fn):
            try:
                stamp = float(fn())
            except Exception:
                stamp = 0.0
        elif s:
            stamp = float(getattr(s, "secs", 0)) + float(getattr(s, "nsecs", 0)) / 1e9
        else:
            stamp = 0.0
        return JointSnapshot(ros_stamp_s=stamp, names=names[:n], positions=positions[:n])


def _open_realsense(serial, w, h, fps):
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("\n  ERROR: No RealSense camera detected.")
        print("  Check: is the USB cable plugged in? Try a different port.")
        print("  Run 'lsusb | grep -i intel' to verify.\n")
        sys.exit(1)
    pipe = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    try:
        pipe.start(cfg)
    except RuntimeError as e:
        print(f"\n  ERROR: RealSense failed to start: {e}")
        print("  Try unplugging and replugging the camera.\n")
        sys.exit(1)
    return pipe


def _video_device_name(idx: int) -> str:
    p = Path(f"/sys/class/video4linux/video{idx}/name")
    try:
        return p.read_text().strip() if p.exists() else "unknown"
    except Exception:
        return "unknown"


def _open_cam1_auto(index_or_auto: str, w: int, h: int, fps: int):
    if index_or_auto.strip().lower() != "auto":
        idx = int(index_or_auto)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open cam1 index={idx}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap, idx

    for idx in range(13):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if not ok:
            cap.release()
            continue
        name = _video_device_name(idx)
        if "realsense" in name.lower() or "intel" in name.lower():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        return cap, idx

    raise RuntimeError("No suitable cam1 found")


# ---------------------------------------------------------------------------
# Record one episode
# ---------------------------------------------------------------------------

def record_episode(episode_dir, rs_pipe, cam1, joint_cache, fps, max_dur, jpeg_q):
    """Returns (frame_count, joint_names, discard).

    When cam1 is None, only cam0 (RealSense) is recorded.
    """
    (episode_dir / "cam0").mkdir(parents=True)
    if cam1 is not None:
        (episode_dir / "cam1").mkdir(parents=True)

    stop = threading.Event()
    user_input = [None]

    def _wait():
        try:
            line = sys.stdin.readline().strip().lower()
        except (EOFError, OSError):
            line = ""
        user_input[0] = line
        stop.set()

    threading.Thread(target=_wait, daemon=True).start()

    idx = 0
    names = None
    t0 = time.perf_counter()
    period = 1.0 / fps
    jp = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]

    with (episode_dir / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        while not stop.is_set():
            lt = time.perf_counter()
            if max_dur > 0 and lt - t0 >= max_dur:
                break

            try:
                fr = rs_pipe.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                continue
            cf = fr.get_color_frame()
            if not cf:
                continue
            img0 = np.asanyarray(cf.get_data())

            if cam1 is not None:
                ok, img1 = cam1.read()
                if not ok:
                    continue

            snap = joint_cache.snapshot()
            if snap is None:
                continue

            if names is None:
                names = snap.names
                header = ["frame_index", "wall_time_ns", "ros_stamp_s", "cam0_relpath"]
                if cam1 is not None:
                    header.append("cam1_relpath")
                header.extend(names)
                w.writerow(header)

            fn = f"{idx:06d}.jpg"
            cv2.imwrite(str(episode_dir / "cam0" / fn), img0, jp)
            if cam1 is not None:
                cv2.imwrite(str(episode_dir / "cam1" / fn), img1, jp)

            pm = dict(zip(snap.names, snap.positions))
            row = [idx, time.time_ns(), f"{snap.ros_stamp_s:.9f}", f"cam0/{fn}"]
            if cam1 is not None:
                row.append(f"cam1/{fn}")
            row.extend(pm.get(n, "") for n in names)
            w.writerow(row)
            if idx % 30 == 0:
                f.flush()
            idx += 1

            sl = period - (time.perf_counter() - lt)
            if sl > 0:
                time.sleep(sl)

    discard = user_input[0] in ("d", "discard")
    return idx, names, discard


# ---------------------------------------------------------------------------
# Task prompt management
# ---------------------------------------------------------------------------

def load_or_prompt_task(experiment_dir: Path) -> str:
    """Load task from task.txt, or prompt user to enter it (first time only)."""
    task_file = experiment_dir / "task.txt"
    if task_file.exists():
        task = task_file.read_text().strip()
        if task:
            return task

    print(f"  [first run] Enter task prompt for this experiment:")
    task = input("  > ").strip()
    if not task:
        print("  ERROR: task prompt cannot be empty.")
        sys.exit(1)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    task_file.write_text(task + "\n")
    return task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-duration-s", type=float, default=0.0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--joint-topic", default="/joint_states_host")
    p.add_argument("--joint-wait-timeout-s", type=float, default=10.0)
    p.add_argument("--cam0-serial", default=None)
    p.add_argument("--cam0-width", type=int, default=640)
    p.add_argument("--cam0-height", type=int, default=480)
    p.add_argument("--cam0-fps", type=int, default=30)
    p.add_argument("--cam1-index", default="auto")
    p.add_argument("--cam1-width", type=int, default=640)
    p.add_argument("--cam1-height", type=int, default=480)
    p.add_argument("--cam1-fps", type=int, default=30)
    p.add_argument("--disable-cam1", action="store_true",
                   help="Skip wrist camera (cam1); record only RealSense (cam0).")
    args = p.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    exp_dir = data_root / args.experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Task prompt (saved to task.txt, asked only once) ─────────────────
    task = load_or_prompt_task(exp_dir)

    # ── Init ROS + cameras ───────────────────────────────────────────────
    print("[collect] importing rospy ...", flush=True)
    import rospy
    from sensor_msgs.msg import JointState

    print("[collect] rospy.init_node ...", flush=True)
    rospy.init_node("a1_collect", anonymous=False, disable_signals=True)
    print("[collect] subscribing to joint topic ...", flush=True)
    jc = JointStateCache()
    rospy.Subscriber(args.joint_topic, JointState, jc.callback, queue_size=10)

    print("[collect] waiting for joint state ...", end=" ", flush=True)
    dl = time.time() + args.joint_wait_timeout_s
    while jc.snapshot() is None:
        if time.time() > dl:
            print("TIMEOUT")
            sys.exit(1)
        time.sleep(0.05)
    print("ok")

    print("[collect] opening cameras ...", end=" ", flush=True)
    rs_pipe = _open_realsense(args.cam0_serial, args.cam0_width, args.cam0_height, args.cam0_fps)
    if args.disable_cam1:
        cam1, cam1_idx = None, None
        print("ok (cam1=DISABLED)")
    else:
        cam1, cam1_idx = _open_cam1_auto(args.cam1_index, args.cam1_width, args.cam1_height, args.cam1_fps)
        print(f"ok (cam1=video{cam1_idx})")

    # ── Count existing episodes ──────────────────────────────────────────
    existing = sorted(d for d in exp_dir.glob("episode_*") if d.is_dir())
    ep = len(existing)

    print(f"\n  experiment : {args.experiment}")
    print(f"  task       : {task}")
    print(f"  output     : {exp_dir}")
    print(f"  next episode: {ep}")
    print(f"  Ctrl+C to quit\n")

    # ── Episode loop ─────────────────────────────────────────────────────
    try:
        while True:
            input(f"  [{ep}] press Enter to START recording ...")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"episode_{ep:03d}_{ts}"
            edir = exp_dir / name

            print(f"  [{ep}] recording ... Enter=save, d+Enter=discard")

            try:
                n_frames, joint_names, discard = record_episode(
                    edir, rs_pipe, cam1, jc,
                    args.fps, args.max_duration_s, args.jpeg_quality,
                )
            except RuntimeError as e:
                print(f"  [{ep}] ERROR: {e}")
                discard = True
                n_frames = 0

            if discard or n_frames == 0:
                import shutil
                shutil.rmtree(edir, ignore_errors=True)
                reason = "user discarded" if discard else "0 frames"
                print(f"  [{ep}] {reason}, episode deleted.\n")
                continue

            # Save metadata
            meta = {
                "task": task,
                "experiment": args.experiment,
                "episode_index": ep,
                "frame_count": n_frames,
                "fps_target": args.fps,
                "joint_names": joint_names or [],
                "cam0": {"serial": args.cam0_serial, "width": args.cam0_width, "height": args.cam0_height},
            }
            if cam1 is not None:
                meta["cam1"] = {"index": cam1_idx, "width": args.cam1_width, "height": args.cam1_height}
            with (edir / "metadata.json").open("w") as f:
                json.dump(meta, f, indent=2)

            print(f"  [{ep}] saved {n_frames} frames -> {name}\n")
            ep += 1

    except (KeyboardInterrupt, EOFError):
        print(f"\n[collect] done. {ep} episodes for '{args.experiment}'.")
    finally:
        if cam1 is not None:
            cam1.release()
        rs_pipe.stop()


if __name__ == "__main__":
    main()
