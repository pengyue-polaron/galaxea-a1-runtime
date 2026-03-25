#!/usr/bin/env python3
"""Teacher-forcing execution: run policy on recorded demo and send predicted actions to A1 arm.

At each inference step t, feeds ground-truth joint state + camera frames (teacher forcing) to the
running policy server.  The full returned action chunk is published to ZMQ (port 5559) sequentially
at --exec-rate Hz, so the A1 server forwards each action to the real robot.

With --step-mode (default), the script pauses after each chunk and waits for Enter before
proceeding to the next inference call.  This lets you observe the robot's motion one chunk
at a time before committing to the next one.

Requires:
  1. `just policy`                     — WebSocket policy server (port 8000)
  2. `just launch a1-server`           — ZMQ→ROS bridge with policy_action_subscriber
  3. `just joint-relay`                — /arm_joint_target_position → /arm_joint_command_host

Usage (processed-data mode, pkl + mp4):
    just tf-exec demo_0                              # step-mode, 10 Hz per action
    just tf-exec demo_0 -- --exec-rate 5            # slower: 5 Hz per action
    just tf-exec demo_0 -- --no-step-mode           # run all chunks without pausing
    just tf-exec demo_0 -- --dry-run                # infer only, don't send to robot

Usage (LeRobot v2.1 mode, parquet + jpg):
    just tf-exec-lerobot /path/to/dataset "swap the marker and block" -- --episode 0
    just tf-exec-lerobot /path/to/dataset "swap the marker and block" -- --episode 0 --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import zmq

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openpi_client import websocket_client_policy as _ws_policy

ACTION_PORT = 5559
ACTION_HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _parse_state7(entry) -> tuple[np.ndarray | None, float]:
    """Parse commanded_states entry → 7D [j1..j6, gripper_rad] + timestamp."""
    data = entry.get("data", entry)
    joints = data.get("joint", data.get("joints", None))
    if joints is None:
        return None, 0.0
    state7 = np.asarray(joints, dtype=np.float32)[:7]
    ts = float(entry.get("timestamp", 0.0))
    return state7, ts


def _read_frame_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _demo_sort_key(path: Path) -> int | str:
    parts = path.name.split("_")
    for p in parts:
        if p.isdigit():
            return int(p)
    return path.name


# ---------------------------------------------------------------------------
# ZMQ publisher
# ---------------------------------------------------------------------------

class ZmqActionPublisher:
    def __init__(self, host: str = ACTION_HOST, port: int = ACTION_PORT):
        ctx = zmq.Context()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://{host}:{port}")
        # Allow subscribers to connect (ZMQ slow-joiner: give a1-server time to reconnect)
        print(f"[Exec] ZMQ action publisher bound to tcp://{host}:{port} — waiting 2s for subscribers...")
        time.sleep(2.0)
        print("[Exec] Ready.")

    def publish(self, joints7: np.ndarray) -> None:
        """Publish 7D joint target (joints[:6] forwarded by A1 server to robot)."""
        msg = {
            "timestamp": time.time(),
            "joints": [float(j) for j in joints7],
        }
        self._pub.send_json(msg)


# ---------------------------------------------------------------------------
# Chunk execution helper
# ---------------------------------------------------------------------------

def _exec_chunk(
    actions: np.ndarray,          # (horizon, 7) float32
    action_pub: ZmqActionPublisher | None,
    step_dt: float,
    dry_run: bool,
    chunk_idx: int,
    step_mode: bool,
) -> None:
    """Publish each action in the chunk at step_dt intervals, then optionally pause."""
    horizon = actions.shape[0]
    tag = "DRY" if dry_run else "EXEC"

    print(f"  [{tag}] chunk {chunk_idx}  horizon={horizon}  "
          f"actions[0]={np.round(actions[0, :6], 3).tolist()}")

    for i, action in enumerate(actions):
        t_start = time.monotonic()
        if not dry_run and action_pub is not None:
            action_pub.publish(action[:7])
        elapsed = time.monotonic() - t_start
        sleep_t = step_dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    if step_mode:
        try:
            cmd = input(f"  [Step] chunk {chunk_idx} done. Enter=next, q=quit: ").strip().lower()
        except EOFError:
            cmd = "q"
        if cmd in {"q", "quit", "exit"}:
            raise StopIteration("user quit")


# ---------------------------------------------------------------------------
# Execution loop — processed data (pkl + mp4)
# ---------------------------------------------------------------------------

def exec_demo(
    *,
    demo_dir: Path,
    policy,
    action_pub: ZmqActionPublisher | None,
    prompt: str,
    max_steps: int,
    exec_rate_hz: float,
    dry_run: bool,
    step_mode: bool,
) -> list[dict]:
    """Teacher-forcing execution on one processed demo (pkl + mp4).

    Each iteration: read GT obs at step t → infer → execute full chunk → advance t by horizon.
    """
    cmd_states = _load_pickle(demo_dir / "commanded_states.pkl")
    n_valid = len(cmd_states) - 1
    n_steps = n_valid if max_steps <= 0 else min(n_valid, max_steps)

    cap0 = cv2.VideoCapture(str(demo_dir / "cam_0_rgb_video.mp4"))
    cap1 = cv2.VideoCapture(str(demo_dir / "cam_1_rgb_video.mp4"))
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError(f"Cannot open videos in {demo_dir}")

    step_dt = 1.0 / exec_rate_hz if exec_rate_hz > 0 else 0.0

    print(
        f"[Exec] {demo_dir.name}  total_steps={n_steps}  "
        f"exec_rate={exec_rate_hz:.1f}Hz  step_mode={step_mode}  dry_run={dry_run}"
    )
    if step_mode:
        print("[Exec] Step mode: press Enter after each chunk to continue, q to quit.")

    records: list[dict] = []
    t = 0
    chunk_idx = 0

    while t < n_steps:
        # Read frame at current position (cap is already at frame t due to sequential reads)
        frame0 = _read_frame_rgb(cap0)
        frame1 = _read_frame_rgb(cap1)
        if frame0 is None or frame1 is None:
            print(f"[Exec] Video ended early at t={t}")
            break

        state7, timestamp = _parse_state7(cmd_states[t])
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
            actions = actions[np.newaxis, :]  # (1, 7)

        horizon = actions.shape[0]
        pred7_first = actions[0, :7]

        # Record this inference step
        if target7 is not None:
            delta_arm  = float(np.linalg.norm(pred7_first[:6] - target7[:6]))
            delta_full = float(np.linalg.norm(pred7_first - target7))
            delta_grip = float(abs(float(pred7_first[6]) - float(target7[6])))
        else:
            delta_arm = delta_full = delta_grip = 0.0

        records.append({
            "step":          t,
            "chunk_idx":     chunk_idx,
            "timestamp":     float(timestamp),
            "gt_state":      state7.tolist(),
            "gt_target":     target7.tolist() if target7 is not None else [],
            "pred_chunk":    actions[:, :7].tolist(),
            "pred_action":   pred7_first.tolist(),
            "delta_arm":     delta_arm,
            "delta_full7":   delta_full,
            "delta_grip":    delta_grip,
            "published":     not dry_run,
        })

        # Execute chunk (publishes all actions, then optionally pauses)
        try:
            _exec_chunk(
                actions=actions[:, :7],
                action_pub=action_pub,
                step_dt=step_dt,
                dry_run=dry_run,
                chunk_idx=chunk_idx,
                step_mode=step_mode,
            )
        except StopIteration:
            print("[Exec] Stopped by user.")
            break

        # Advance demo by horizon steps (skip frames we already executed)
        skip = horizon - 1  # we already read 1 frame above
        for _ in range(skip):
            cap0.read()
            cap1.read()

        t += horizon
        chunk_idx += 1

    cap0.release()
    cap1.release()
    print(f"[Exec] Done. {chunk_idx} chunks, {len(records)} inference calls.")
    return records


# ---------------------------------------------------------------------------
# Execution loop — LeRobot v2.1 (parquet + jpg)
# ---------------------------------------------------------------------------

def _load_lerobot_episode(dataset_root: Path, episode_index: int):
    """Load one episode from either LeRobot v2.1 (per-episode parquet) or v2.0 (file-NNN.parquet)."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas required: pip install pandas pyarrow")

    chunk_ds = episode_index // 1000

    # v2.1: data/chunk-000/episode_000000.parquet
    parquet_v21 = dataset_root / "data" / f"chunk-{chunk_ds:03d}" / f"episode_{episode_index:06d}.parquet"
    # v2.0: data/chunk-000/file-000.parquet  (all episodes in one file)
    parquet_v20 = dataset_root / "data" / f"chunk-{chunk_ds:03d}" / f"file-{chunk_ds:03d}.parquet"

    if parquet_v21.exists():
        df = pd.read_parquet(parquet_v21)
    elif parquet_v20.exists():
        full = pd.read_parquet(parquet_v20)
        df = full[full["episode_index"] == episode_index].reset_index(drop=True)
        if df.empty:
            raise FileNotFoundError(f"Episode {episode_index} not found in {parquet_v20}")
    else:
        raise FileNotFoundError(f"No parquet found for episode {episode_index} in {dataset_root}/data/chunk-{chunk_ds:03d}/")

    return df


def _read_image_from_row(row, col: str) -> np.ndarray | None:
    """Read image from either a path dict {'path': '...'} or a frame_index + cam_dir approach."""
    val = row[col]
    if isinstance(val, dict):
        path = val.get("path")
        if path:
            img = cv2.imread(str(path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def exec_demo_lerobot(
    *,
    dataset_root: Path,
    episode_index: int,
    policy,
    action_pub: ZmqActionPublisher | None,
    prompt: str,
    max_steps: int,
    exec_rate_hz: float,
    dry_run: bool,
    step_mode: bool,
) -> list[dict]:
    """Teacher-forcing execution on a LeRobot episode (v2.0 or v2.1 format)."""
    df = _load_lerobot_episode(dataset_root, episode_index)

    # Detect image column names
    has_v20_images = "observation.images.cam0" in df.columns
    if not has_v20_images:
        # v2.1: images stored as files in images/ dir
        chunk_ds = episode_index // 1000
        images_dir = dataset_root / "images" / f"chunk-{chunk_ds:03d}" / f"episode_{episode_index:06d}"
        cam0_dir = images_dir / "cam0"
        cam1_dir = images_dir / "cam1"
        if not cam0_dir.exists() or not cam1_dir.exists():
            raise FileNotFoundError(f"Image dirs not found under {images_dir}")
    else:
        cam0_dir = cam1_dir = None

    n_frames = len(df)
    n_steps = n_frames if max_steps <= 0 else min(n_frames, max_steps)
    step_dt = 1.0 / exec_rate_hz if exec_rate_hz > 0 else 0.0

    print(
        f"[Exec] episode_{episode_index:06d}  total_steps={n_steps}  "
        f"exec_rate={exec_rate_hz:.1f}Hz  step_mode={step_mode}  dry_run={dry_run}"
    )
    if step_mode:
        print("[Exec] Step mode: press Enter after each chunk to continue, q to quit.")

    records: list[dict] = []
    t = 0
    chunk_idx = 0

    while t < n_steps:
        row = df.iloc[t]
        frame_idx = int(row["frame_index"])

        if has_v20_images:
            frame0 = _read_image_from_row(row, "observation.images.cam0")
            frame1 = _read_image_from_row(row, "observation.images.cam1")
        else:
            img0 = cv2.imread(str(cam0_dir / f"{frame_idx:06d}.jpg"))
            img1 = cv2.imread(str(cam1_dir / f"{frame_idx:06d}.jpg"))
            frame0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB) if img0 is not None else None
            frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) if img1 is not None else None

        if frame0 is None or frame1 is None:
            print(f"[Exec] Missing image at frame {frame_idx}, stopping.")
            break

        state7  = np.asarray(row["observation.state"], dtype=np.float32)[:7]
        target7 = np.asarray(row["action"],            dtype=np.float32)[:7]
        timestamp = float(row["timestamp"])

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
        pred7_first = actions[0, :7]

        delta_arm  = float(np.linalg.norm(pred7_first[:6] - target7[:6]))
        delta_full = float(np.linalg.norm(pred7_first - target7))
        delta_grip = float(abs(float(pred7_first[6]) - float(target7[6])))

        records.append({
            "step":        t,
            "chunk_idx":   chunk_idx,
            "timestamp":   timestamp,
            "gt_state":    state7.tolist(),
            "gt_target":   target7.tolist(),
            "pred_chunk":  actions[:, :7].tolist(),
            "pred_action": pred7_first.tolist(),
            "delta_arm":   delta_arm,
            "delta_full7": delta_full,
            "delta_grip":  delta_grip,
            "published":   not dry_run,
        })

        try:
            _exec_chunk(
                actions=actions[:, :7],
                action_pub=action_pub,
                step_dt=step_dt,
                dry_run=dry_run,
                chunk_idx=chunk_idx,
                step_mode=step_mode,
            )
        except StopIteration:
            print("[Exec] Stopped by user.")
            break

        t += horizon
        chunk_idx += 1

    print(f"[Exec] Done. {chunk_idx} chunks, {len(records)} inference calls.")
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Teacher-forcing execution: feed GT obs to policy and send actions to A1 arm."
    )
    # --- Processed-data mode (pkl + mp4) ---
    parser.add_argument(
        "--processed-root",
        default=str(ROOT_DIR / "data" / "processed_data" / "pick_twice"),
    )
    parser.add_argument("--demo", default=None, help="Single demo name, e.g. demo_0")
    # --- LeRobot v2.1 mode (parquet + jpg) ---
    parser.add_argument("--lerobot-root", default=None)
    parser.add_argument("--episode", type=int, default=None, help="Episode index for lerobot mode")
    # --- Policy ---
    parser.add_argument("--host",   default="127.0.0.1", help="WebSocket policy server host")
    parser.add_argument("--port",   type=int, default=8001)
    parser.add_argument("--prompt", default="swap the position of the marker and the yellow block")
    # --- Execution ---
    parser.add_argument(
        "--exec-rate", type=float, default=10.0,
        help="Rate (Hz) at which each action in a chunk is sent to the robot. Default: 10 Hz.",
    )
    parser.add_argument(
        "--step-mode", action="store_true", default=True,
        help="Pause after each chunk and wait for Enter before the next inference (default: on).",
    )
    parser.add_argument(
        "--no-step-mode", dest="step_mode", action="store_false",
        help="Run all chunks continuously without pausing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run inference but do NOT publish actions to robot.",
    )
    parser.add_argument(
        "--action-port", type=int, default=ACTION_PORT,
        help=f"ZMQ port to publish joint targets (default: {ACTION_PORT})",
    )
    parser.add_argument(
        "--action-host", default=ACTION_HOST,
        help="ZMQ bind host for action publisher",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 = all steps")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "teacher_forcing_exec"),
    )
    # Legacy compat
    parser.add_argument("--policy-dir",    default=None)
    parser.add_argument("--policy-config", default=None)
    args = parser.parse_args()

    print(f"[Exec] Connecting to WebSocket policy at ws://{args.host}:{args.port} ...")
    policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[Exec] Connected. Server metadata: {policy.get_server_metadata()}")

    action_pub: ZmqActionPublisher | None = None
    if not args.dry_run:
        action_pub = ZmqActionPublisher(host=args.action_host, port=args.action_port)
        print(f"[Exec] Execution ENABLED at {args.exec_rate:.1f} Hz — robot will move!")
    else:
        print("[Exec] DRY RUN — inference only, no commands sent to robot.")

    out_root = Path(args.output_dir).expanduser().resolve()

    # ---- LeRobot mode ----
    if args.lerobot_root:
        dataset_root = Path(args.lerobot_root).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_root}")

        if args.episode is not None:
            episodes = [args.episode]
        else:
            parquet_files = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found under {dataset_root}/data/")
            episodes = [int(pf.stem.split("_")[1]) for pf in parquet_files if pf.stem.split("_")[1].isdigit()]

        for ep_idx in episodes:
            ep_name = f"episode_{ep_idx:06d}"
            print(f"\n[Exec] === {ep_name} ===")
            records = exec_demo_lerobot(
                dataset_root=dataset_root,
                episode_index=ep_idx,
                policy=policy,
                action_pub=action_pub,
                prompt=args.prompt,
                max_steps=args.max_steps,
                exec_rate_hz=args.exec_rate,
                dry_run=args.dry_run,
                step_mode=args.step_mode,
            )
            out_dir = out_root / ep_name
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / "trajectory.json"
            json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"[Exec] Saved JSON: {json_path}")

        print(f"\n[Exec] All done. Results in: {out_root}")
        return

    # ---- Processed-data mode ----
    data_root = Path(args.processed_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    if args.demo:
        demo_dirs = [data_root / args.demo]
    else:
        demo_dirs = sorted(
            [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("demo_")],
            key=_demo_sort_key,
        )

    for demo_dir in demo_dirs:
        if not demo_dir.exists():
            raise FileNotFoundError(f"Demo not found: {demo_dir}")

    for demo_dir in demo_dirs:
        print(f"\n[Exec] === {demo_dir.name} ===")
        records = exec_demo(
            demo_dir=demo_dir,
            policy=policy,
            action_pub=action_pub,
            prompt=args.prompt,
            max_steps=args.max_steps,
            exec_rate_hz=args.exec_rate,
            dry_run=args.dry_run,
            step_mode=args.step_mode,
        )
        out_dir = out_root / demo_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "trajectory.json"
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"[Exec] Saved JSON: {json_path}")

    print(f"\n[Exec] All done. Results in: {out_root}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
