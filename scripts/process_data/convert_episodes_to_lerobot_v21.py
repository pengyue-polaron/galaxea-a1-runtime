#!/usr/bin/env python3
"""Convert A1 teleop episodes to LeRobot v2.1 dataset (7D joint space).

Reads raw episodes produced by `just collect teleop`:
    source_root/episode_YYYYMMDD_HHMMSS/{frames.csv, cam0/, cam1/, metadata.json}

Writes LeRobot v2.1 dataset:
    output_root/
    ├── data/chunk-000/episode_000000.parquet
    ├── meta/{info.json, stats.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}
    └── images/chunk-000/episode_000000/{cam_0,cam_1}/*.jpg

Action definition: action[t] = state[t+1]  (last frame: action[-1] = state[-1])

Usage:
    python scripts/process_data/convert_episodes_to_lerobot_v21.py \\
        --source-root data/a1 --output-root data/a1_lerobot
    python scripts/process_data/convert_episodes_to_lerobot_v21.py \\
        --source-root data/a1 --output-root data/a1_lerobot --task "pick up the block"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V21 = "v2.1"
# Columns in frames.csv that are NOT joint angles
NON_JOINT_COLS = {"frame_index", "wall_time_ns", "ros_stamp_s", "cam0_relpath", "cam1_relpath"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_episode_dirs(source_root: Path, max_episodes: int) -> list[Path]:
    episodes = sorted(p for p in source_root.glob("episode_*") if p.is_dir())
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    return episodes


def resolve_joint_names(df: pd.DataFrame, metadata: dict) -> list[str]:
    """Find joint angle column names in frames.csv."""
    candidate = metadata.get("joint_names")
    if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
        available = [n for n in candidate if n in df.columns]
        if available:
            return available
    return [c for c in df.columns if c not in NON_JOINT_COLS]


def infer_image_shape(sample_path: Path) -> tuple[int, int, int]:
    with Image.open(sample_path) as img:
        w, h = img.size
    return (h, w, 3)


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert A1 teleop episodes to LeRobot v2.1 (7D joint).")
    parser.add_argument("--source-root", type=Path, required=True, help="data/raw/{experiment} directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--fps", type=int, default=30, help="Fallback FPS if metadata missing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = all.")
    parser.add_argument("--disable-cam1", action="store_true",
                        help="Convert cam0-only data (episodes recorded with `just collect1`).")
    args = parser.parse_args()
    cam_map = {"cam0": "cam_0"} if args.disable_cam1 else {"cam0": "cam_0", "cam1": "cam_1"}

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"source-root does not exist: {source_root}")

    # Read task from task.txt (created during collection)
    task_file = source_root / "task.txt"
    if not task_file.exists():
        raise FileNotFoundError(f"No task.txt in {source_root}. Was this created by 'just collect'?")
    args.task = task_file.read_text().strip()
    if not args.task:
        raise ValueError(f"task.txt is empty in {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output-root exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)

    episode_dirs = list_episode_dirs(source_root, args.max_episodes)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* folders under {source_root}.")

    (output_root / "data" / "chunk-000").mkdir(parents=True)
    (output_root / "meta").mkdir(parents=True)

    # Accumulators
    task_to_index: dict[str, int] = {}
    episodes_jsonl: list[dict] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    features: dict | None = None
    total_frames = 0
    global_idx = 0
    ep_idx = 0
    dataset_fps = args.fps

    for episode_dir in tqdm(episode_dirs, desc="converting"):
        csv_path = episode_dir / "frames.csv"
        meta_path = episode_dir / "metadata.json"
        if not csv_path.exists() or not meta_path.exists():
            print(f"  skip {episode_dir.name}: missing frames.csv or metadata.json")
            continue

        with meta_path.open() as f:
            metadata = json.load(f)
        if ep_idx == 0:
            dataset_fps = int(metadata.get("fps_target", args.fps))

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  skip {episode_dir.name}: empty frames.csv")
            continue

        joint_names = resolve_joint_names(df, metadata)
        if not joint_names:
            raise RuntimeError(f"No joint columns found in {csv_path}")

        n_joints = len(joint_names)
        state = df[joint_names].to_numpy(dtype=np.float32)  # (T, n_joints)
        n_frames = len(state)

        # ── Action = next state ──────────────────────────────────────────
        action = np.empty_like(state)
        action[:-1] = state[1:]       # action[t] = state[t+1]
        action[-1] = state[-1]        # last frame repeats

        all_states.append(state)
        all_actions.append(action)

        # ── Timestamps ───────────────────────────────────────────────────
        # Use relative timestamps (seconds since episode start) so they fit in float32.
        # Casting raw Unix epoch (~1.78e9) to float32 has ~128s precision — collapses
        # all frames in an episode to the same value and breaks LeRobot delta_timestamps.
        if "ros_stamp_s" in df.columns:
            ts_full = df["ros_stamp_s"].to_numpy(dtype=np.float64)
            timestamp = (ts_full - ts_full[0]).astype(np.float32)
        elif "wall_time_ns" in df.columns:
            ts_full = df["wall_time_ns"].to_numpy(dtype=np.float64) / 1e9
            timestamp = (ts_full - ts_full[0]).astype(np.float32)
        else:
            timestamp = np.arange(n_frames, dtype=np.float32) / float(dataset_fps)

        frame_index = df["frame_index"].to_numpy(dtype=np.int64) if "frame_index" in df.columns else np.arange(n_frames, dtype=np.int64)
        episode_index = np.full(n_frames, ep_idx, dtype=np.int64)
        dataset_index = np.arange(global_idx, global_idx + n_frames, dtype=np.int64)

        task = str(metadata.get("task", args.task))
        ti = task_to_index.setdefault(task, len(task_to_index))
        task_index = np.full(n_frames, ti, dtype=np.int64)

        # ── Copy images (cam0 → cam_0, cam1 → cam_1 if enabled) ─────────
        img_base = output_root / "images" / "chunk-000" / f"episode_{ep_idx:06d}"
        for src_name, dst_name in cam_map.items():
            src_dir = episode_dir / src_name
            dst_dir = img_base / dst_name
            if src_dir.exists():
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            elif src_name == "cam1" and not args.disable_cam1:
                raise FileNotFoundError(
                    f"{src_dir} missing — episode was recorded without cam1. "
                    f"Re-run with --disable-cam1 (or use `just convert1`)."
                )

        cam_paths = {
            dst_name: [
                str((img_base / dst_name / f"{fi:06d}.jpg").relative_to(output_root))
                for fi in frame_index
            ]
            for dst_name in cam_map.values()
        }

        # ── Build features on first episode ──────────────────────────────
        if features is None:
            features = {
                "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
                "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
                "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
                "index": {"dtype": "int64", "shape": (1,), "names": None},
                "task_index": {"dtype": "int64", "shape": (1,), "names": None},
                "state": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
                "action": {"dtype": "float32", "shape": (n_joints,), "names": joint_names},
            }
            for src_name, dst_name in cam_map.items():
                shape = infer_image_shape(episode_dir / src_name / f"{frame_index[0]:06d}.jpg")
                features[dst_name] = {
                    "dtype": "image",
                    "shape": shape,
                    "names": ["height", "width", "channels"],
                }

        # ── Write parquet ────────────────────────────────────────────────
        df_cols = {
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": episode_index,
            "index": dataset_index,
            "task_index": task_index,
            "state": state.tolist(),
            "action": action.tolist(),
        }
        for dst_name, paths in cam_paths.items():
            df_cols[dst_name] = [{"path": p, "bytes": None} for p in paths]
        episode_df = pd.DataFrame(df_cols)
        parquet_path = output_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        episode_df.to_parquet(parquet_path, index=False)

        episodes_jsonl.append({
            "episode_index": ep_idx,
            "tasks": [task],
            "length": int(n_frames),
            "dataset_from_index": global_idx,
            "dataset_to_index": global_idx + n_frames,
            "data/chunk_index": 0,
            "data/file_index": ep_idx,
        })

        total_frames += n_frames
        global_idx += n_frames
        ep_idx += 1

    if ep_idx == 0:
        raise RuntimeError("No valid episodes converted.")

    # ── Compute stats ────────────────────────────────────────────────────
    all_s = np.concatenate(all_states, axis=0)
    all_a = np.concatenate(all_actions, axis=0)

    def compute_stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
        }

    stats = {
        "state": compute_stats(all_s),
        "action": compute_stats(all_a),
    }
    with (output_root / "meta" / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # ── Write meta/info.json ─────────────────────────────────────────────
    assert features is not None
    info = {
        "codebase_version": V21,
        "robot_type": "a1_single_arm",
        "fps": dataset_fps,
        "features": features,
        "total_episodes": ep_idx,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_chunks": 1,
        "total_videos": 0,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "splits": {"train": f"0:{ep_idx}"},
    }
    with (output_root / "meta" / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    # ── Write meta/tasks.jsonl ───────────────────────────────────────────
    tasks_jsonl = [{"task_index": idx, "task": t} for t, idx in sorted(task_to_index.items(), key=lambda x: x[1])]
    write_jsonl(tasks_jsonl, output_root / "meta" / "tasks.jsonl")
    write_jsonl(episodes_jsonl, output_root / "meta" / "episodes.jsonl")

    print(f"\nDone: {output_root}")
    print(f"  episodes: {ep_idx}  frames: {total_frames}  joints: {features['state']['shape'][0]}")
    print(f"  action definition: action[t] = state[t+1]")


if __name__ == "__main__":
    main()
