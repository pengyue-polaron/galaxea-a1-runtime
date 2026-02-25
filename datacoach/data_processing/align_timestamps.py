import argparse
import shutil
import pickle
import numpy as np
from pathlib import Path
import hydra
import cv2
import sys


def find_start_index(ts, start_ts):
    idx = np.searchsorted(ts, start_ts, side="left")
    if idx > 0 and (
        idx == len(ts)
        or abs(ts[idx - 1] - start_ts) < abs(ts[idx] - start_ts)
    ):
        idx -= 1
    return idx


def trim_video_by_index(
    src_video: Path,
    dst_video: Path,
    start_idx: int,
    num_frames: int,
):
    cap = cv2.VideoCapture(str(src_video))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (width, height))

    cur = 0
    written = 0

    while cap.isOpened() and written < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if cur >= start_idx:
            writer.write(frame)
            written += 1

        cur += 1

    cap.release()
    writer.release()


def align_by_timestamp(ref_ts, src_ts):
    """
    For each ref timestamp, find the nearest src timestamp index.
    Assumes src_ts is sorted.
    """
    indices = []
    j = 0
    for t in ref_ts:
        while (
            j + 1 < len(src_ts)
            and abs(src_ts[j + 1] - t) < abs(src_ts[j] - t)
        ):
            j += 1
        indices.append(j)
    return np.array(indices)

def process_single_demo(raw_demo_dir: Path, save_demo_dir: Path):
    save_demo_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    with open(raw_demo_dir / "states.pkl", "rb") as f:
        states = pickle.load(f)

    with open(raw_demo_dir / "commanded_states.pkl", "rb") as f:
        cmds = pickle.load(f)

    with open(raw_demo_dir / "cam_0_rgb_video.metadata", "rb") as f:
        cam_meta = pickle.load(f)

    cam_ts = np.asarray(cam_meta["timestamps"])
    state_ts = np.asarray([s["timestamp"] for s in states])
    cmd_ts = np.asarray([c["timestamp"] for c in cmds])

    # ---------- Global start ----------
    global_start = max(cam_ts[0], state_ts[0], cmd_ts[0])

    cam_start_idx = find_start_index(cam_ts, global_start)
    
    cam_ts = cam_ts[cam_start_idx:]
    states = states[find_start_index(state_ts, global_start):]
    cmds   = cmds[find_start_index(cmd_ts,   global_start):]

    state_ts = np.asarray([s["timestamp"] for s in states])
    cmd_ts   = np.asarray([c["timestamp"] for c in cmds])

    # ---------- Reference ----------
    ref_ts = cam_ts
    T = len(ref_ts)

    # ---------- Align ----------
    state_idx = align_by_timestamp(ref_ts, state_ts)
    cmd_idx   = align_by_timestamp(ref_ts, cmd_ts)

    aligned_states = [states[i] for i in state_idx]
    aligned_cmds   = [cmds[i]   for i in cmd_idx]

    # ---------- Save aligned states ----------
    with open(save_demo_dir / "states.pkl", "wb") as f:
        pickle.dump(aligned_states, f)

    with open(save_demo_dir / "commanded_states.pkl", "wb") as f:
        pickle.dump(aligned_cmds, f)

    # ---------- Trim + save video ----------
    trim_video_by_index(
        raw_demo_dir / "cam_0_rgb_video.mp4",
        save_demo_dir / "cam_0_rgb_video.mp4",
        start_idx=cam_start_idx,
        num_frames=T,
    )

    # ---------- Save trimmed metadata ----------
    cam_meta["timestamps"] = cam_ts.tolist()
    cam_meta["num_image_frames"] = T
    cam_meta["record_start_time"] = cam_ts[0]
    cam_meta["record_end_time"] = cam_ts[-1]
    cam_meta["filename"] = str(save_demo_dir / "cam_0_rgb_video.mp4")
    print(str(save_demo_dir / "cam_0_rgb_video.mp4"))

    with open(save_demo_dir / "cam_0_rgb_video.metadata", "wb") as f:
        pickle.dump(cam_meta, f)

    print(f"✅ {raw_demo_dir.name} | frames={T}")