import pickle
from pathlib import Path

import cv2
import numpy as np


def find_start_index(ts, start_ts):
    idx = np.searchsorted(ts, start_ts, side="left")
    if idx > 0 and (idx == len(ts) or abs(ts[idx - 1] - start_ts) < abs(ts[idx] - start_ts)):
        idx -= 1
    return idx


def align_by_timestamp(ref_ts, src_ts):
    """For each ref timestamp, find the nearest src timestamp index."""
    indices = []
    j = 0
    for t in ref_ts:
        while j + 1 < len(src_ts) and abs(src_ts[j + 1] - t) < abs(src_ts[j] - t):
            j += 1
        indices.append(j)
    return np.asarray(indices, dtype=np.int64)


def _trim_to_range(ts, start_ts, end_ts):
    start_idx = find_start_index(ts, start_ts)
    end_idx = np.searchsorted(ts, end_ts, side="right")
    return start_idx, end_idx, ts[start_idx:end_idx]


def _discover_camera_ids(raw_demo_dir: Path, preferred_cameras=None):
    discovered = []
    for p in sorted(raw_demo_dir.glob("*_rgb_video.metadata")):
        cam_id = p.name.replace("_rgb_video.metadata", "")
        if (raw_demo_dir / f"{cam_id}_rgb_video.mp4").exists():
            discovered.append(cam_id)

    if preferred_cameras:
        ordered = [cam for cam in preferred_cameras if cam in discovered]
        tail = [cam for cam in discovered if cam not in ordered]
        return ordered + tail
    return discovered


def _read_frame_by_indices(src_video: Path, dst_video: Path, indices: np.ndarray):
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {src_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size for {src_video}")

    writer = cv2.VideoWriter(
        str(dst_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )

    cur = 0
    ptr = 0
    total = len(indices)

    while cap.isOpened() and ptr < total:
        ok, frame = cap.read()
        if not ok:
            break

        while ptr < total and indices[ptr] == cur:
            writer.write(frame)
            ptr += 1
        cur += 1

    cap.release()
    writer.release()

    if ptr != total:
        raise RuntimeError(
            f"Failed to sample enough frames from {src_video}: requested={total}, written={ptr}."
        )


def process_single_demo(
    raw_demo_dir: Path,
    save_demo_dir: Path,
    reference_camera: str = "cam_0",
    camera_ids=None,
):
    save_demo_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_demo_dir / "states.pkl", "rb") as f:
        states = pickle.load(f)
    with open(raw_demo_dir / "commanded_states.pkl", "rb") as f:
        cmds = pickle.load(f)

    if not states or not cmds:
        raise RuntimeError(f"{raw_demo_dir}: empty states/cmds.")

    discovered_cameras = _discover_camera_ids(raw_demo_dir, preferred_cameras=camera_ids)
    if not discovered_cameras:
        raise RuntimeError(f"{raw_demo_dir}: no camera metadata/video found.")

    camera_data = {}
    for cam_id in discovered_cameras:
        with open(raw_demo_dir / f"{cam_id}_rgb_video.metadata", "rb") as f:
            meta = pickle.load(f)
        ts = np.asarray(meta["timestamps"], dtype=np.float64)
        if len(ts) == 0:
            raise RuntimeError(f"{raw_demo_dir}: camera {cam_id} has empty timestamps.")
        camera_data[cam_id] = {"meta": meta, "ts": ts}

    if reference_camera not in camera_data:
        reference_camera = discovered_cameras[0]

    state_ts = np.asarray([float(s["timestamp"]) for s in states], dtype=np.float64)
    cmd_ts = np.asarray([float(c["timestamp"]) for c in cmds], dtype=np.float64)

    global_start = max(
        [state_ts[0], cmd_ts[0]] + [camera_data[cam]["ts"][0] for cam in discovered_cameras]
    )
    global_end = min(
        [state_ts[-1], cmd_ts[-1]] + [camera_data[cam]["ts"][-1] for cam in discovered_cameras]
    )
    if global_end <= global_start:
        raise RuntimeError(f"{raw_demo_dir}: no valid overlap among streams.")

    state_start, state_end, state_trim_ts = _trim_to_range(state_ts, global_start, global_end)
    cmd_start, cmd_end, cmd_trim_ts = _trim_to_range(cmd_ts, global_start, global_end)
    states_trim = states[state_start:state_end]
    cmds_trim = cmds[cmd_start:cmd_end]

    if len(state_trim_ts) == 0 or len(cmd_trim_ts) == 0:
        raise RuntimeError(f"{raw_demo_dir}: empty trimmed state/cmd streams.")

    ref_ts_full = camera_data[reference_camera]["ts"]
    ref_start, ref_end, ref_ts = _trim_to_range(ref_ts_full, global_start, global_end)
    if len(ref_ts) == 0:
        raise RuntimeError(f"{raw_demo_dir}: empty reference camera stream ({reference_camera}).")

    state_idx = align_by_timestamp(ref_ts, state_trim_ts)
    cmd_idx = align_by_timestamp(ref_ts, cmd_trim_ts)
    aligned_states = [states_trim[i] for i in state_idx]
    aligned_cmds = [cmds_trim[i] for i in cmd_idx]

    with open(save_demo_dir / "states.pkl", "wb") as f:
        pickle.dump(aligned_states, f)
    with open(save_demo_dir / "commanded_states.pkl", "wb") as f:
        pickle.dump(aligned_cmds, f)

    # Also export a concise trajectory artifact for downstream checks/inspection.
    with open(save_demo_dir / "trajectory.csv", "w", encoding="utf-8") as f:
        f.write("t,x,y,z,qx,qy,qz,qw,gripper\n")
        for item in aligned_cmds:
            t = float(item["timestamp"])
            d = item["data"]
            pos = d["pos"]
            ori = d["ori"]
            gripper = float(d["gripper"])
            f.write(
                f"{t:.9f},{pos[0]:.9f},{pos[1]:.9f},{pos[2]:.9f},"
                f"{ori[0]:.9f},{ori[1]:.9f},{ori[2]:.9f},{ori[3]:.9f},{gripper:.9f}\n"
            )

    T = len(ref_ts)
    for cam_id in discovered_cameras:
        src_video = raw_demo_dir / f"{cam_id}_rgb_video.mp4"
        dst_video = save_demo_dir / f"{cam_id}_rgb_video.mp4"

        cam_ts = camera_data[cam_id]["ts"]
        cam_start, _, cam_trim_ts = _trim_to_range(cam_ts, global_start, global_end)
        cam_local_idx = align_by_timestamp(ref_ts, cam_trim_ts)
        cam_abs_idx = cam_local_idx + cam_start

        _read_frame_by_indices(src_video, dst_video, cam_abs_idx)
        selected_ts = cam_ts[cam_abs_idx]

        cam_meta = dict(camera_data[cam_id]["meta"])
        cam_meta["timestamps"] = selected_ts.tolist()
        cam_meta["num_image_frames"] = T
        cam_meta["record_start_time"] = float(selected_ts[0])
        cam_meta["record_end_time"] = float(selected_ts[-1])
        cam_meta["filename"] = str(dst_video)
        cam_meta["reference_camera"] = reference_camera
        with open(save_demo_dir / f"{cam_id}_rgb_video.metadata", "wb") as f:
            pickle.dump(cam_meta, f)

    print(f"✅ {raw_demo_dir.name} | ref={reference_camera} | frames={T} | cameras={discovered_cameras}")
