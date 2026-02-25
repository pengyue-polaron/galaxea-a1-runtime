import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def encode_state(state_dict):
    d = state_dict["data"]
    pos = list(d.get("pos", [0.0, 0.0, 0.0]))
    ori = list(d.get("ori", [0.0, 0.0, 0.0, 1.0]))
    gripper = d.get("gripper", 0.0)
    if gripper is None:
        gripper = 0.0
    return np.asarray(pos + ori + [float(gripper)], dtype=np.float32)


def _discover_episode_dirs(processed_root: Path):
    return sorted([d for d in processed_root.iterdir() if d.is_dir()])


def _discover_cameras(episode_dir: Path, preferred_cameras=None):
    available = []
    for p in sorted(episode_dir.glob("*_rgb_video.mp4")):
        cam_id = p.name.replace("_rgb_video.mp4", "")
        available.append(cam_id)

    if preferred_cameras:
        ordered = [cam for cam in preferred_cameras if cam in available]
        tail = [cam for cam in available if cam not in ordered]
        return ordered + tail
    return available


def _detect_camera_shapes(episode_dir: Path, cameras: list[str]):
    shapes = {}
    for cam in cameras:
        video_path = episode_dir / f"{cam}_rgb_video.mp4"
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read first frame from {video_path}")
        h, w = frame.shape[:2]
        shapes[cam] = (h, w)
    return shapes


def convert_a1_to_lerobot(cfg):
    output_path = Path(cfg.formatted_data_path) / cfg.task_name
    if output_path.exists():
        try:
            input(
                f"⚠️ Dataset already exists at:\n"
                f"  {output_path}\n\n"
                f"Press ENTER to REMOVE it.\n"
                f"Press Ctrl+C to ABORT.\n"
            )
        except KeyboardInterrupt:
            print("\n❌ Aborted. Dataset not removed.")
            return

        import shutil

        shutil.rmtree(output_path)
        print("🗑️ Existing dataset removed.")

    raw_data_root = Path(cfg.processed_data_path) / cfg.task_name
    os.makedirs(raw_data_root, exist_ok=True)
    demo_dirs = _discover_episode_dirs(raw_data_root)
    if not demo_dirs:
        raise RuntimeError(f"No processed demos under {raw_data_root}")

    configured_cameras = list(getattr(cfg, "cameras", [])) or None
    cameras = _discover_cameras(demo_dirs[0], preferred_cameras=configured_cameras)
    if not cameras:
        raise RuntimeError(f"No camera videos found under {demo_dirs[0]}")

    cam_shapes = _detect_camera_shapes(demo_dirs[0], cameras)

    features = {
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["action"],
        },
    }
    for cam in cameras:
        h, w = cam_shapes[cam]
        features[cam] = {
            "dtype": cfg.mode,
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id="local_data",
        fps=cfg.fps,
        features=features,
        robot_type="A1",
        root=output_path,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
        video_backend="ffmpeg",
    )

    print(f"Using cameras: {cameras}")
    for episode in demo_dirs:
        print(f"▶ Processing {episode.name}")
        with open(episode / "states.pkl", "rb") as f:
            states = pickle.load(f)
        with open(episode / "commanded_states.pkl", "rb") as f:
            actions = pickle.load(f)

        if len(states) != len(actions):
            raise RuntimeError(f"State/action length mismatch in {episode}")
        T = len(states)
        if T == 0:
            raise RuntimeError(f"Empty states/actions in {episode}")

        episode_cameras = _discover_cameras(episode, preferred_cameras=cameras)
        if episode_cameras != cameras:
            raise RuntimeError(
                f"Camera mismatch in {episode}. expected={cameras}, found={episode_cameras}"
            )

        caps = {cam: cv2.VideoCapture(str(episode / f"{cam}_rgb_video.mp4")) for cam in cameras}
        for cam, cap in caps.items():
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video for {cam} in {episode}")

        for t in range(T):
            frame = {}
            for cam in cameras:
                ret, image = caps[cam].read()
                if not ret:
                    raise RuntimeError(f"Video {cam} ended early at t={t} in {episode}")
                frame[cam] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frame["state"] = encode_state(states[t])
            frame["action"] = encode_state(actions[t])
            frame["task"] = cfg.task_name
            dataset.add_frame(frame)

        for cap in caps.values():
            cap.release()

        dataset.save_episode()
        print(f"✅ Saved episode {episode.name}")

    print("✅ Conversion finished.")
