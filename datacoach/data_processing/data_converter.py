from pathlib import Path
import pickle
import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
import lerobot.datasets.lerobot_dataset as lr

from pathlib import Path
import os

import hydra

def encode_state(state_dict):
    d = state_dict["data"]
    return np.array(
        d["pos"] + d["ori"] + [d["gripper"]],
        dtype=np.float32
    )


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

    features={
            "state": {
                "dtype": "float32",
                "shape": (8,),  # 自动从 numpy 推断
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
        },
        
    }
    
    
    cameras = [
        "cam_0",
    ]
    for cam in cameras:
        features[cam] = {
            "dtype": cfg.mode,
            "shape": (3, 480, 640),  # (channels, height, width)
            "names": [
                "channels","height","width",
            ],
        }
        
    # create LeRobotDataset)
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
        
  
    RAW_DATA_ROOT = Path(cfg.processed_data_path)/ cfg.task_name
    os.makedirs(RAW_DATA_ROOT, exist_ok=True)
    demo_dirs = sorted([d for d in RAW_DATA_ROOT.iterdir() if d.is_dir()])
    
    for episode in demo_dirs:
        print(f"▶ Processing {episode.name}")

        with open(episode / "states.pkl", "rb") as f:
            states = pickle.load(f)

        with open(episode / "commanded_states.pkl", "rb") as f:
            actions = pickle.load(f)

        assert len(states) == len(actions), "State/action length mismatch"

        T = len(states)
        
        # 先打开所有 cameras
        caps = {cam: cv2.VideoCapture(str(episode / f"{cam}_rgb_video.mp4")) for cam in cameras}

        for t in range(T):
            frame = {}
            # 处理每个 camera
            for cam in cameras:
                ret, image = caps[cam].read()
                if not ret:
                    raise RuntimeError(f"Video {cam} ended early at t={t}")
                frame[cam] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 添加 state/action/task
            frame["state"] = encode_state(states[t])
            frame["action"] = encode_state(actions[t])
            frame["task"] = cfg.task_name

            dataset.add_frame(frame)

        # release all video caps
        for cam in cameras:
            caps[cam].release()

        # save this episode
        dataset.save_episode()
        print(f"✅ Saved episode {episode.name}")

    print(" Conversion finished!")



