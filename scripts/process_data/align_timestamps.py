import argparse
import shutil
import pickle
import numpy as np
from pathlib import Path
import hydra
import cv2
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/../ -> DataCoach
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from datacoach.data_processing.align_timestamps import process_single_demo
import hydra


@hydra.main(config_path="../../configs", config_name="process_data.yaml", version_base="1.2")
def main(cfg):
    raw_root = Path(cfg.raw_data_path) / cfg.task_name
    save_root = Path(cfg.processed_data_path) / cfg.task_name
    save_root.mkdir(parents=True, exist_ok=True)

    demo_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])

    for demo_dir in demo_dirs:
        save_demo_dir = save_root / demo_dir.name
        process_single_demo(demo_dir, save_demo_dir)


if __name__ == "__main__":
    main()
