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
    
from datacoach.data_processing.data_converter import convert_a1_to_lerobot
import hydra


@hydra.main(config_path="../../configs", config_name="process_data.yaml", version_base="1.2")
def main(cfg):
    convert_a1_to_lerobot(cfg)

if __name__ == "__main__":
    main()
