import threading
import time
import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent # scripts/../ -> DataCoach
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from datacoach.data_collection import a1_replay_bridge, camera_server

import hydra

threads = []

def start_thread(target, kwargs=None):
 
    t = threading.Thread(target=target, kwargs=kwargs or {}, daemon=True)
    t.start()
    threads.append(t)
    return t

@hydra.main(config_path="../../configs", config_name="collect_data.yaml", version_base="1.2")
def main(cfg):
    print("===== Starting data collection services (Replay Mode) =====")
    print("[1] Starting camera_server ...")
    start_thread(camera_server.main, {"cfg": cfg.camera_server})
    time.sleep(2)

    print("[2] Starting a1_replay_bridge ...")
    a1_replay_bridge.main(cfg.a1_replay_bridge)

    print("Services stopped.")

if __name__ == "__main__":
    main()
