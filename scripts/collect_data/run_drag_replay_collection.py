import threading
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datacoach.data_collection import a1_replay_bridge, camera_server, data_collector
import hydra


threads = []


def start_thread(target, kwargs=None):
    t = threading.Thread(target=target, kwargs=kwargs or {}, daemon=True)
    t.start()
    threads.append(t)
    return t


@hydra.main(config_path="../../configs", config_name="drag_replay.yaml", version_base="1.2")
def main(cfg):
    print("===== DragDataCoach Replay Collection =====")
    print("[1] Starting camera_server ...")
    cam_thread = start_thread(camera_server.main, {"cfg": cfg.camera_server})
    time.sleep(1.5)
    if not cam_thread.is_alive():
        raise RuntimeError("camera_server exited early. Check camera config/device availability.")

    print("[2] Starting a1_replay_bridge ...")
    bridge_cfg = cfg.a1_replay_bridge
    bridge_cfg.disable_ros_signals = True
    bridge_thread = start_thread(a1_replay_bridge.main, {"cfg": bridge_cfg})
    time.sleep(1.0)
    if not bridge_thread.is_alive():
        raise RuntimeError(
            "a1_replay_bridge exited early. Check ROS master and source setup.bash in this terminal."
        )

    print("[3] Starting data_collector ...")
    data_collector.main(cfg.data_collector)
    print("✅ DragDataCoach collection finished.")


if __name__ == "__main__":
    main()
