import threading
import sys
import time

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent # scripts/../ -> DataCoach
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from datacoach.data_collection import a1_replay_bridge, a1_server, camera_server
from datacoach.utils import cfg_get as _cfg_get

import hydra

threads = []

def start_thread(target, kwargs=None):
 
    t = threading.Thread(target=target, kwargs=kwargs or {}, daemon=True)
    t.start()
    threads.append(t)
    return t


def _start_camera_server(cfg):
    camera_cfg = _cfg_get(cfg, "camera_server", None)
    camera_enabled = bool(_cfg_get(camera_cfg, "enabled", True))
    if not camera_enabled:
        print("[1] Skipping camera_server (disabled by config).")
        return

    print("[1] Starting camera_server ...")
    start_thread(camera_server.main, {"cfg": camera_cfg})
    time.sleep(2)


def _run_replay_mode(cfg):
    print("===== Starting data collection services (Replay Mode) =====")
    _start_camera_server(cfg)
    print("[2] Starting a1_replay_bridge ...")
    a1_replay_bridge.main(_cfg_get(cfg, "a1_replay_bridge", None))


def _run_live_mode(cfg):
    print("===== Starting data collection services (Live Mode) =====")
    _start_camera_server(cfg)

    server_cfg = _cfg_get(cfg, "a1_server", None)
    components_cfg = _cfg_get(server_cfg, "components", None)
    if components_cfg is None:
        components = [
            "leader_data_receiver",
            "ros_publisher",
            "ros_subscriber",
            "policy_action_subscriber",
        ]
    else:
        components = list(components_cfg)

    if not components:
        print("[2] Skipping A1 live bridge components (empty component list).")
        print("Camera-only live services started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    import rospy

    node_name = str(_cfg_get(server_cfg, "node_name", "a1_server_node"))
    anonymous = bool(_cfg_get(server_cfg, "anonymous", True))
    disable_ros_signals = bool(_cfg_get(server_cfg, "disable_ros_signals", False))
    default_components = [
        "leader_data_receiver",
        "ros_publisher",
        "ros_subscriber",
        "policy_action_subscriber",
    ]
    if components == ["default"]:
        components = default_components

    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=anonymous, disable_signals=disable_ros_signals)

    server = a1_server.A1Server(server_cfg)
    print("[2] Starting A1 live bridge components ...")
    for component in components:
        target = getattr(server, component, None)
        if target is None:
            raise ValueError(f"Unknown A1Server component: {component}")
        print(f"  - {component}")
        start_thread(target)

    print("Live services started. Press Ctrl+C to stop.")
    while not rospy.is_shutdown():
        time.sleep(1)

@hydra.main(config_path="../../configs", config_name="collect_data.yaml", version_base="1.2")
def main(cfg):
    mode = str(_cfg_get(cfg, "service_mode", "replay")).strip().lower()
    if mode == "replay":
        _run_replay_mode(cfg)
    elif mode == "live":
        _run_live_mode(cfg)
    else:
        raise ValueError(f"Unsupported service_mode: {mode}, expected one of: replay/live")

if __name__ == "__main__":
    main()
