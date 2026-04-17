import sys
import threading
import time
from pathlib import Path

import hydra

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from a1.data_collection import a1_server
from a1.utils import cfg_get as _cfg_get

threads = []


def start_thread(target, kwargs=None):
    t = threading.Thread(target=lambda: target(**(kwargs or {})), daemon=True)
    t.start()
    threads.append(t)
    return t


@hydra.main(config_path="../../configs", config_name="collect_data.yaml", version_base="1.2")
def main(cfg):
    print("===== Starting A1 Server =====")
    import rospy

    server_cfg = _cfg_get(cfg, "a1_server", None)
    node_name = str(_cfg_get(server_cfg, "node_name", "a1_server_node"))
    anonymous = bool(_cfg_get(server_cfg, "anonymous", True))
    disable_ros_signals = bool(_cfg_get(server_cfg, "disable_ros_signals", False))
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

    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=anonymous, disable_signals=disable_ros_signals)

    server = a1_server.A1Server(server_cfg)
    for idx, component in enumerate(components, start=1):
        target = getattr(server, component, None)
        if target is None:
            raise ValueError(f"Unknown A1Server component: {component}")
        print(f"[{idx}] Starting {component} ...")
        start_thread(target)

    print("A1 Server is running. Press Ctrl+C to stop.")
    try:
        while not rospy.is_shutdown():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all threads...")
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
        print("A1 Server stopped.")


if __name__ == "__main__":
    main()
