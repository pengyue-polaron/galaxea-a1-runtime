import sys
from pathlib import Path
import threading
import time
import signal
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/../ -> DataCoach
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datacoach.data_collection import a1_server
# from omegaconf import DictConfig

threads = []

def start_thread(target, kwargs=None):
    t = threading.Thread(target=lambda: target(**(kwargs or {})), daemon=True)
    t.start()
    threads.append(t)
    return t



def main():
    print("===== Starting A1 Server =====")
    import rospy
    rospy.init_node('a1_server_node', anonymous=True)
    server = a1_server.A1Server()

    print("[1] Starting leader_data_receiver ...")
    start_thread(server.leader_data_receiver)

    print("[2] Starting ros_publisher ...")
    start_thread(server.ros_publisher)

    print("[3] Starting ros_subscriber ...")
    start_thread(server.ros_subscriber)

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
