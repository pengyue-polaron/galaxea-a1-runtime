import threading
import time
import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datacoach.data_collection import a1_replay_bridge, camera_server, data_collector
from datacoach.data_collection.data_collector import ReplayAction
import hydra


threads = []


def start_thread(name, target, shutdown_event, failure_handler, kwargs=None):
    def runner():
        try:
            target(**(kwargs or {}))
            if not shutdown_event.is_set():
                raise RuntimeError(f"{name} exited unexpectedly.")
        except BaseException as exc:
            failure_handler(name, exc, traceback.format_exc())

    t = threading.Thread(target=runner, name=name, daemon=False)
    t.start()
    threads.append(t)
    return t


def run_single_collection(cfg, shutdown_event, failure_lock, background_failure, collector_ref):
    """Run a single data collection session.

    Returns:
        ReplayAction: The action chosen by the user after completion.
    """

    def handle_background_failure(name, exc, tb):
        with failure_lock:
            if background_failure["error"] is not None:
                return
            message = f"{name} failed: {exc}"
            background_failure["error"] = RuntimeError(message)
        print(f"[FATAL] {message}")
        if tb:
            print(tb.rstrip())
        shutdown_event.set()
        collector = collector_ref["value"]
        if collector is not None:
            collector.abort(message, save_recording=False)

    def raise_background_failure():
        err = background_failure["error"]
        if err is not None:
            raise err

    print("[1] Starting camera_server ...")
    cam_thread = start_thread(
        "camera_server",
        camera_server.main,
        shutdown_event,
        handle_background_failure,
        {"cfg": cfg.camera_server, "stop_event": shutdown_event},
    )
    time.sleep(1.5)
    if not cam_thread.is_alive():
        raise_background_failure()
        raise RuntimeError("camera_server exited early. Check camera config/device availability.")

    print("[2] Starting a1_replay_bridge ...")
    bridge_cfg = cfg.a1_replay_bridge
    bridge_cfg.disable_ros_signals = True
    bridge_thread = start_thread(
        "a1_replay_bridge",
        a1_replay_bridge.main,
        shutdown_event,
        handle_background_failure,
        {"cfg": bridge_cfg, "stop_event": shutdown_event},
    )
    time.sleep(1.0)
    if not bridge_thread.is_alive():
        raise_background_failure()
        raise RuntimeError(
            "a1_replay_bridge exited early. Check ROS master and source setup.bash in this terminal."
        )

    print("[3] Starting data_collector ...")
    collector = data_collector.DataCollector(cfg.data_collector)
    collector_ref["value"] = collector
    raise_background_failure()

    action = ReplayAction.CLOSE
    try:
        action = collector.run()
        raise_background_failure()
        print("✅ DragDataCoach collection finished.")
    finally:
        print("[4] Stopping background services ...")
        shutdown_event.set()
        for name, thread in (("a1_replay_bridge", bridge_thread), ("camera_server", cam_thread)):
            thread.join(timeout=3.0)
            if thread.is_alive():
                print(f"[WARN] {name} did not stop within timeout.")
            else:
                print(f"[OK] {name} stopped.")

    return action


@hydra.main(config_path="../../configs", config_name="drag_replay.yaml", version_base="1.2")
def main(cfg):
    print("===== DragDataCoach Replay Collection =====")

    replay_count = 0
    while True:
        replay_count += 1
        print(f"\n{'=' * 50}")
        print(f"Replay Session #{replay_count}")
        print(f"{'=' * 50}\n")

        # Reset shared state for each replay session
        shutdown_event = threading.Event()
        failure_lock = threading.Lock()
        background_failure = {"error": None}
        collector_ref = {"value": None}
        threads.clear()

        try:
            action = run_single_collection(
                cfg, shutdown_event, failure_lock, background_failure, collector_ref
            )

            if action == ReplayAction.CLOSE:
                print("\nUser chose to close. Exiting...")
                break
            elif action == ReplayAction.REPLAY:
                print("\nUser chose to replay again...")
                continue
        except Exception as e:
            print(f"\nError during replay: {e}")
            # On error, also prompt user for action
            print("\nAfter error, choose next action:")
            print("  [1] Close")
            print("  [2] Replay again")
            while True:
                try:
                    user_input = input("Enter your choice: ").strip()
                    if user_input == "1":
                        print("\nUser chose to close. Exiting...")
                        return
                    elif user_input == "2":
                        print("\nUser chose to replay again...")
                        break
                    else:
                        print("Invalid input. Please enter 1 or 2:")
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting...")
                    return


if __name__ == "__main__":
    main()
