import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/../ -> DataCoach
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datacoach.data_collection import data_collector
from datacoach.data_collection.data_collector import ReplayAction
import hydra


@hydra.main(config_path="../../configs", config_name="collect_data.yaml", version_base="1.2")
def main(cfg):
    print("===== DataCoach Data Collection =====")

    session_count = 0
    while True:
        session_count += 1
        print(f"\n{'=' * 50}")
        print(f"Collection Session #{session_count}")
        print(f"{'=' * 50}\n")

        try:
            collector = data_collector.DataCollector(cfg.data_collector)
            action = collector.run()

            if action == ReplayAction.CLOSE:
                print("\nUser chose to close. Exiting...")
                break
            elif action == ReplayAction.REPLAY:
                print("\nUser chose to record again...")
                continue
        except Exception as e:
            print(f"\nError during collection: {e}")
            # On error, also prompt user for action
            print("\nAfter error, choose next action:")
            print("  [1] Close")
            print("  [2] Record again")
            while True:
                try:
                    user_input = input("Enter your choice: ").strip()
                    if user_input == "1":
                        print("\nUser chose to close. Exiting...")
                        return
                    elif user_input == "2":
                        print("\nUser chose to record again...")
                        break
                    else:
                        print("Invalid input. Please enter 1 or 2:")
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting...")
                    return

    print("✅ Data collection finished.")


if __name__ == "__main__":
    main()
