import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/../ -> DataCoach
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from datacoach.data_collection import data_collector
import hydra

@hydra.main(config_path="../../configs", config_name="collect_data.yaml", version_base="1.2")
def main(cfg):
    print("Running collect_data ...")
    data_collector.main(cfg.data_collector)  # blocking
    print("✅ Data collection finished.")

if __name__ == "__main__":
    main()
