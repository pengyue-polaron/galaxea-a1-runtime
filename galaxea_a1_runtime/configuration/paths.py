"""Canonical tracked configuration locations used by repository entrypoints."""

from pathlib import Path

SYSTEM_CONFIG = Path("configs/system/a1.toml")
TELEOP_CONFIG = Path("configs/teleop/a1_so100.toml")
LINGBOT_CONFIG = Path("configs/deployments/lingbot/fruit_placement_eef.toml")
LINGBOT_BATCH_CONFIG = Path("configs/runs/lingbot/fruit_placement.toml")
A1_RESET_POSE = Path("configs/poses/a1_collection_start.toml")
PI05_CONFIG = Path("configs/deployments/pi05/fruit_placement_eef.toml")
