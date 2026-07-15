"""Shared Galaxea A1 runtime constants."""

from __future__ import annotations

ARM_JOINT_COUNT = 6

IDLE_TIMEOUT_CODE = 1 << 6

LEROBOT_DATASET_FORMAT = "v3.0"

# roslaunch node names are basenames; ROS graph lookups use resolved global names.
EE_TRACKER_NODE_NAME = "eeTracker_demo_node"
JOINT_TRACKER_NODE_NAME = "jointTracker_demo_node"
EE_TRACKER_NODE = f"/{EE_TRACKER_NODE_NAME}"
JOINT_TRACKER_NODE = f"/{JOINT_TRACKER_NODE_NAME}"
SAFE_RELAY_NODE = "/safe_arm_command_relay"
SAFE_RELAY_SCRIPT = "safe_arm_command_relay.py"
