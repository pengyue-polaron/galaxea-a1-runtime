"""Shared Galaxea A1 runtime constants."""

from __future__ import annotations

ARM_JOINT_COUNT = 6
GRIPPER_JOINT_COUNT = 1
TOTAL_JOINT_COUNT = ARM_JOINT_COUNT + GRIPPER_JOINT_COUNT

IDLE_TIMEOUT_CODE = 1 << 6

EE_TARGET_TOPIC = "/a1_ee_target"
STAGED_ARM_COMMAND_TOPIC = "/arm_joint_command_a1_staged"
HOST_ARM_COMMAND_TOPIC = "/arm_joint_command_host"
RELAY_ENABLE_TOPIC = "/a1_arm_motion_enable"
RELAY_STATUS_TOPIC = "/a1_arm_relay_status"
GRIPPER_COMMAND_TOPIC = "/gripper_position_control_host"
EEF_FEEDBACK_TOPIC = "/end_effector_pose"
JOINT_FEEDBACK_TOPIC = "/joint_states_host"
ARM_STATUS_TOPIC = "/arm_status_host"

DEFAULT_MAX_COMMAND_AGE_S = 0.25
DEFAULT_RELAY_ARMING_TIMEOUT_S = 1.0
DEFAULT_MAX_INITIAL_COMMAND_ERROR_RAD = 0.05
DEFAULT_MAX_EEF_DELTA_M = 0.03
DEFAULT_MAX_ROT_DELTA_RAD = 0.15

LEROBOT_BASELINE = "0.6.x"
LEROBOT_DATASET_FORMAT = "v3.0"
