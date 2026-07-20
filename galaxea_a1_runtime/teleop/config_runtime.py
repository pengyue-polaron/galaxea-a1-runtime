"""Render teleop process-lifecycle settings for the shell supervisor."""

from __future__ import annotations

from galaxea_a1_runtime.configuration.base import number, shell_assign
from galaxea_a1_runtime.configuration.system import render_shell_values
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def bash_config(config: TeleopConfig) -> str:
    system = config.system
    system_exports = render_shell_values(
        system,
        (
            "SYSTEM_CONFIG_PATH",
            "IMAGE",
            "SERIAL",
            "JOINT_STATES_TOPIC",
            "JOINT_TARGET_TOPIC",
            "STAGED_TOPIC",
            "RELAY_STATUS_TOPIC",
            "EEF_POSE_TOPIC",
            "JOINT_TRACKER_NODE",
            "JOINT_TRACKER_NODE_NAME",
            "GRIPPER_MIN_STROKE_MM",
            "GRIPPER_MAX_STROKE_MM",
            "ROS_MASTER_STARTUP_TIMEOUT_S",
            "JOINT_FEEDBACK_STARTUP_TIMEOUT_S",
            "TOPIC_STARTUP_TIMEOUT_S",
            "EMBODIED_OPS_ENDPOINT",
            "EMBODIED_OPS_SERVER_STARTUP_TIMEOUT_S",
            "EMBODIED_OPS_SERVER_SHUTDOWN_TIMEOUT_S",
        ),
    )
    app_values = (
        ("CONFIG_PATH", str(config.path)),
        ("RESET_CONFIG_PATH", str(config.reset.config)),
        ("LEADER_PORT", config.leader.port),
        ("LEADER_ID", config.leader.id),
        ("PREFIX", config.runtime.prefix),
        ("RUN_DIR", config.runtime.run_dir),
        (
            "BRIDGE_STARTUP_TIMEOUT_S",
            number(config.runtime.bridge_startup_timeout_s),
        ),
        ("BRIDGE_STOP_TIMEOUT_S", number(config.runtime.bridge_stop_timeout_s)),
    )
    app_exports = "\n".join(shell_assign(name, value) for name, value in app_values)
    return f"{system_exports}\n{app_exports}"
