"""Translate ACT configuration into shell-owned lifecycle values."""

from __future__ import annotations

from galaxea_a1_runtime.apps.act.config_schema import ActConfig
from galaxea_a1_runtime.configuration.base import shell_assign
from galaxea_a1_runtime.configuration.system import render_shell_values


def bash_config(config: ActConfig) -> str:
    system_exports = render_shell_values(
        config.system,
        (
            "SYSTEM_CONFIG_PATH",
            "WRIST_BACKEND",
            "WRIST_CAMERA",
            "JOINT_TRACKER_NODE",
            "TMUX_STARTUP_GRACE_S",
        ),
    )
    app_exports = "\n".join(
        [
            shell_assign("CONFIG_PATH", str(config.path)),
            shell_assign("SESSION", config.session.tmux),
            shell_assign("PREFIX", config.runtime.prefix),
            shell_assign("CHECKPOINT", str(config.policy.checkpoint)),
            shell_assign(
                "DEPLOYMENT_READY", "1" if config.policy.deployment_ready else "0"
            ),
        ]
    )
    return f"{system_exports}\n{app_exports}"
