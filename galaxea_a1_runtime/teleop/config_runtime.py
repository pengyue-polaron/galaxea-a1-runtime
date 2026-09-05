"""Render teleop process-lifecycle settings for the shell supervisor."""

from __future__ import annotations

from galaxea_a1_runtime.configuration.base import number, shell_assign
from galaxea_a1_runtime.configuration.system import bash_config as system_bash_config
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def bash_config(config: TeleopConfig) -> str:
    system_exports = system_bash_config(config.system)
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
