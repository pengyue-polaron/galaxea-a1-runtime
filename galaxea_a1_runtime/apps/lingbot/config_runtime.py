"""Render LingBot process-lifecycle settings for the shell supervisor."""

from __future__ import annotations

from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.configuration.base import number, shell_assign
from galaxea_a1_runtime.configuration.system import render_shell_values


def bash_config(config: LingBotConfig) -> str:
    system = config.system
    system_exports = render_shell_values(
        system,
        (
            "SYSTEM_CONFIG_PATH",
            "WRIST_BACKEND",
            "WRIST_CAMERA",
            "WEB_PREVIEW_BIND",
            "WEB_PREVIEW_PORT",
        ),
    )
    values = (
        ("CONFIG_PATH", str(config.path)),
        ("LINGBOT_HOST", config.server.host),
        ("LINGBOT_PORT", str(config.server.port)),
        ("LINGBOT_CONNECT_TIMEOUT", number(config.server.connect_timeout_s)),
        ("MODEL_CHECKOUT", str(config.policy_server.backend.source.checkout)),
        ("MODEL_PYTHON", str(config.policy_server.backend.environment.python)),
        ("MODEL_ROOT", str(config.policy_server.model.artifact_root)),
        ("TASK_CATALOG_PATH", str(config.task_catalog.path)),
        ("MODEL_SAVE_ROOT", str(config.policy_server.save_root)),
        ("MODEL_MASTER_PORT", str(config.policy_server.master_port)),
        ("MODEL_WORLD_SIZE", str(config.policy_server.world_size)),
        ("MODEL_STARTUP_TIMEOUT", number(config.policy_server.startup_timeout_s)),
        ("MODEL_SHUTDOWN_TIMEOUT", number(config.policy_server.shutdown_timeout_s)),
        (
            "DEPLOYMENT_READY",
            "1" if config.policy_server.deployment_ready else "0",
        ),
    )
    app_exports = "\n".join(shell_assign(name, value) for name, value in values)
    return f"{system_exports}\n{app_exports}"
