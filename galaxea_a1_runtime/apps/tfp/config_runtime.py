"""Render TFP-Ultra process lifecycle settings for the shell supervisor."""

from __future__ import annotations

from galaxea_a1_runtime.apps.tfp.config_schema import TFPConfig
from galaxea_a1_runtime.configuration.base import number, shell_assign
from galaxea_a1_runtime.configuration.system import render_shell_values
from galaxea_a1_runtime.models.backend import isolated_backend_pythonpath


def bash_config(config: TFPConfig) -> str:
    system_exports = render_shell_values(
        config.system,
        (
            "SYSTEM_CONFIG_PATH",
            "WEB_PREVIEW_BIND",
            "WEB_PREVIEW_PORT",
            "A1_ROBOT_SERVICE_ENDPOINT",
            "A1_ROBOT_SERVICE_SERVER_STARTUP_TIMEOUT_S",
            "A1_ROBOT_SERVICE_SERVER_SHUTDOWN_TIMEOUT_S",
        ),
    )
    values = (
        ("CONFIG_PATH", str(config.path)),
        ("MODEL_CHECKOUT", str(config.backend.source.checkout)),
        ("MODEL_PYTHON", str(config.backend.environment.python)),
        ("A1_REPO_PYTHONPATH", isolated_backend_pythonpath(config.repo_root)),
        ("MODEL_CHECKPOINT", str(config.model.checkpoint)),
        ("MODEL_METADATA", str(config.model.metadata)),
        ("DEPLOYMENT_READY", "1" if config.deployment_ready else "0"),
        ("TFP_TASK_ID", config.task_id),
        ("TFP_RESET_BEFORE_RUN", "1" if config.reset.before_run else "0"),
        ("TFP_RESET_CONFIG", str(config.reset.config)),
        ("TFP_EXEC_RATE", number(config.execution.exec_rate)),
    )
    app_exports = "\n".join(shell_assign(name, value) for name, value in values)
    return f"{system_exports}\n{app_exports}"
