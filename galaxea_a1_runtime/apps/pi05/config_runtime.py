"""Render narrow pi0.5 process-lifecycle values for shell supervision."""

from __future__ import annotations

from galaxea_a1_runtime.apps.pi05.config_schema import Pi05Config
from galaxea_a1_runtime.configuration.base import number, shell_assign
from galaxea_a1_runtime.configuration.system import render_shell_values


def bash_config(config: Pi05Config) -> str:
    system_exports = render_shell_values(
        config.system,
        (
            "SYSTEM_CONFIG_PATH",
            "TMUX_STARTUP_GRACE_S",
        ),
    )
    values = (
        ("CONFIG_PATH", str(config.path)),
        ("SESSION", config.session.tmux),
        ("MODEL_SESSION", config.session.model_tmux),
        ("MODEL_CHECKOUT", str(config.backend.source.checkout)),
        ("MODEL_PYTHON", str(config.backend.environment.python)),
        ("TASK_CATALOG_PATH", str(config.task_catalog.path)),
        ("MODEL_HOST", config.server.host),
        ("MODEL_PORT", str(config.server.port)),
        ("MODEL_STARTUP_TIMEOUT", number(config.session.startup_timeout_s)),
        ("DEPLOYMENT_READY", "1" if config.deployment_ready else "0"),
    )
    return f"{system_exports}\n" + "\n".join(
        shell_assign(name, value) for name, value in values
    )
