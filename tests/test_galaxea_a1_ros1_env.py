import os
from pathlib import Path

import pytest

from galaxea_a1_runtime.runtime.ros1_env import (
    ROS1_PYTHON_LOG_CONFIG,
    configure_ros1_python,
)


REPO = Path(__file__).resolve().parents[1]


def test_ros1_python_bootstrap_registers_repo_logging_config(monkeypatch):
    monkeypatch.delenv("ROS_PYTHON_LOG_CONFIG_FILE", raising=False)

    configure_ros1_python(REPO)

    assert os.environ["ROS_PYTHON_LOG_CONFIG_FILE"] == str(
        REPO / ROS1_PYTHON_LOG_CONFIG
    )


def test_ros1_python_bootstrap_preserves_explicit_logging_config(monkeypatch, tmp_path):
    custom_config = tmp_path / "python_logging.conf"
    custom_config.touch()
    monkeypatch.setenv("ROS_PYTHON_LOG_CONFIG_FILE", str(custom_config))

    configure_ros1_python(REPO)

    assert os.environ["ROS_PYTHON_LOG_CONFIG_FILE"] == str(custom_config)


def test_ros1_python_bootstrap_rejects_missing_logging_config(monkeypatch, tmp_path):
    missing_config = tmp_path / "missing.conf"
    monkeypatch.setenv("ROS_PYTHON_LOG_CONFIG_FILE", str(missing_config))

    with pytest.raises(FileNotFoundError, match="logging configuration not found"):
        configure_ros1_python(REPO)
