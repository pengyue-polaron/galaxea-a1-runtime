import os
import sys
from pathlib import Path

import pytest

from galaxea_a1_runtime.runtime.ros1_env import (
    ROS1_PYTHON_LOG_CONFIG,
    configure_ros1_python,
)


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_python_path(monkeypatch):
    # configure_ros1_python intentionally mutates sys.path. Give each test a
    # private copy so the default system-site behavior cannot leak into later
    # dataset tests running in the same pytest process.
    monkeypatch.setattr(sys, "path", list(sys.path))


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


def test_ros1_python_bootstrap_can_exclude_system_site(monkeypatch):
    monkeypatch.setattr(
        sys,
        "path",
        [*sys.path, "/usr/lib/python3/dist-packages"],
    )

    configure_ros1_python(REPO, include_system_site=False)

    assert "/usr/lib/python3/dist-packages" not in sys.path
