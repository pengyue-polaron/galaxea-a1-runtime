import subprocess
from pathlib import Path

import pytest

from galaxea_a1_runtime.configuration.system import (
    load_system_config,
    render_shell_values,
)


REPO = Path(__file__).resolve().parents[1]
LOADER = REPO / "scripts/runtime/a1_config.sh"
SYSTEM = REPO / "configs/system/a1.toml"


def run_bash(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'set -e; source "{LOADER}"; {body}'],
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_config_loader_propagates_renderer_failure():
    result = run_bash("a1_load_shell_config bash -c 'exit 17'")

    assert result.returncode == 17
    assert "Configuration renderer failed" in result.stderr


def test_shell_config_loader_rejects_empty_output():
    result = run_bash("a1_load_shell_config true")

    assert result.returncode == 2


def test_shell_config_loader_applies_assignments_in_calling_shell():
    result = run_bash(
        "a1_load_shell_config printf 'A1_TEST_VALUE=%s\\n' ready; "
        'test "${A1_TEST_VALUE}" = ready'
    )

    assert result.returncode == 0


def test_repo_pythonpath_includes_runtime_and_embodied_ops_sdk():
    result = run_bash("a1_repo_pythonpath /workspace")

    assert result.returncode == 0
    assert result.stdout.strip() == ("/workspace:/workspace/external/embodied-ops/src")


def test_rosbag_topic_exports_come_from_system_config():
    config = load_system_config(
        SYSTEM,
        repo_root=REPO,
    )
    names = (
        "HOST_COMMAND_TOPIC",
        "MOTOR_STATUS_TOPIC",
        "MOTION_ENABLE_TOPIC",
        "GRIPPER_TARGET_TOPIC",
        "GRIPPER_COMMAND_TOPIC",
        "GRIPPER_FEEDBACK_TOPIC",
    )

    rendered = dict(
        line.split("=", 1) for line in render_shell_values(config, names).splitlines()
    )

    assert rendered == {
        "HOST_COMMAND_TOPIC": config.topics.host_command,
        "MOTOR_STATUS_TOPIC": config.topics.motor_status,
        "MOTION_ENABLE_TOPIC": config.topics.motion_enable,
        "GRIPPER_TARGET_TOPIC": config.topics.gripper_target,
        "GRIPPER_COMMAND_TOPIC": config.topics.gripper_command,
        "GRIPPER_FEEDBACK_TOPIC": config.topics.gripper_feedback,
    }


def test_camera_web_lifecycle_exports_come_from_system_config():
    config = load_system_config(SYSTEM, repo_root=REPO)
    names = (
        "WEB_PREVIEW_BIND",
        "WEB_PREVIEW_PORT",
        "WEB_PREVIEW_STARTUP_TIMEOUT_S",
        "WEB_PREVIEW_SHUTDOWN_TIMEOUT_S",
    )

    rendered = dict(
        line.split("=", 1) for line in render_shell_values(config, names).splitlines()
    )

    assert rendered == {
        "WEB_PREVIEW_BIND": config.web_preview.bind,
        "WEB_PREVIEW_PORT": str(config.web_preview.port),
        "WEB_PREVIEW_STARTUP_TIMEOUT_S": "15",
        "WEB_PREVIEW_SHUTDOWN_TIMEOUT_S": "5",
    }


def test_embodied_ops_lifecycle_exports_come_from_system_config():
    config = load_system_config(SYSTEM, repo_root=REPO)
    names = (
        "EMBODIED_OPS_ENDPOINT",
        "EMBODIED_OPS_SERVER_STARTUP_TIMEOUT_S",
        "EMBODIED_OPS_SERVER_SHUTDOWN_TIMEOUT_S",
    )

    rendered = dict(
        line.split("=", 1) for line in render_shell_values(config, names).splitlines()
    )

    assert rendered == {
        "EMBODIED_OPS_ENDPOINT": config.embodied_ops.endpoint,
        "EMBODIED_OPS_SERVER_STARTUP_TIMEOUT_S": "5",
        "EMBODIED_OPS_SERVER_SHUTDOWN_TIMEOUT_S": "5",
    }


def test_system_config_rejects_non_unix_embodied_ops_endpoint(tmp_path):
    path = tmp_path / "a1.toml"
    path.write_text(
        SYSTEM.read_text().replace(
            'endpoint = "unix:///tmp/galaxea-a1-runtime/embodied-ops.sock"',
            'endpoint = "127.0.0.1:50051"',
        )
    )

    with pytest.raises(ValueError, match="unix"):
        load_system_config(path, repo_root=REPO)


def test_system_config_rejects_command_timeout_outside_rpc_lease_window(tmp_path):
    path = tmp_path / "a1.toml"
    path.write_text(
        SYSTEM.read_text().replace(
            "command_timeout_s = 0.75",
            "command_timeout_s = 0.25",
        )
    )

    with pytest.raises(
        ValueError, match="command_timeout_s must be above rpc_timeout_s"
    ):
        load_system_config(path, repo_root=REPO)


def test_system_config_rejects_removed_orientation_mode(tmp_path):
    path = tmp_path / "a1.toml"
    path.write_text(
        SYSTEM.read_text().replace(
            "[eef]\n",
            '[eef]\norientation_mode = "hold-current"\n',
        )
    )

    with pytest.raises(ValueError, match=r"invalid eef keys.*orientation_mode"):
        load_system_config(path, repo_root=REPO)
