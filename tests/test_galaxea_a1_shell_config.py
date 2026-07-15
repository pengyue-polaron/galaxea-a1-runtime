import subprocess
from pathlib import Path

from galaxea_a1_runtime.configuration.system import (
    load_system_config,
    render_shell_values,
)


REPO = Path(__file__).resolve().parents[1]
LOADER = REPO / "scripts/runtime/a1_config.sh"


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


def test_rosbag_topic_exports_come_from_system_config():
    config = load_system_config(
        REPO / "configs/system/a1.toml",
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
