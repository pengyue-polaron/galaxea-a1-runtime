import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SERVICES = REPO / "scripts/runtime/a1_services.sh"


def run_services_bash(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{SERVICES}"; {body}'],
        text=True,
        capture_output=True,
        check=False,
    )


def test_wait_topic_fails_immediately_with_stopped_container_log() -> None:
    result = run_services_bash(
        r"""
        TOPIC_STARTUP_TIMEOUT_S=15
        docker() {
          case "$1" in
            exec) return 1 ;;
            inspect) printf '%s\n' 'exited (exit 1)' ;;
            logs) printf '%s\n' 'roslaunch failure marker' ;;
            *) return 2 ;;
          esac
        }
        a1_wait_topic dead-tracker /end_effector_pose
        """
    )

    assert result.returncode == 1
    assert (
        "Container dead-tracker is exited (exit 1) while waiting for a message on "
        "/end_effector_pose."
    ) in result.stderr
    assert "roslaunch failure marker" in result.stderr
    assert "after 15s" not in result.stderr


def test_wait_topic_succeeds_without_inspect_after_message() -> None:
    result = run_services_bash(
        r"""
        TOPIC_STARTUP_TIMEOUT_S=15
        docker() {
          case "$1" in
            exec) return 0 ;;
            *) printf '%s\n' "unexpected docker command: $1" >&2; return 2 ;;
          esac
        }
        a1_wait_topic live-tracker /end_effector_pose
        """
    )

    assert result.returncode == 0
    assert result.stderr == ""


def test_relay_command_executes_its_python312_entrypoint_directly() -> None:
    result = run_services_bash(
        r"""
        ROOT=/workspace
        SYSTEM_CONFIG_PATH=/workspace/configs/system/a1.toml
        a1_container_run() {
          printf '%s\n' "$1|$2|$3"
        }
        a1_start_command_relay relay-container
        """
    )

    assert result.returncode == 0
    assert "relay|relay-container|" in result.stdout
    assert "exec /workspace/scripts/runtime/safe_arm_command_relay.py" in result.stdout
    assert "--config '/workspace/configs/system/a1.toml'" in result.stdout
    assert "exec python3 " not in result.stdout
