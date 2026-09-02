import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SERVICES = REPO / "scripts/runtime/a1_services.sh"
OBSERVABILITY_RUNTIME = REPO / "scripts/runtime/a1_observability_runtime.sh"


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


def test_output_writer_mounts_only_outputs_read_write(tmp_path: Path) -> None:
    docker_log = tmp_path / "docker.log"
    result = run_services_bash(
        f"""
        ROOT={tmp_path}
        IMAGE=a1-test-image
        DOCKER_LOG={docker_log}
        docker() {{ printf '%s\\n' "$*" >> "${{DOCKER_LOG}}"; }}
        a1_container_run output-writer recorder 'exec rosbag record'
        """
    )

    assert result.returncode == 0
    commands = docker_log.read_text()
    assert f"{tmp_path}:/workspace:ro" in commands
    assert f"{tmp_path}/outputs:/workspace/outputs:rw" in commands
    assert f"{tmp_path}:/workspace:rw" not in commands
    assert "-e PYTHONPATH=/workspace:/workspace/external/embodied-ops/src" in commands


def test_managed_cleanup_preserves_exact_persistent_observation_names(
    tmp_path: Path,
) -> None:
    count_file = tmp_path / "count"
    removed_file = tmp_path / "removed"
    result = run_services_bash(
        f"""
        COUNT_FILE={count_file}
        REMOVED_FILE={removed_file}
        printf '0\n' > "${{COUNT_FILE}}"
        docker() {{
          case "$1" in
            ps)
              local count
              count="$(cat "${{COUNT_FILE}}")"
              if [[ "${{count}}" == "0" ]]; then
                printf '1\n' > "${{COUNT_FILE}}"
                printf '%s\n' roscore telemetry foxglove driver
              else
                printf '%s\n' roscore telemetry foxglove
              fi
              ;;
            inspect)
              case "${{@: -1}}" in
                roscore) printf '/a1-observability-roscore\n' ;;
                telemetry) printf '/a1-observability-telemetry\n' ;;
                foxglove) printf '/a1-observability-foxglove\n' ;;
                driver) printf '/active-driver\n' ;;
              esac
              ;;
            rm)
              shift 2
              printf '%s\n' "$@" > "${{REMOVED_FILE}}"
              ;;
            *) return 2 ;;
          esac
        }}
        a1_remove_all_managed_containers \
          a1-observability-roscore \
          a1-observability-telemetry \
          a1-observability-foxglove
        """
    )

    assert result.returncode == 0
    assert removed_file.read_text().splitlines() == ["driver"]


def test_observability_starts_telemetry_and_scoped_operator_bridge_profiles() -> None:
    result = run_services_bash(
        f"""
        ROOT={REPO}
        SYSTEM_CONFIG_PATH={REPO}/configs/system/a1.toml
        OBSERVABILITY_ENABLED=true
        FOXGLOVE_BIND=0.0.0.0
        FOXGLOVE_PORT=8766
        FOXGLOVE_STARTUP_TIMEOUT_S=15
        FOXGLOVE_GRAPH_UPDATE_MS=1000
        FOXGLOVE_SEND_BUFFER_LIMIT_BYTES=10000000
        FOXGLOVE_TOPIC_WHITELIST_YAML='["^/joint_states_host$"]'
        FOXGLOVE_SERVICE_WHITELIST_YAML='["^/a1/ops/collection/start$"]'
        FOXGLOVE_NO_MATCH_ALLOWLIST_YAML='["^$"]'
        FOXGLOVE_CAPABILITIES_YAML='["connectionGraph","assets","services"]'
        FOXGLOVE_ASSET_URI_ALLOWLIST_YAML='["^package://mobiman/robot.urdf$"]'
        OBSERVABILITY_DIAGNOSTICS_TOPIC=/a1/diagnostics
        A1_ROS_PREFIX=ros
        a1_container_run() {{ printf '%s|%s|%s\n' "$1" "$2" "$3"; }}
        a1_wait_topic() {{ printf 'wait|%s|%s\n' "$1" "$2"; }}
        a1_observability_stack_is_ready() {{ return 1; }}
        a1_foxglove_port_is_listening() {{ return 1; }}
        timeout() {{ return 0; }}
        a1_start_observability telemetry-container foxglove-container
        """
    )

    assert result.returncode == 0
    assert "telemetry|telemetry-container|" in result.stdout
    assert "a1_observability.py" in result.stdout
    assert "core|foxglove-container|" in result.stdout
    assert "foxglove_bridge_scoped.launch" in result.stdout
    assert "service_whitelist:" in result.stdout
    assert "clientPublish" not in result.stdout
    assert "services" in result.stdout


def test_observability_reuses_the_healthy_persistent_stack() -> None:
    result = run_services_bash(
        f"""
        ROOT={REPO}
        SYSTEM_CONFIG_PATH={REPO}/configs/system/a1.toml
        OBSERVABILITY_ENABLED=true
        FOXGLOVE_BIND=0.0.0.0
        FOXGLOVE_PORT=8766
        FOXGLOVE_STARTUP_TIMEOUT_S=15
        FOXGLOVE_GRAPH_UPDATE_MS=1000
        FOXGLOVE_SEND_BUFFER_LIMIT_BYTES=10000000
        FOXGLOVE_TOPIC_WHITELIST_YAML='["^/joint_states_host$"]'
        FOXGLOVE_SERVICE_WHITELIST_YAML='["^/a1/ops/collection/start$"]'
        FOXGLOVE_NO_MATCH_ALLOWLIST_YAML='["^$"]'
        FOXGLOVE_CAPABILITIES_YAML='["connectionGraph","assets","services"]'
        FOXGLOVE_ASSET_URI_ALLOWLIST_YAML='["^package://mobiman/robot.urdf$"]'
        OBSERVABILITY_DIAGNOSTICS_TOPIC=/a1/diagnostics
        a1_observability_stack_is_ready() {{ return 0; }}
        a1_container_run() {{ printf 'unexpected start\n' >&2; return 2; }}
        a1_start_observability telemetry-container foxglove-container
        """
    )

    assert result.returncode == 0
    assert "already ready" in result.stdout
    assert "unexpected start" not in result.stderr


def test_standalone_observability_entrypoint_exposes_no_execution_service() -> None:
    result = subprocess.run(
        ["bash", str(OBSERVABILITY_RUNTIME), "help"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    source = OBSERVABILITY_RUNTIME.read_text(encoding="utf-8")

    assert result.returncode == 0
    assert "never starts the A1 driver" in result.stdout
    assert "a1_preflight_container_host" in source
    assert "a1_start_observability" in source
    assert "a1_start_driver" not in source
    assert "a1_start_command_relay" not in source


def test_standalone_observability_status_fails_when_stack_is_absent(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\nif [[ \"$1\" == inspect ]]; then printf 'false\\n'; fi\n"
    )
    timeout = fake_bin / "timeout"
    timeout.write_text("#!/usr/bin/env bash\nexit 1\n")
    docker.chmod(0o755)
    timeout.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(OBSERVABILITY_RUNTIME), "status"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "not listening" in result.stdout
