from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "scripts/apps/cameras/a1_camera_web_runtime.sh"


def test_camera_web_runtime_is_marked_persistent_and_has_no_tmux_or_start_command():
    source = RUNTIME.read_text()

    assert "a1_process_start" in source
    assert "a1_process_is_running" in source
    assert "a1_tmux" not in source
    assert "start)" not in source
    result = subprocess.run(
        [str(RUNTIME), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "[stop|status|logs]" in result.stdout


def test_persistent_camera_bridge_owns_web_while_apps_use_its_raw_stream():
    standalone = (REPO / "scripts/apps/cameras/a1_camera_web.py").read_text()
    policy = (REPO / "galaxea_a1_runtime/apps/policy_camera.py").read_text()
    teleop = (REPO / "galaxea_a1_runtime/apps/teleop/collector_camera.py").read_text()
    lingbot_runtime = (REPO / "scripts/apps/lingbot/a1_lingbot_runtime.sh").read_text()
    pi05_runtime = (REPO / "scripts/apps/pi05/a1_pi05_runtime.sh").read_text()
    teleop_runtime = (REPO / "scripts/apps/teleop/a1_teleop_runtime.sh").read_text()

    assert "CameraBridgeServer" in standalone
    assert "CameraWebPreview" in standalone
    for source in (policy, teleop):
        assert "CameraBridgeReaders" in source
        assert "open_configured_camera" not in source
        assert "CameraWebPreview" not in source
    for source in (lingbot_runtime, pi05_runtime, teleop_runtime):
        assert "Handing camera ownership" not in source
        assert '"${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}" stop' not in source
    assert "suspend_camera_monitor" not in lingbot_runtime
    assert "ensure_camera_monitor" in lingbot_runtime
