from pathlib import Path
import re


def test_lingbot_gripper_scale_default_is_linear_with_physical_stroke():
    repo = Path(__file__).resolve().parents[1]
    bridge = repo / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"
    stats = repo / "scripts" / "process_data" / "compute_eef_norm_stats_from_bags.py"

    assert '"--gripper-stroke-scale", type=float, default=60.0' in bridge.read_text()
    assert '"--gripper-stroke-max", type=float, default=60.0' in bridge.read_text()
    assert 'default=60.0,' in stats.read_text()


def test_lingbot_bridge_resolves_repo_root_from_nested_app_path():
    repo = Path(__file__).resolve().parents[1]
    bridge = repo / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"

    assert "ROOT_DIR = Path(__file__).resolve().parents[3]" in bridge.read_text()


def test_lingbot_condition_state_does_not_workspace_clamp_feedback():
    repo = Path(__file__).resolve().parents[1]
    bridge = repo / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"
    text = bridge.read_text()
    match = re.search(r"def _normalize_condition_action\(.*?\n    def _current_or_initial_action8", text, flags=re.S)

    assert match is not None
    assert "action[:3] = np.minimum" not in match.group(0)


def test_lingbot_runtime_wrapper_keeps_advanced_flags_out_of_default_command():
    repo = Path(__file__).resolve().parents[1]
    runtime = repo / "scripts" / "apps" / "lingbot" / "a1_lingbot_runtime.sh"
    text = runtime.read_text()

    assert 'dirname "${BASH_SOURCE[0]}")/../../..' in text
    assert "A1_LINGBOT_BRIDGE_EXTRA_ARGS" in text
    assert "A1_LINGBOT_MAX_XYZ_DELTA" not in text
    assert "--max-xyz-delta" not in text
    assert 'EEF_SERVO_GAIN="${A1_LINGBOT_EEF_SERVO_GAIN' not in text
    assert '--eef-servo-gain "${EEF_SERVO_GAIN}"' not in text


def test_lingbot_bridge_uses_shared_camera_io():
    repo = Path(__file__).resolve().parents[1]
    bridge = repo / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"
    text = bridge.read_text()

    assert "from galaxea_a1_runtime.hardware.cameras import" in text
    assert "class RealSenseCamera" not in text
    assert "class OpenCVCamera" not in text
