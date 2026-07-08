from pathlib import Path
import re

from galaxea_a1_runtime.apps.lingbot.config import bridge_argv, load_lingbot_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "inference" / "lingbot_va_a1.toml"


def test_lingbot_gripper_scale_default_is_linear_with_physical_stroke():
    bridge = REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"
    stats = REPO / "scripts" / "process_data" / "compute_eef_norm_stats_from_bags.py"

    assert '"--gripper-stroke-scale", type=float, default=60.0' in bridge.read_text()
    assert '"--gripper-stroke-max", type=float, default=60.0' in bridge.read_text()
    assert 'default=60.0,' in stats.read_text()


def test_lingbot_bridge_resolves_repo_root_from_nested_app_path():
    bridge = REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"

    assert "ROOT_DIR = Path(__file__).resolve().parents[3]" in bridge.read_text()


def test_lingbot_condition_state_does_not_workspace_clamp_feedback():
    bridge = REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"
    text = bridge.read_text()
    match = re.search(r"def _normalize_condition_action\(.*?\n    def _current_or_initial_action8", text, flags=re.S)

    assert match is not None
    assert "action[:3] = np.minimum" not in match.group(0)


def test_lingbot_runtime_wrapper_reads_tracked_config_without_extra_args():
    runtime = REPO / "scripts" / "apps" / "lingbot" / "a1_lingbot_runtime.sh"
    text = runtime.read_text()

    assert 'dirname "${BASH_SOURCE[0]}")/../../..' in text
    assert "configs/inference/lingbot_va_a1.toml" in text
    assert "galaxea_a1_runtime.apps.lingbot.config" in text
    assert '"${BRIDGE_ARGS[@]}"' in text
    assert "A1_LINGBOT_CONFIG" not in text
    assert "A1_LINGBOT_BRIDGE_EXTRA_ARGS" not in text
    assert "A1_LINGBOT_MAX_XYZ_DELTA" not in text
    assert "--max-xyz-delta" not in text
    assert 'EEF_SERVO_GAIN="${A1_LINGBOT_EEF_SERVO_GAIN' not in text
    assert '--eef-servo-gain "${EEF_SERVO_GAIN}"' not in text


def test_lingbot_config_locks_runtime_defaults():
    config = load_lingbot_config(CONFIG, repo_root=REPO)
    args = bridge_argv(config)

    assert config.session.tmux == "lingbot-a1"
    assert config.server.host == "127.0.0.1"
    assert config.execution.execute is True
    assert config.execution.step_actions is True
    assert config.eef.orientation_mode == "hold-current"
    assert config.eef.xyz_min == (0.06, -0.27, 0.06)
    assert config.gripper.stroke_scale_mm == 60.0
    assert args[args.index("--cmd-pose-topic") + 1] == "/a1_ee_target"
    assert args[args.index("--cam1-device") + 1].startswith("/dev/v4l/by-id/")
    assert "--execute" in args
    assert "--step-actions" in args


def test_lingbot_bridge_uses_shared_camera_io():
    bridge = REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"
    text = bridge.read_text()

    assert "from galaxea_a1_runtime.hardware.cameras import" in text
    assert "class RealSenseCamera" not in text
    assert "class OpenCVCamera" not in text


def test_lingbot_aux_tools_do_not_hardcode_personal_checkout_paths():
    decoder = (REPO / "scripts" / "apps" / "lingbot" / "decode_lingbot_latents.py").read_text()

    assert "/home/pengyue" not in decoder
    assert "--lingbot-root" in decoder
    assert "LINGBOT_VA_ROOT" in decoder
