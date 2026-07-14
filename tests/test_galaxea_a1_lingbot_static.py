from pathlib import Path
import re
import subprocess

from galaxea_a1_runtime.apps.lingbot.config import bridge_argv, load_lingbot_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "inference" / "lingbot_va_a1.toml"


def test_lingbot_gripper_default_uses_binary_full_stroke():
    bridge = REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"

    assert '"--gripper-stroke-scale", type=float, default=200.0' in bridge.read_text()
    assert '"--gripper-stroke-max", type=float, default=200.0' in bridge.read_text()
    assert '"--gripper-command-open-threshold", type=float, default=0.5' in bridge.read_text()


def test_lingbot_bridge_resolves_repo_root_from_nested_app_path():
    bridge = REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"

    assert "ROOT_DIR = Path(__file__).resolve().parents[3]" in bridge.read_text()
    assert 'default="Put the banana in the blue plate."' in bridge.read_text()


def test_lingbot_condition_state_does_not_workspace_clamp_feedback():
    bridge = REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_ee_bridge.py"
    text = bridge.read_text()
    match = re.search(r"def _normalize_condition_action\(.*?\n    def _current_absolute_action8", text, flags=re.S)

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
    assert config.server.prompt == "Put the banana in the blue plate."
    assert config.execution.execute is True
    assert config.execution.step_mode is False
    assert config.execution.step_actions is False
    assert config.execution.max_model_calls == 36
    assert config.execution.execute_frames == 4
    assert config.execution.lingbot_action_per_frame == 4
    assert config.policy_server.checkpoint.name == "checkpoint_step_500"
    assert config.policy_server.expected_weight_size_bytes == 10_177_831_668
    assert config.policy_server.height == 256
    assert config.policy_server.width == 256
    assert config.policy_server.text_encoder_device == "cpu"
    assert config.eef.action_pose_mode == "episode-relative"
    assert config.eef.orientation_mode == "hold-current"
    assert config.eef.xyz_min == (0.06, -0.27, 0.06)
    assert config.servo.cache_actual_feedback is False
    assert config.gripper.command_mode == "continuous"
    assert config.gripper.stroke_max_mm == 80.0
    assert config.cameras.front_observation_key == "observation.images.front"
    assert config.cameras.wrist_observation_key == "observation.images.wrist"
    assert args[args.index("--cmd-pose-topic") + 1] == "/a1_ee_target"
    assert args[args.index("--cam1-device") + 1].startswith("/dev/v4l/by-id/")
    assert "--execute" in args
    assert "--no-step-mode" in args
    assert "--step-actions" not in args
    assert "--no-cache-actual-feedback" in args


def test_lingbot_runtime_manages_the_tracked_step500_server():
    runtime = (REPO / "scripts" / "apps" / "lingbot" / "a1_lingbot_runtime.sh").read_text()
    server = (REPO / "scripts" / "apps" / "lingbot" / "lingbot_va_policy_server.py").read_text()

    assert "start_model_server" in runtime
    assert "MODEL_EXPECTED_WEIGHT_SIZE" in runtime
    assert "lingbot_va_policy_server.py" in runtime
    assert 'VA_CONFIGS["a1_step500"]' in server
    assert "job.inverse_used_action_channel_ids" in server


def test_lingbot_bridge_dependencies_are_declared_and_checked_before_hardware_startup():
    project = (REPO / "pyproject.toml").read_text()
    runtime = (REPO / "scripts" / "apps" / "lingbot" / "a1_lingbot_runtime.sh").read_text()
    pipeline = re.search(r"start_pipeline\(\).*?\n}\n", runtime, flags=re.S)

    assert '"msgpack>=1.1.0,<2.0.0"' in project
    assert pipeline is not None
    assert pipeline.group(0).index("check_bridge_environment") < pipeline.group(0).index("start_model_server")


def test_lingbot_bridge_guard_stops_runtime_when_bridge_exits(tmp_path):
    guard = REPO / "scripts" / "apps" / "lingbot" / "a1_lingbot_bridge_guard.sh"
    runtime_log = tmp_path / "runtime.log"
    fake_runtime = tmp_path / "base-runtime"
    fake_runtime.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{runtime_log}"\n')
    fake_runtime.chmod(0o755)

    result = subprocess.run(
        [str(guard), str(fake_runtime), "lingbot-test-session-does-not-exist", "--", "bash", "-c", "exit 7"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert runtime_log.read_text().splitlines() == ["stop"]
    assert "BRIDGE_EXIT=7" in result.stdout


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
