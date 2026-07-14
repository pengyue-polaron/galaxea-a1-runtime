from pathlib import Path

from galaxea_a1_runtime.apps.act.config import bash_config, bridge_argv, load_act_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "inference" / "a1_act_joint.toml"


def _load_config_with_temp_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    text = CONFIG.read_text()
    text = text.replace(
        'checkpoint = "models/checkpoints/act/a1_agentview_square/latest"',
        f'checkpoint = "{checkpoint}"',
    )
    path = tmp_path / "a1_act_joint.toml"
    path.write_text(text)
    return load_act_config(path, repo_root=REPO)


def test_act_config_locks_safe_runtime_defaults(tmp_path):
    config = _load_config_with_temp_checkpoint(tmp_path)

    assert config.session.tmux == "act-a1"
    assert config.policy.checkpoint.name == "pretrained_model"
    assert config.policy.deployment_ready is False
    assert config.execution.execute is False
    assert config.execution.step_mode is True
    assert config.execution.execute_steps_per_inference == 100
    assert config.execution.max_model_calls == 0
    assert config.topics.target == "/arm_joint_target_position"
    assert config.topics.staged_command == "/arm_joint_command_a1_staged"
    assert config.topics.motion_enable == "/a1_arm_motion_enable"
    assert config.safety.target_joint_names == (
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    )
    assert config.safety.max_first_target_delta_rad == 10.0
    assert config.safety.initial_alignment_tolerance_rad == 0.05
    assert config.gripper.command_mode == "continuous"
    assert config.gripper.stroke_min_mm == 0.0
    assert config.gripper.stroke_max_mm == 80.0
    assert config.gripper.command_open_threshold == 0.5
    assert config.cameras.front_crop is not None
    assert config.cameras.front_crop.xywh == (103, 0, 480, 480)


def test_act_config_points_at_new_square_agentview_slot():
    text = CONFIG.read_text()

    assert "models/checkpoints/act/a1_agentview_square/latest" in text


def test_act_bridge_args_include_safe_topics_and_dry_run_flag(tmp_path):
    args = bridge_argv(_load_config_with_temp_checkpoint(tmp_path))

    assert args[args.index("--target-topic") + 1] == "/arm_joint_target_position"
    assert args[args.index("--staged-command-topic") + 1] == "/arm_joint_command_a1_staged"
    assert args[args.index("--motion-enable-topic") + 1] == "/a1_arm_motion_enable"
    assert "--no-execute" in args
    assert "--step-mode" in args
    assert args[args.index("--execute-steps-per-inference") + 1] == "100"
    assert args[args.index("--max-model-calls") + 1] == "0"
    assert args[args.index("--gripper-command-mode") + 1] == "continuous"
    assert args[args.index("--gripper-stroke-max") + 1] == "80"
    assert "--disable-backbone-download" in args
    assert args[args.index("--cam0-serial") + 1] == "341522300456"
    assert args[args.index("--cam1-backend") + 1] == "realsense"
    assert args[args.index("--cam1-serial") + 1] == "218622276998"
    assert "--cam0-crop-enabled" in args
    assert args[args.index("--cam0-crop-x") + 1] == "103"
    assert args[args.index("--cam0-crop-width") + 1] == "480"
    assert "--web-preview" in args


def test_act_bash_config_exports_joint_runtime_environment(tmp_path):
    text = bash_config(_load_config_with_temp_checkpoint(tmp_path))

    assert "SESSION=act-a1" in text
    assert "PREFIX=a1-act" in text
    assert "TARGET_TOPIC=/arm_joint_target_position" in text
    assert "STAGED_TOPIC=/arm_joint_command_a1_staged" in text
    assert "RELAY_ENABLE_TOPIC=/a1_arm_motion_enable" in text
    assert "BRIDGE_ARGS=(" in text
    assert "DEPLOYMENT_READY=0" in text
    assert "--no-execute" in text
    assert "--step-mode" in text
    assert "--gripper-command-mode continuous" in text


def test_act_runtime_refuses_unreviewed_deployment():
    runtime = (REPO / "scripts/apps/act/a1_act_joint_runtime.sh").read_text()

    assert '"${DEPLOYMENT_READY}" != "1"' in runtime
