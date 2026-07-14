from pathlib import Path

from galaxea_a1_runtime.apps.act.config import bridge_argv, load_act_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "deployments" / "act_joint.toml"


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
    assert config.safety.action_step_guard_enabled is False
    assert config.safety.initial_alignment_tolerance_rad == 0.05
    assert config.gripper.stroke_min_mm == 0.0
    assert config.gripper.stroke_max_mm == 100.0
    assert config.topics.gripper_target == "/a1_gripper_target"
    assert config.cameras.front_crop is not None
    assert config.cameras.front_crop.xywh == (103, 0, 480, 480)


def test_act_bridge_args_include_safe_topics_and_dry_run_flag(tmp_path):
    args = bridge_argv(_load_config_with_temp_checkpoint(tmp_path))

    assert args[args.index("--target-topic") + 1] == "/arm_joint_target_position"
    assert args[args.index("--staged-command-topic") + 1] == "/arm_joint_command_a1_staged"
    assert args[args.index("--motion-enable-topic") + 1] == "/a1_arm_motion_enable"
    assert "--no-execute" in args
    assert "--no-action-step-guard-enabled" in args
    assert "--step-mode" in args
    assert args[args.index("--execute-steps-per-inference") + 1] == "100"
    assert args[args.index("--max-model-calls") + 1] == "0"
    assert args[args.index("--gripper-stroke-max") + 1] == "100"
    assert args[args.index("--gripper-target-topic") + 1] == "/a1_gripper_target"
    assert "--gripper-command-mode" not in args
    assert "--disable-backbone-download" in args
    assert args[args.index("--cam0-serial") + 1] == "341522300456"
    assert args[args.index("--cam1-backend") + 1] == "realsense"
    assert args[args.index("--cam1-serial") + 1] == "218622276998"
    assert "--cam0-crop-enabled" in args
    assert args[args.index("--cam0-crop-x") + 1] == "103"
    assert args[args.index("--cam0-crop-width") + 1] == "480"
    assert "--web-preview" in args
