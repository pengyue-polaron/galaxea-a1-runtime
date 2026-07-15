from pathlib import Path

from galaxea_a1_runtime.apps.act.config import load_act_config


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
    assert config.system.topics.joint_target == "/arm_joint_target_position"
    assert config.system.topics.staged_command == "/arm_joint_command_a1_staged"
    assert config.system.topics.motion_enable == "/a1_arm_motion_enable"
    assert config.system.joint_safety.names == (
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    )
    assert config.system.joint_safety.action_step_guard_enabled is False
    assert config.system.joint_safety.initial_alignment_tolerance_rad == 0.05
    assert config.system.gripper.stroke_min_mm == 0.0
    assert config.system.gripper.stroke_max_mm == 104.0
    assert config.system.topics.gripper_target == "/a1_gripper_target"
    assert config.system.cameras.front.crop is not None
    assert config.system.cameras.front.crop.xywh == (103, 0, 480, 480)
