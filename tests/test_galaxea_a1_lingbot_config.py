from pathlib import Path
import subprocess

import pytest

from galaxea_a1_runtime.apps.lingbot.config import bridge_argv, load_lingbot_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "deployments" / "lingbot_va.toml"


def test_lingbot_deployment_composes_with_shared_system_config():
    config = load_lingbot_config(CONFIG, repo_root=REPO)
    args = bridge_argv(config)

    assert config.system.path == REPO / "configs/system/a1.toml"
    assert config.server.prompt == "REPLACE_WITH_NEW_TASK_PROMPT"
    assert config.execution.execute is False
    assert config.policy_server.deployment_ready is False
    assert config.eef.orientation_mode == "hold-current"
    assert config.eef.action_pose_mode == "episode-relative"
    assert config.gripper.stroke_min_mm == 0.0
    assert config.gripper.stroke_max_mm == 100.0
    assert config.policy_server.q01_source == ()
    assert config.policy_server.q99_source == ()
    assert config.cameras.front_crop is not None
    assert config.cameras.front_crop.xywh == (103, 0, 480, 480)
    assert args[args.index("--cmd-pose-topic") + 1] == "/a1_ee_target"
    assert args[args.index("--cam1-backend") + 1] == "realsense"
    assert args[args.index("--gripper-stroke-max") + 1] == "100"
    assert args[args.index("--cmd-gripper-topic") + 1] == "/a1_gripper_target"
    assert "--gripper-command-mode" not in args
    assert "--execute" not in args


def test_lingbot_bridge_guard_stops_runtime_when_bridge_exits(tmp_path):
    guard = REPO / "scripts/apps/lingbot/a1_lingbot_bridge_guard.sh"
    runtime_log = tmp_path / "runtime.log"
    fake_runtime = tmp_path / "base-runtime"
    fake_runtime.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{runtime_log}"\n')
    fake_runtime.chmod(0o755)

    result = subprocess.run(
        [str(guard), str(fake_runtime), "missing-test-session", "--", "bash", "-c", "exit 7"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert runtime_log.read_text().splitlines() == ["stop"]


def test_lingbot_ready_rejects_missing_real_statistics(tmp_path):
    text = CONFIG.read_text().replace("deployment_ready = false", "deployment_ready = true")
    text = text.replace("REPLACE_WITH_NEW_TASK_PROMPT", "pick the object")
    text = text.replace("expected_weight_size_bytes = 0", "expected_weight_size_bytes = 1")
    path = tmp_path / "lingbot.toml"
    path.write_text(text)

    with pytest.raises(ValueError, match="requires real q01/q99"):
        load_lingbot_config(path, repo_root=REPO)


def test_lingbot_ready_accepts_complete_checkpoint_statistics(tmp_path):
    text = CONFIG.read_text().replace("deployment_ready = false", "deployment_ready = true")
    text = text.replace("REPLACE_WITH_NEW_TASK_PROMPT", "pick the object")
    text = text.replace("expected_weight_size_bytes = 0", "expected_weight_size_bytes = 1")
    text = text.replace(
        "q01_source = []",
        "q01_source = [-1, -1, -1, -1, -1, -1, -1, 0]",
    )
    text = text.replace(
        "q99_source = []",
        "q99_source = [1, 1, 1, 1, 1, 1, 1, 1]",
    )
    path = tmp_path / "lingbot.toml"
    path.write_text(text)

    config = load_lingbot_config(path, repo_root=REPO)

    assert config.policy_server.deployment_ready is True
    assert len(config.policy_server.q01_source) == 8
