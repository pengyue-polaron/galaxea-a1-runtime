from pathlib import Path
import subprocess

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
    assert config.gripper.command_mode == "continuous"
    assert config.gripper.stroke_max_mm == 80.0
    assert config.cameras.front_crop is not None
    assert config.cameras.front_crop.xywh == (103, 0, 480, 480)
    assert args[args.index("--cmd-pose-topic") + 1] == "/a1_ee_target"
    assert args[args.index("--cam1-backend") + 1] == "realsense"
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
