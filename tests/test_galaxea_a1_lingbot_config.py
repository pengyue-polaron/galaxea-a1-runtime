from pathlib import Path
import json
import re
import subprocess
from types import SimpleNamespace

import pytest

from galaxea_a1_runtime.apps.eef_policy_actions import build_action_transform_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.lingbot import doctor as doctor_module
from galaxea_a1_runtime.apps.lingbot.verify import validate_training_summary


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/deployments/lingbot/fruit_placement_eef.toml"
MODEL = REPO / "configs/models/lingbot/fruit_placement_eef.toml"
CONTRACT = REPO / "configs/models/lingbot/fruit_placement_eef.contract.toml"


def _deployment_copy(
    tmp_path: Path,
    *,
    model_text: str | None = None,
    contract_text: str | None = None,
) -> Path:
    contract_path = tmp_path / "contract.toml"
    contract_path.write_text(contract_text or CONTRACT.read_text())
    descriptor = model_text or MODEL.read_text()
    descriptor = descriptor.replace(
        'contract = "configs/models/lingbot/fruit_placement_eef.contract.toml"',
        f'contract = "{contract_path}"',
    )
    model_path = tmp_path / "model.toml"
    model_path.write_text(descriptor)
    deployment = CONFIG.read_text().replace(
        'config = "configs/models/lingbot/fruit_placement_eef.toml"',
        f'config = "{model_path}"',
    )
    path = tmp_path / "deployment.toml"
    path.write_text(deployment)
    return path


def test_lingbot_deployment_composes_with_shared_system_config():
    config = load_lingbot_config(CONFIG, repo_root=REPO)
    assert config.system.path == REPO / "configs/system/a1.toml"
    assert config.server.prompt == "Put the banana into the blue plate"
    assert config.execution.execute is False
    assert config.policy_server.deployment_ready is True
    assert config.system.eef.orientation_mode == "hold-current"
    assert config.action.pose_mode == "episode-relative"
    assert config.system.gripper.stroke_min_mm == 0.0
    assert config.system.gripper.stroke_max_mm == 104.0
    assert config.execution.kv_observations_per_frame == 4
    assert config.execution.initial_ee_pose is None

    transform = build_action_transform_config(
        system=config.system,
        servo_gain=config.servo.gain,
        servo_max_extra_m=config.servo.max_extra_m,
    )
    assert transform.gripper_stroke_max == config.system.gripper.stroke_max_mm
    assert transform.eef_servo_gain == config.servo.gain
    assert len(config.policy_server.q01_source) == 8
    assert len(config.policy_server.q99_source) == 8
    assert config.policy_server.vendor_config == "a1"
    assert config.policy_server.attention_mode == "torch"
    assert config.policy_server.enable_offload is False
    assert config.policy_server.world_size == 1
    assert config.policy_server.python == (
        REPO / "external/lingbot-va/.env312/bin/python"
    )
    assert config.policy_server.model_root == config.policy_server.model.artifact_root
    assert config.policy_server.checkpoint == config.policy_server.model.artifact_root
    assert config.system.cameras.front.crop is not None
    assert config.system.cameras.front.crop.xywh == (103, 0, 480, 480)


def test_lingbot_bridge_guard_stops_runtime_when_bridge_exits(tmp_path):
    guard = REPO / "scripts/apps/a1_eef_policy_bridge_guard.sh"
    runtime_log = tmp_path / "runtime.log"
    fake_runtime = tmp_path / "base-runtime"
    fake_runtime.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{runtime_log}"\n'
    )
    fake_runtime.chmod(0o755)

    result = subprocess.run(
        [
            str(guard),
            str(fake_runtime),
            "missing-test-session",
            "--",
            "bash",
            "-c",
            "exit 7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 7
    assert runtime_log.read_text().splitlines() == ["stop"]


def test_lingbot_app_doctor_is_independent_of_runtime_ros_checks(
    tmp_path, monkeypatch, capsys
):
    wrist = tmp_path / "wrist-camera"
    wrist.touch()
    config = SimpleNamespace(
        system=SimpleNamespace(
            cameras=SimpleNamespace(
                wrist=SimpleNamespace(backend="v4l2", device=str(wrist))
            )
        ),
        server=SimpleNamespace(host="127.0.0.1", port=8000, connect_timeout_s=0.1),
    )
    monkeypatch.setattr(doctor_module, "load_lingbot_config", lambda *_a, **_k: config)
    monkeypatch.setattr(doctor_module, "websocket_open", lambda *_a, **_k: True)

    result = doctor_module.main(
        ["--config", str(tmp_path / "unused.toml"), "--require-execution", "--json"]
    )

    assert result == 0
    checks = json.loads(capsys.readouterr().out)
    assert [check["name"] for check in checks] == [
        "wrist_camera",
        "lingbot_server",
    ]


def test_lingbot_ready_rejects_missing_real_statistics(tmp_path):
    text = re.sub(r"q01_source = \[.*\]", "q01_source = []", CONTRACT.read_text())
    text = re.sub(r"q99_source = \[.*\]", "q99_source = []", text)
    path = _deployment_copy(tmp_path, contract_text=text)

    with pytest.raises(ValueError, match="requires q01/q99"):
        load_lingbot_config(path, repo_root=REPO)


def test_lingbot_ready_accepts_complete_checkpoint_statistics(tmp_path):
    path = _deployment_copy(tmp_path)

    config = load_lingbot_config(path, repo_root=REPO)

    assert config.policy_server.deployment_ready is True
    assert len(config.policy_server.q01_source) == 8


def test_lingbot_rejects_abbreviated_model_revision(tmp_path):
    text = MODEL.read_text().replace(
        'revision = "90e017bdbc6afac2e441b4634c9192776bbcb8b7"',
        'revision = "90e017b"',
    )
    path = _deployment_copy(tmp_path, model_text=text)

    with pytest.raises(ValueError, match="source.revision.*40-character"):
        load_lingbot_config(path, repo_root=REPO)


def test_lingbot_requires_a_predicted_frame_after_conditioning(tmp_path):
    text = CONTRACT.read_text().replace("frame_chunk_size = 4", "frame_chunk_size = 1")
    path = _deployment_copy(tmp_path, contract_text=text)

    with pytest.raises(ValueError, match="conditioned first frame"):
        load_lingbot_config(path, repo_root=REPO)


def test_lingbot_training_metadata_matches_backend_and_action_contract(tmp_path):
    config = load_lingbot_config(CONFIG, repo_root=REPO)
    policy = config.policy_server
    summary = {
        "checkpoint_step": policy.model.checkpoint_step,
        "code_repository": policy.code_repository.removesuffix(".git"),
        "code_revision": policy.code_revision,
        "source_action_dimension": len(policy.action_channel_ids),
        "model_action_dimension": policy.model_action_dim,
        "used_action_channel_ids": list(policy.action_channel_ids),
        "includes_optimizer_state": False,
    }
    path = tmp_path / "training_summary.json"
    path.write_text(json.dumps(summary))

    validate_training_summary(config, tmp_path)

    summary["code_revision"] = "0" * 40
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="training summary contract mismatch"):
        validate_training_summary(config, tmp_path)
