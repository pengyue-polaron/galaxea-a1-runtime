from dataclasses import replace
from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.teleop.dataset_contract import tracked_config_reference
from galaxea_a1_runtime.teleop.config import (
    load_teleop_config,
    validate_collection_config,
    validate_teleop_config,
)
from galaxea_a1_runtime.teleop.config_runtime import bash_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/teleop/a1_so100.toml"
LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A016967-if00"


def test_default_teleop_config_locks_continuous_gripper_contract():
    config = load_teleop_config(CONFIG, repo_root=REPO)

    assert config.leader.port == LEADER_PORT
    assert config.leader.id == "my_leader"
    assert config.leader.motor_write_retries == 5
    assert config.reset.config == REPO / "configs/poses/a1_so100_collection_start.toml"
    assert config.runtime.bridge_startup_timeout_s == 65.0
    assert config.runtime.bridge_stop_timeout_s == 5.0
    assert config.collection.dataset_root == REPO / "data/datasets"
    assert config.collection.repo_id_prefix == "pengyue-polaron/galaxea-a1"
    assert config.system.joint_safety.names == (
        "arm_joint1",
        "arm_joint2",
        "arm_joint3",
        "arm_joint4",
        "arm_joint5",
        "arm_joint6",
    )
    assert config.bridge.mapping.sign == (-1.0, 1.0, 1.0, -1.0, 1.0, -1.0)
    assert config.system.robot_service.endpoint == (
        "unix:///tmp/galaxea-a1-runtime/robot-service.sock"
    )
    assert config.system.robot_service.device_connect_timeout_s == 30.0
    assert config.system.robot_service.rpc_timeout_s == 0.5
    assert config.system.robot_service.command_timeout_s == 0.75
    assert config.system.robot_service.lease_timeout_s == 1.0
    assert config.system.joint_safety.initial_alignment_tolerance_rad == 0.05
    assert (config.gripper.source_min, config.gripper.source_max) == (0.0, 53.16)
    assert config.gripper.saturate_out_of_range is True
    assert config.system.gripper.stroke_min_mm == 0.0
    assert config.system.gripper.stroke_max_mm == 104.0
    assert config.system.relay.gripper_ignored_error_mask == 8
    assert config.system.topics.gripper_target == "/a1_gripper_target"
    assert config.system.cameras.front.depth is False
    assert config.system.cameras.front.backend == "realsense"
    assert config.system.cameras.front.serial == "341522300456"
    assert config.system.cameras.front.align_depth_to_color is None
    assert config.system.cameras.front.require_usb3 is False
    assert config.system.cameras.front.crop is not None
    assert config.system.cameras.front.crop.xywh == (103, 0, 480, 480)
    assert config.system.cameras.max_age_s == 0.5
    assert config.system.joint_safety.max_feedback_age_s == 0.5
    assert config.system.eef.max_feedback_age_s == 0.5
    assert config.system.eef.xyz_min == pytest.approx((0.04, -0.27, 0.06))
    assert config.system.eef.xyz_max == pytest.approx((0.47, 0.17, 0.50))
    assert config.collection.auto_reset_after_save is True
    assert config.collection.auto_reset_after_discard is True
    assert config.collection.max_joint_action_step_rad == 0.35
    assert config.system.cameras.wrist.backend == "realsense"
    assert config.system.cameras.wrist.serial == "218622276998"
    assert config.system.cameras.wrist.crop is None
    assert config.system.web_preview.enabled is True
    assert config.system.web_preview.bind == "0.0.0.0"
    assert config.system.web_preview.startup_timeout_s == 15.0
    assert config.system.web_preview.shutdown_timeout_s == 5.0
    assert config.system.camera_diagnostics.output_root == (
        REPO / "outputs/camera_diagnostics"
    )
    assert config.system.camera_diagnostics.frame_timeout_s == 5.0
    assert config.system.camera_diagnostics.rate_probe_s == 2.0
    assert config.system.camera_diagnostics.jpeg_quality == 95
    assert config.system.startup.tmux_process_grace_s == 4


def test_teleop_shell_contract_renders_lifecycle_values():
    rendered = bash_config(load_teleop_config(CONFIG, repo_root=REPO))

    assert "BRIDGE_STARTUP_TIMEOUT_S=65" in rendered
    assert "BRIDGE_STOP_TIMEOUT_S=5" in rendered
    assert (
        "A1_ROBOT_SERVICE_ENDPOINT=unix:///tmp/galaxea-a1-runtime/robot-service.sock"
        in rendered
    )
    assert "JOINT_TRACKER_NODE=/jointTracker_demo_node" in rendered
    assert "JOINT_TRACKER_NODE_NAME=jointTracker_demo_node" in rendered


def test_teleop_config_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "teleop.toml"
    path.write_text(CONFIG.read_text() + "\n[unexpected]\nvalue = true\n")

    with pytest.raises(ValueError):
        load_teleop_config(path, repo_root=REPO)


def test_teleop_config_rejects_fractional_collection_fps(tmp_path: Path):
    path = tmp_path / "teleop.toml"
    path.write_text(CONFIG.read_text().replace("fps = 30.0", "fps = 29.97"))

    with pytest.raises(ValueError, match="integer for LeRobot recording"):
        load_teleop_config(path, repo_root=REPO)


def test_formal_collection_config_reference_must_be_portable(tmp_path: Path):
    config = load_teleop_config(CONFIG, repo_root=REPO)

    assert tracked_config_reference(config, repo_root=REPO) == (
        "configs/teleop/a1_so100.toml"
    )
    with pytest.raises(ValueError, match="tracked inside the repository"):
        tracked_config_reference(
            replace(config, path=tmp_path / "external.toml"),
            repo_root=REPO,
        )


def test_realsense_is_required_by_collection_not_general_teleop():
    config = load_teleop_config(CONFIG, repo_root=REPO)
    alternate_front = replace(config.system.cameras.front, backend="opencv")
    alternate_cameras = replace(config.system.cameras, front=alternate_front)
    alternate = replace(
        config,
        system=replace(config.system, cameras=alternate_cameras),
    )

    validate_teleop_config(alternate)
    with pytest.raises(ValueError, match="canonical Teleop collection"):
        validate_collection_config(alternate)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("motor_write_retries = 5", "motor_write_retries = 0", "must be at least 1"),
        (
            "bridge_startup_timeout_s = 65.0",
            "bridge_startup_timeout_s = 15.0",
            "must cover two System robot_service.device_connect_timeout_s",
        ),
    ],
)
def test_teleop_config_rejects_unsupported_plugin_contracts(
    tmp_path: Path, old: str, new: str, message: str
):
    path = tmp_path / "teleop.toml"
    path.write_text(CONFIG.read_text().replace(old, new))

    with pytest.raises(ValueError, match=message):
        load_teleop_config(path, repo_root=REPO)
