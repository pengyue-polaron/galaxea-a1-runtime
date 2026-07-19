from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.apps.teleop.reset_config import load_home_pose
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.teleop.config import load_teleop_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/poses/a1_so100_collection_start.toml"
A1_CONFIG = REPO / "configs/poses/a1_collection_start.toml"
SYSTEM_CONFIG = REPO / "configs/system/a1.toml"
TELEOP_CONFIG = REPO / "configs/teleop/a1_so100.toml"


def _load_pose(path=CONFIG):
    teleop = load_teleop_config(TELEOP_CONFIG, repo_root=REPO)
    return load_home_pose(path, teleop=teleop)


def _load_a1_pose(path=A1_CONFIG):
    system = load_system_config(SYSTEM_CONFIG, repo_root=REPO)
    return load_a1_home_pose(path, system=system, repo_root=REPO)


def test_reset_pose_derives_hardware_identity_and_joint_schema_from_teleop():
    pose = _load_pose()

    assert pose.a1.names == tuple(f"arm_joint{index}" for index in range(1, 7))
    assert pose.a1.path == REPO / "configs/poses/a1_collection_start.toml"
    assert pose.leader.config.port.startswith("/dev/serial/by-id/")
    assert pose.leader.config.use_degrees is True
    assert pose.leader.config.motor_write_retries == 5
    assert tuple(pose.leader.action) == (
        "joint0.pos",
        "joint1.pos",
        "joint2.pos",
        "joint3.pos",
        "joint4.pos",
        "joint5.pos",
        "gripper.pos",
    )


def test_reset_pose_rejects_unknown_keys(tmp_path):
    path = tmp_path / "pose.toml"
    path.write_text(CONFIG.read_text() + "\nunexpected = true\n")

    with pytest.raises(ValueError):
        _load_pose(path)


def test_shared_a1_reset_pose_maps_every_motion_and_topic_owner():
    system = load_system_config(SYSTEM_CONFIG, repo_root=REPO)
    pose = _load_a1_pose()

    assert pose.names == system.joint_safety.names
    assert pose.positions == (
        0.012999534606933594,
        0.0010004043579101562,
        -0.07400035858154297,
        -1.569000244140625,
        0.33699989318847656,
        -0.048999786376953125,
    )
    assert pose.topics.joint_states == system.topics.joint_states
    assert pose.topics.target == system.topics.joint_target
    assert pose.topics.staged_command == system.topics.staged_command
    assert pose.topics.relay_enable == system.topics.motion_enable
    assert pose.topics.relay_status == system.topics.relay_status
    assert pose.topics.gripper_target == system.topics.gripper_target
    assert pose.motion.hz == 30.0
    assert pose.motion.min_duration_s == 6.0
    assert pose.motion.max_velocity_rad_s == 0.3
    assert pose.motion.tracker_alignment_timeout_s == 30.0
    assert (
        pose.motion.tracker_alignment_tolerance_rad
        == system.joint_safety.initial_alignment_tolerance_rad
    )
    assert pose.motion.relay_enable_timeout_s == system.relay.enable_timeout_s
    assert pose.motion.max_relay_status_age_s == system.relay.max_status_age_s
    assert pose.motion.hold_s == 1.0
    assert pose.motion.goal_tolerance_rad == 0.08
    assert pose.gripper.enabled is True
    assert pose.gripper.closed_stroke_mm == 0.0
    assert pose.gripper.publish_hz == 10.0
    assert pose.gripper.publish_s == 1.0


def test_shared_a1_reset_pose_rejects_unknown_keys_and_out_of_limit_targets(
    tmp_path,
):
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(A1_CONFIG.read_text() + "\nunexpected = true\n")
    with pytest.raises(ValueError, match="unexpected"):
        _load_a1_pose(unknown)

    outside = tmp_path / "outside.toml"
    outside.write_text(
        A1_CONFIG.read_text().replace(
            "0.012999534606933594,",
            "99.0,",
            1,
        )
    )
    with pytest.raises(ValueError, match="violates system limits"):
        _load_a1_pose(outside)
