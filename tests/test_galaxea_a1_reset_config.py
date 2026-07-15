from pathlib import Path

import pytest

from galaxea_a1_runtime.apps.teleop.reset_config import load_home_pose
from galaxea_a1_runtime.teleop.config import load_teleop_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/poses/a1_so100_collection_start.toml"
TELEOP_CONFIG = REPO / "configs/teleop/a1_so100.toml"


def _load_pose(path=CONFIG):
    teleop = load_teleop_config(TELEOP_CONFIG, repo_root=REPO)
    return load_home_pose(path, teleop=teleop)


def test_reset_pose_derives_hardware_identity_and_joint_schema_from_teleop():
    pose = _load_pose()

    assert pose.names == tuple(f"arm_joint{index}" for index in range(1, 7))
    assert pose.leader.config.port.startswith("/dev/serial/by-id/")
    assert pose.leader.config.use_degrees is True
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
