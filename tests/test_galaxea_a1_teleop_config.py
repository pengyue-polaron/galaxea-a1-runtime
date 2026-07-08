from pathlib import Path

from galaxea_a1_runtime.collection import StateMode
from galaxea_a1_runtime.teleop.config import bridge_argv, collect_argv, load_teleop_config


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/teleop/a1_so100.toml"


def test_default_teleop_config_locks_old_working_behavior():
    config = load_teleop_config(CONFIG, repo_root=REPO)

    assert config.leader.port == "/dev/ttyACM0"
    assert config.leader.id == "my_leader"
    assert config.collection.state_mode == StateMode.JOINT
    assert config.bridge.mapping.relative is True
    assert config.bridge.mapping.sign == (-1.0, 1.0, 1.0, -1.0, 1.0, -1.0)
    assert config.gripper.max_stroke_mm == 200.0


def test_config_builds_bridge_args_without_per_run_env_overrides():
    config = load_teleop_config(CONFIG, repo_root=REPO)
    args = bridge_argv(config)

    assert args[args.index("--leader-port") + 1] == "/dev/ttyACM0"
    assert args[args.index("--target-topic") + 1] == "/arm_joint_target_position"
    assert args[args.index("--gripper-max-stroke-mm") + 1] == "200"
    assert args[args.index("--sign") + 1] == "-1,1,1,-1,1,-1"


def test_config_builds_collector_args_from_tracked_file():
    config = load_teleop_config(CONFIG, repo_root=REPO)
    args = collect_argv(config)

    assert args[args.index("--data-root") + 1] == str(REPO / "data/raw")
    assert args[args.index("--state-mode") + 1] == "joint"
    assert args[args.index("--gripper-stroke-scale") + 1] == "200"
    assert args[args.index("--cam1-device") + 1] == "auto"
