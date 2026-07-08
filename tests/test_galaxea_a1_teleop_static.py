from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_teleop_runtime_reads_tracked_config_and_disables_extra_args():
    runtime = (REPO / "scripts/apps/teleop/a1_teleop_runtime.sh").read_text()

    assert "configs/teleop/a1_so100.toml" in runtime
    assert "galaxea_a1_runtime.teleop.config" in runtime
    assert '"${BRIDGE_ARGS[@]}"' in runtime
    assert '"${COLLECT_ARGS[@]}"' in runtime
    assert "Per-run teleop collector args are disabled" in runtime
    assert "A1_TELEOP_BRIDGE_EXTRA_ARGS" not in runtime
    assert "A1_TELEOP_COLLECT_EXTRA_ARGS" not in runtime


def test_teleop_collector_keeps_old_default_and_cli_aliases():
    collector = (REPO / "scripts/apps/teleop/teleop_collect.py").read_text()

    assert "default=StateMode.JOINT.value" in collector
    assert '"--joint-wait-timeout-s"' in collector
    assert '"--cam1-index"' in collector
    assert "default=200.0" in collector


def test_teleop_bridge_accepts_old_gripper_topic_flag():
    bridge = (REPO / "scripts/apps/teleop/so100_joint_bridge.py").read_text()

    assert '"--gripper-position-topic"' in bridge
    assert '"--gripper-max-stroke-mm", type=float, default=200.0' in bridge


def test_vendored_lerobot_so_leader_keeps_a1_custom_six_axis_shape():
    leader = (
        REPO / "third_party/lerobot/src/lerobot/teleoperators/so_leader/so_leader.py"
    ).read_text()

    for index in range(6):
        assert f'"joint{index}": Motor({index}, "sts3215", norm_mode_body)' in leader
    assert '"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)' in leader
    assert '"shoulder_pan": Motor(' not in leader


def test_teleop_camera_io_is_shared_and_snapshot_command_exists():
    collector = (REPO / "scripts/apps/teleop/teleop_collect.py").read_text()
    runtime = (REPO / "scripts/apps/teleop/a1_teleop_runtime.sh").read_text()
    snapshot = REPO / "scripts/apps/teleop/camera_snapshot.py"

    assert "from galaxea_a1_runtime.hardware.cameras import" in collector
    assert "class RealSenseCamera" not in collector
    assert "class OpenCVCamera" not in collector
    assert snapshot.is_file()
    assert "camera_snapshot.py" in runtime
    assert "cameras)" in runtime
    assert "cam0_depth" in snapshot.read_text()
