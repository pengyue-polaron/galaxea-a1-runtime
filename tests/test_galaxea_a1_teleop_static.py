from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_teleop_runtime_reads_tracked_config_and_disables_extra_args():
    runtime = (REPO / "scripts/apps/teleop/a1_teleop_runtime.sh").read_text()

    assert "configs/teleop/a1_so100.toml" in runtime
    assert "galaxea_a1_runtime.teleop.config" in runtime
    assert '"${BRIDGE_ARGS[@]}"' in runtime
    assert '"${COLLECT_ARGS[@]}"' in runtime
    assert "configs/poses/a1_initial.toml" in runtime
    assert "a1_home.py" in runtime
    assert "Using teleop config:" in runtime
    assert "A1_TELEOP_CONFIG" not in runtime
    assert 'A1_SERIAL="${SERIAL}"' in runtime
    assert 'A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}"' in runtime
    assert 'A1_TRACKER_NODE="/jointTracker_demo_node"' in runtime
    assert "Per-run teleop collector args are disabled" in runtime
    assert "A1_TELEOP_BRIDGE_EXTRA_ARGS" not in runtime
    assert "A1_TELEOP_COLLECT_EXTRA_ARGS" not in runtime


def test_home_pose_command_is_tracked_and_uses_relay_path():
    justfile = (REPO / "Justfile").read_text()
    config = REPO / "configs/poses/a1_initial.toml"
    home = (REPO / "scripts/runtime/a1_home.py").read_text()

    assert "home:" in justfile
    assert "a1_teleop_runtime.sh home" in justfile
    assert config.is_file()
    config_text = config.read_text()
    assert "/arm_joint_target_position" in config_text
    assert "/gripper_position_control_host" in config_text
    assert "closed_stroke_mm = 0.0" in config_text
    assert "[leader]" in config_text
    assert "joint0.pos" in config_text
    assert "gripper.pos" in config_text
    assert "rospy.Publisher(pose.topics.target, JointState" in home
    assert "rospy.Publisher(pose.topics.relay_enable, Bool" in home
    assert "gripper_position_control" in home
    assert "closing A1 gripper" in home
    assert "A1SOLeader" in home
    assert "leader.enable_torque()" in home
    assert "leader.disable_torque()" in home
    assert "relay disabled" in home


def test_teleop_collector_uses_canonical_config_args_only():
    collector = (REPO / "scripts/apps/teleop/teleop_collect.py").read_text()

    assert "default=StateMode.JOINT.value" in collector
    assert '"--ready-timeout-s", type=float' in collector
    assert '"--cam1-device", default="auto"' in collector
    assert '"--joint-wait-timeout-s"' not in collector
    assert '"--cam1-index"' not in collector
    assert "default=200.0" in collector


def test_teleop_bridge_uses_canonical_gripper_topic_flag():
    bridge = (REPO / "scripts/apps/teleop/so100_joint_bridge.py").read_text()

    assert '"--gripper-topic", default="/gripper_position_control_host"' in bridge
    assert '"--staged-command-topic", default="/arm_joint_command_a1_staged"' in bridge
    assert "wait_for_staged_alignment(" in bridge
    assert '"--initial-alignment-tolerance", type=float, default=0.05' in bridge
    assert '"--gripper-position-topic"' not in bridge
    assert '"--gripper-max-stroke-mm", type=float, default=200.0' in bridge


def test_a1_so_leader_keeps_custom_six_axis_shape_outside_third_party():
    leader = (REPO / "galaxea_a1_runtime/teleop/a1_so_leader.py").read_text()
    vendored = (
        REPO / "third_party/lerobot/src/lerobot/teleoperators/so_leader/so_leader.py"
    ).read_text()

    for index in range(6):
        assert f'"joint{index}": Motor({index}, "sts3215", norm_mode_body)' in leader
    assert '"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)' in leader
    assert '"shoulder_pan": Motor(' in vendored
    assert '"joint0": Motor(' not in vendored


def test_teleop_bridge_uses_first_party_a1_leader_adapter():
    bridge = (REPO / "scripts/apps/teleop/so100_joint_bridge.py").read_text()

    assert "from galaxea_a1_runtime.teleop.a1_so_leader import A1SOLeader" in bridge
    assert "leader = A1SOLeader(" in bridge


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


def test_a1_sdk_python_messages_precede_local_ros_overlay():
    files = [
        REPO / "scripts/runtime/a1_runtime.sh",
        REPO / "scripts/apps/teleop/a1_teleop_runtime.sh",
        REPO / "scripts/apps/lingbot/a1_lingbot_runtime.sh",
    ]
    for path in files:
        text = path.read_text()
        assert (
            "third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay"
            in text
        )

    python_files = [
        REPO / "scripts/runtime/eef_nudge.py",
        REPO / "scripts/apps/teleop/so100_joint_bridge.py",
        REPO / "scripts/apps/teleop/teleop_collect.py",
        REPO / "scripts/apps/lingbot/lingbot_va_ee_bridge.py",
    ]
    for path in python_files:
        text = path.read_text()
        assert text.index('str(_A1_SDK / "lib" / "python3" / "dist-packages")') < text.index(
            "str(_ROS1_OVERLAY)"
        )
