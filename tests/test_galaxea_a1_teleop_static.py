from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_teleop_runtime_reads_tracked_config_and_disables_extra_args():
    runtime = (REPO / "scripts/apps/teleop/a1_teleop_runtime.sh").read_text()

    assert "configs/teleop/a1_so100.toml" in runtime
    assert "galaxea_a1_runtime.teleop.config" in runtime
    assert '"${BRIDGE_ARGS[@]}"' in runtime
    assert '"${COLLECT_ARGS[@]}"' in runtime
    assert 'reset_live()' in runtime
    assert '--reset-runtime-script "${ROOT}/scripts/apps/teleop/a1_teleop_runtime.sh"' in runtime
    assert "configs/poses/a1_initial.toml" in runtime
    assert "a1_home.py" in runtime
    assert 'step "[Setup] ${CONFIG_PATH}"' in runtime
    assert "_reset-live)" in runtime
    assert "A1_TELEOP_CONFIG" not in runtime
    assert 'A1_SERIAL="${SERIAL}"' in runtime
    assert 'A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}"' in runtime
    assert 'A1_TRACKER_NODE="/jointTracker_demo_node"' in runtime
    assert "Per-run teleop collector args are disabled" in runtime
    assert "bridge_group_has_live_process()" in runtime
    assert '"${PYTHON_BIN}" "${ROOT}/scripts/apps/teleop/so100_joint_bridge.py"' in runtime
    assert 'uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/so100_joint_bridge.py"' not in runtime
    assert "A1_TELEOP_BRIDGE_EXTRA_ARGS" not in runtime
    assert "A1_TELEOP_COLLECT_EXTRA_ARGS" not in runtime


def test_reset_pose_command_is_tracked_and_uses_relay_path():
    justfile = (REPO / "Justfile").read_text()
    config = REPO / "configs/poses/a1_initial.toml"
    reset = (REPO / "scripts/runtime/a1_home.py").read_text()

    assert "reset:" in justfile
    assert "a1_teleop_runtime.sh reset" in justfile
    assert "home:" not in justfile
    assert config.is_file()
    config_text = config.read_text()
    assert "/arm_joint_target_position" in config_text
    assert "/gripper_position_control_host" in config_text
    assert "closed_stroke_mm = 0.0" in config_text
    assert "[leader]" in config_text
    assert "joint0.pos" in config_text
    assert "gripper.pos" in config_text
    assert "rospy.Publisher(pose.topics.target, JointState" in reset
    assert "rospy.Publisher(pose.topics.relay_enable, Bool" in reset
    assert "gripper_position_control" in reset
    assert "A1SOLeader" in reset
    assert "leader.enable_torque()" in reset
    assert "leader.disable_torque()" in reset
    assert "ThreadPoolExecutor" in reset
    assert 'jobs["leader"]' in reset
    assert "class ResetProgress" in reset
    assert '"[Reset] Complete"' in reset
    assert "[home]" not in reset


def test_teleop_collector_uses_canonical_config_args_only():
    collector = (REPO / "scripts/apps/teleop/teleop_collect.py").read_text()

    assert "default=StateMode.EEF_JOINT.value" in collector
    assert '"--ready-timeout-s", type=float' in collector
    assert '"--max-camera-age-s", type=float' in collector
    assert '"--auto-reset-after-save", action=argparse.BooleanOptionalAction' in collector
    assert "reset_for_next_episode(" in collector
    assert "find_joint_action_step_violation(" in collector
    assert "REJECTED: joint action discontinuity" in collector
    assert '"--cam0-require-usb3", action=argparse.BooleanOptionalAction' in collector
    assert '"--cam1-device", default="auto"' in collector
    assert '"--cam1-pixel-format", default="YUYV"' in collector
    assert '"--joint-wait-timeout-s"' not in collector
    assert '"--cam1-index"' not in collector
    assert "default=200.0" in collector
    assert "LatestCameraReader" in collector
    assert "front.read_frameset" in collector
    assert "_wait_for_new_camera_samples(" in collector
    assert "_fresh_camera_sample(" in collector
    assert "recording failed; episode deleted" in collector


def test_teleop_bridge_uses_canonical_gripper_topic_flag():
    bridge = (REPO / "scripts/apps/teleop/so100_joint_bridge.py").read_text()

    assert '"--gripper-topic", default="/gripper_position_control_host"' in bridge
    assert '"--staged-command-topic", default="/arm_joint_command_a1_staged"' in bridge
    assert "wait_for_staged_alignment(" in bridge
    assert '"--initial-alignment-tolerance", type=float, default=0.05' in bridge
    assert '"--gripper-position-topic"' not in bridge
    assert '"--gripper-max-stroke-mm", type=float, default=200.0' in bridge
    assert "signal.signal(signal.SIGTERM, request_stop)" in bridge
    assert "not stop_requested.is_set()" in bridge
    assert "if stop_requested.is_set():" in bridge


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
    assert "class LatestCameraReader" in (REPO / "galaxea_a1_runtime/hardware/cameras.py").read_text()
    assert "class RealSenseCamera" not in collector
    assert "class OpenCVCamera" not in collector
    assert snapshot.is_file()
    assert "camera_snapshot.py" in runtime
    assert "cameras)" in runtime
    assert "cam0_depth" in snapshot.read_text()


def test_hardware_enumeration_check_command_exists():
    justfile = (REPO / "Justfile").read_text()
    script = REPO / "scripts/runtime/a1_hardware_check.py"

    assert "hardware *args:" in justfile
    assert "a1_hardware_check.py" in justfile
    assert script.is_file()
    text = script.read_text()
    assert "load_teleop_config" in text
    assert "realsense_device_info" in text
    assert "resolve_video_source" in text
    assert "so100_joint_bridge" not in text


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
