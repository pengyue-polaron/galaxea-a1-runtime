from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_act_bridge_uses_joint_target_path_and_relay_guard():
    bridge = (REPO / "scripts" / "apps" / "act" / "act_joint_policy_bridge.py").read_text()

    assert 'parser.add_argument("--target-topic", default="/arm_joint_target_position")' in bridge
    assert 'parser.add_argument("--staged-command-topic", default="/arm_joint_command_a1_staged")' in bridge
    assert 'parser.add_argument("--motion-enable-topic", default="/a1_arm_motion_enable")' in bridge
    assert "rospy.Publisher(args.target_topic, JointState" in bridge
    assert "rospy.Publisher(args.motion_enable_topic, Bool" in bridge
    assert "_wait_for_staged_alignment(current_joints)" in bridge
    assert "_require_relay_active()" in bridge
    assert "predict_action_chunk" in bridge
    assert "/usr/lib/python3/dist-packages" not in bridge
    assert 'default="/arm_joint_command_host"' not in bridge
    assert "rospy.Publisher('/arm_joint_command_host'" not in bridge
    assert 'rospy.Publisher("/arm_joint_command_host"' not in bridge


def test_act_runtime_wrapper_uses_tracked_config_and_joint_runtime():
    runtime = (REPO / "scripts" / "apps" / "act" / "a1_act_joint_runtime.sh").read_text()

    assert "configs/inference/act_joint_a1.toml" in runtime
    assert "galaxea_a1_runtime.apps.act.config" in runtime
    assert 'JOINT_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"' in runtime
    assert '"${BRIDGE_ARGS[@]}"' in runtime
    assert 'A1_JOINT_TARGET_TOPIC="${TARGET_TOPIC}"' in runtime
    assert 'A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}"' in runtime
    assert 'A1_TRACKER_NODE="/jointTracker_demo_node"' in runtime
    assert "leader" not in runtime.lower()
    assert "A1_ACT_BRIDGE_EXTRA_ARGS" not in runtime


def test_joint_runtime_starts_joint_tracker_and_locked_relay():
    runtime = (REPO / "scripts" / "runtime" / "a1_joint_runtime.sh").read_text()

    assert "joint_tracker_staged.launch" in runtime
    assert "target_topic:=${TARGET_TOPIC}" in runtime
    assert "safe_arm_command_relay.py" in runtime
    assert "A1_JOINT_TARGET_TOPIC" in runtime
    assert "A1_STAGED_COMMAND_TOPIC" in runtime
    assert "ee_tracker_staged.launch" not in runtime


def test_justfile_exposes_act_and_stops_it():
    justfile = (REPO / "Justfile").read_text()

    assert "act:" in justfile
    assert "scripts/apps/act/a1_act_joint_runtime.sh start" in justfile
    assert "scripts/apps/act/a1_act_joint_runtime.sh stop" in justfile
    assert "scripts/runtime/a1_joint_runtime.sh stop" in justfile
