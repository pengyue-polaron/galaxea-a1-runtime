from galaxea_a1_runtime.runtime.safety_report import (
    build_architecture_findings,
    build_safety_settings,
    format_safety_report,
    safety_report_as_dict,
)


def test_safety_report_discloses_non_obvious_motion_controls():
    settings = {item.name: item for item in build_safety_settings()}

    assert "joint_tracking_limiter" not in settings
    assert "initial_command_alignment" in settings
    assert "generic_ros1_adapter_arm_motion" not in settings
    assert "generic_ros1_gripper_range_check" not in settings
    assert "lingbot_xyz_delta_clamp" not in settings
    assert "eef_policy_workspace_bounds" in settings
    assert (
        "rejected without publication"
        in settings["eef_policy_workspace_bounds"].behavior
    )
    assert settings["eef_policy_task_selection"].default == "6 tracked prompts"
    assert "before model" in settings["eef_policy_task_selection"].behavior
    assert "train/OOD" in settings["eef_policy_task_selection"].visibility
    assert "lingbot_eef_servo_compensation" not in settings
    assert "pi05_eef_servo_compensation" not in settings
    assert "eef_policy_ik" in settings
    assert "bounded IK" in settings["safe_command_path"].path
    assert "position_tolerance=0.002m" in settings["eef_policy_ik"].default
    assert "max_joint_delta=1.5rad" in settings["eef_policy_ik"].default
    assert "eef_policy_relay_status_guard" in settings
    assert settings["gripper_position_jump_compatibility"].default == "mask=8"
    assert settings["lingbot_execution_gate"].default == (
        "execute=true, step_mode=false, step_actions=false, max_model_calls=66"
    )
    assert settings["pi05_execution_gate"].default == (
        "execute=true, step_mode=false, step_actions=false, max_model_calls=53"
    )
    for name in ("lingbot_execution_gate", "pi05_execution_gate"):
        assert "rollout cadence" in settings[name].operator_note
        assert "finite model-call cap" in settings[name].operator_note
    assert settings["teleop_gripper_mapping"].default == (
        "leader=[0,53.16], invert=false"
    )
    assert settings["gripper_scale_mapping"].default.startswith("continuous 0..1")
    assert (
        "does not modify commands"
        in settings["initial_command_alignment"].operator_note
    )
    assert "staged current-joint hold" in settings["initial_command_alignment"].behavior
    assert "sole owner" in settings["eef_policy_ik"].operator_note


def test_safety_report_has_text_and_json_forms():
    text = format_safety_report()
    payload = safety_report_as_dict()

    assert "Galaxea A1 Runtime Safety Report" in text
    assert "Architecture findings" in text
    assert payload["settings"]
    assert payload["architecture_findings"] == list(build_architecture_findings())
