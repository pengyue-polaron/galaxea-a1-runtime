from galaxea_a1_runtime.safety_report import (
    build_architecture_findings,
    build_safety_settings,
    format_safety_report,
    safety_report_as_dict,
)


def test_safety_report_discloses_non_obvious_motion_controls():
    settings = {item.name: item for item in build_safety_settings()}

    assert {
        "initial_command_alignment",
        "eef_policy_workspace_bounds",
        "eef_policy_task_selection",
        "eef_policy_ik",
        "eef_policy_relay_status_guard",
        "gripper_position_jump_compatibility",
        "lingbot_execution_gate",
        "pi05_execution_gate",
        "teleop_gripper_mapping",
        "gripper_scale_mapping",
    } <= settings.keys()
    assert {
        "joint_tracking_limiter",
        "generic_ros1_adapter_arm_motion",
        "generic_ros1_gripper_range_check",
        "lingbot_xyz_delta_clamp",
        "lingbot_eef_servo_compensation",
        "pi05_eef_servo_compensation",
    }.isdisjoint(settings)
    assert settings["eef_policy_task_selection"].default == "6 tracked prompts"
    assert "position_tolerance=0.003m" in settings["eef_policy_ik"].default
    assert "orientation_tolerance=0.02rad" in settings["eef_policy_ik"].default
    assert "max_joint_delta=1.7rad" in settings["eef_policy_ik"].default
    assert settings["gripper_position_jump_compatibility"].default == "mask=8"
    assert settings["lingbot_execution_gate"].default == (
        "execute=true, step_mode=false, step_actions=false, max_model_calls=66"
    )
    assert settings["pi05_execution_gate"].default == (
        "execute=true, step_mode=false, step_actions=false, max_model_calls=53"
    )
    assert settings["teleop_gripper_mapping"].default == (
        "leader=[0,53.16], invert=false"
    )
    assert settings["gripper_scale_mapping"].default.startswith("continuous 0..1")


def test_safety_report_has_text_and_json_forms():
    text = format_safety_report()
    payload = safety_report_as_dict()

    assert "Galaxea A1 Runtime Safety Report" in text
    assert "Architecture findings" in text
    assert payload["settings"]
    assert payload["architecture_findings"] == list(build_architecture_findings())
