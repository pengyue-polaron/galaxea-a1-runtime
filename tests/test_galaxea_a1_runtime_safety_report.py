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
    assert "lingbot_orientation_mode" in settings
    assert "lingbot_eef_servo_compensation" in settings
    assert "lingbot_cache_actual_feedback" in settings
    assert settings["lingbot_cache_actual_feedback"].default
    assert "lingbot_relay_status_guard" in settings
    assert settings["gripper_position_jump_compatibility"].default == "mask=8"
    assert settings["joint_action_step_guard"].default.startswith("enabled=false")
    assert settings["act_execution_gate"].default.startswith("execute=false")
    assert settings["lingbot_execution_gate"].default.startswith("execute=false")
    assert settings["teleop_gripper_mapping"].default == (
        "leader=[0,53.16], invert=false"
    )
    assert settings["lingbot_eef_servo_compensation"].default == (
        "gain=1, max_extra=0.04m"
    )
    assert settings["lingbot_cache_actual_feedback"].default == "false"
    assert settings["gripper_scale_mapping"].default.startswith("continuous 0..1")
    assert (
        "does not modify commands"
        in settings["initial_command_alignment"].operator_note
    )


def test_safety_report_has_text_and_json_forms():
    text = format_safety_report()
    payload = safety_report_as_dict()

    assert "Galaxea A1 Runtime Safety Report" in text
    assert "Architecture findings" in text
    assert payload["settings"]
    assert payload["architecture_findings"] == list(build_architecture_findings())
