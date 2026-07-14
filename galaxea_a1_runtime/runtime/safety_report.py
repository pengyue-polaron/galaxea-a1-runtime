"""Static safety disclosure for the Galaxea A1 runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from galaxea_a1_runtime.constants import (
    DEFAULT_MAX_COMMAND_AGE_S,
    DEFAULT_MAX_INITIAL_COMMAND_ERROR_RAD,
    DEFAULT_RELAY_ARMING_TIMEOUT_S,
    EE_TARGET_TOPIC,
    GRIPPER_COMMAND_TOPIC,
    HOST_ARM_COMMAND_TOPIC,
    IDLE_TIMEOUT_CODE,
    RELAY_ENABLE_TOPIC,
    RELAY_STATUS_TOPIC,
    STAGED_ARM_COMMAND_TOPIC,
)


@dataclass(frozen=True)
class SafetySetting:
    name: str
    path: str
    default: str
    behavior: str
    visibility: str
    operator_note: str


def build_safety_settings() -> tuple[SafetySetting, ...]:
    """Return the runtime safety controls that can change execution behavior."""

    return (
        SafetySetting(
            name="safe_command_path",
            path=f"{EE_TARGET_TOPIC} -> {STAGED_ARM_COMMAND_TOPIC} -> relay -> {HOST_ARM_COMMAND_TOPIC}",
            default="enabled for normal apps",
            behavior="App EE targets are staged and only forwarded by the relay after validation.",
            visibility=f"relay status on {RELAY_STATUS_TOPIC}",
            operator_note="Direct host command publishing exists only in the direct-debug profile.",
        ),
        SafetySetting(
            name="relay_lock",
            path=RELAY_ENABLE_TOPIC,
            default="LOCKED",
            behavior="The relay publishes nothing to the host command topic until an app explicitly enables motion.",
            visibility=f"LOCKED/ARMING/ACTIVE/FAULT JSON on {RELAY_STATUS_TOPIC}",
            operator_note="This is intentional fail-closed behavior, not a motion-planning feature.",
        ),
        SafetySetting(
            name="freshness_gate",
            path="safe_arm_command_relay.py",
            default=f"{DEFAULT_MAX_COMMAND_AGE_S:.2f}s",
            behavior="Joint feedback, staged tracker command, and motor status must all be fresh.",
            visibility="ARMING reason names the stale input; FAULT latches after arming timeout.",
            operator_note="A stale source looks like no motion unless the relay status is checked.",
        ),
        SafetySetting(
            name="arming_timeout",
            path="safe_arm_command_relay.py",
            default=f"{DEFAULT_RELAY_ARMING_TIMEOUT_S:.1f}s",
            behavior="If enabled but validation cannot become healthy, the relay latches FAULT.",
            visibility=f"FAULT reason on {RELAY_STATUS_TOPIC}",
            operator_note="Stop/restart or explicitly re-enable only after inspecting the reason.",
        ),
        SafetySetting(
            name="initial_command_alignment",
            path="safe_arm_command_relay.py",
            default=f"{DEFAULT_MAX_INITIAL_COMMAND_ERROR_RAD:.2f}rad",
            behavior="Only the first staged tracker command is checked against current joint feedback.",
            visibility="FAULT with initial command error values if exceeded.",
            operator_note="This check does not modify commands; after validation the relay forwards staged joint commands unchanged.",
        ),
        SafetySetting(
            name="motor_status_filter",
            path="galaxea_a1_runtime.safety",
            default=f"0 or {IDLE_TIMEOUT_CODE} accepted for arm joints",
            behavior="Pure idle timeout code 64 is non-blocking; extra error bits block/fault.",
            visibility="relay reason lists bad joint error codes.",
            operator_note="This matches the observed A1 idle behavior and keeps 68-style faults blocking.",
        ),
        SafetySetting(
            name="generic_policy_delta_limits",
            path="GalaxeaA1Robot.send_action",
            default="off unless RuntimeConfig.safety sets max_eef_delta_m or max_rot_delta_rad",
            behavior=(
                "LeRobot EEF delta actions are forwarded unchanged by default; explicit runtime "
                "limits clamp them before hardware IO."
            ),
            visibility="Returned action dict shows the post-limit action when limits are configured.",
            operator_note="No model-output delta clamp is active in the default generic robot config.",
        ),
        SafetySetting(
            name="generic_ros1_adapter_arm_motion",
            path="galaxea_a1_runtime.hardware.ros1",
            default="feedback-driven EEF target synthesis",
            behavior=(
                "EEF translation/delta actions are added to live /end_effector_pose feedback, "
                "published to /a1_ee_target, and paired with relay enable; joint_absolute is rejected."
            ),
            visibility="RuntimeError before arm motion if /end_effector_pose has not been received.",
            operator_note="This path never publishes directly to /arm_joint_command_host.",
        ),
        SafetySetting(
            name="generic_ros1_gripper_range_check",
            path="galaxea_a1_runtime.hardware.ros1",
            default="continuous 0..1 -> 0..200mm",
            behavior="The generic ROS1 adapter clips finite normalized policy gripper values and maps them linearly to millimeters.",
            visibility="Published gripper stroke follows the normalized value continuously.",
            operator_note="The managed ACT and LingBot paths read the same range from configs/system/a1.toml.",
        ),
        SafetySetting(
            name="teleop_staged_joint_path",
            path="/arm_joint_target_position -> jointTracker -> relay",
            default="enabled for just teleop <experiment>",
            behavior=(
                "The SO leader bridge publishes joint targets to the joint tracker; "
                "the tracker output is staged and forwarded by the same relay."
            ),
            visibility="Teleop bridge log plus relay status on /a1_arm_relay_status.",
            operator_note="This preserves the old teleop workflow without direct app publishing to host commands.",
        ),
        SafetySetting(
            name="teleop_relative_mapping",
            path="configs/teleop/a1_so100.toml [bridge]",
            default="relative leader delta from startup pose",
            behavior="A1 target starts from current A1 joints plus mapped leader delta, avoiding a startup jump.",
            visibility="Bridge log prints leader keys and A1 target joint names.",
            operator_note="Edit the tracked teleop config, then restart the bridge after repositioning the leader baseline.",
        ),
        SafetySetting(
            name="teleop_joint_limits",
            path="configs/system/a1.toml [joint_safety.lower_limits / joint_safety.upper_limits]",
            default="A1 SDK joint limits",
            behavior="Mapped leader targets are clipped to explicit joint limits before publishing target positions.",
            visibility="Limits are tracked config values and covered by pure mapping/config tests.",
            operator_note="This changes the requested target if the leader moves beyond the configured A1 range.",
        ),
        SafetySetting(
            name="lingbot_workspace_clamp",
            path="configs/system/a1.toml [eef.xyz_min / eef.xyz_max]",
            default="x=[0.06,0.44], y=[-0.27,0.14], z=[0.06,0.50]",
            behavior="Absolute LingBot targets outside the configured workspace are clamped.",
            visibility="Bridge preview prints clamp=workspace:<axis>.",
            operator_note="This can make a policy look pinned at the boundary; the preview should reveal it.",
        ),
        SafetySetting(
            name="lingbot_orientation_mode",
            path="configs/system/a1.toml [eef.orientation_mode]",
            default="hold-current",
            behavior="LingBot quaternion channels are ignored unless orientation mode is model-quat.",
            visibility="Bridge preview prints orientation_mode.",
            operator_note="This is deliberate for translation-first EEF control.",
        ),
        SafetySetting(
            name="lingbot_eef_servo_compensation",
            path="configs/deployments/lingbot_va.toml [action.servo_gain]",
            default="1.0/off",
            behavior="When gain is greater than 1, the bridge sends an amplified tracker target toward the policy target.",
            visibility="Bridge preview prints tracker_cmd_xyz whenever it differs from the policy target.",
            operator_note="Default is off; enable only when you intentionally want to compensate official tracker under-tracking.",
        ),
        SafetySetting(
            name="lingbot_cache_actual_feedback",
            path="configs/deployments/lingbot_va.toml [action.cache_actual_feedback]",
            default="off; cache tracker command",
            behavior="The LingBot KV cache records the tracker command, matching the training action contract.",
            visibility="Bridge startup prints cache_action_source=tracker-command.",
            operator_note="Enable measured feedback only for a model trained with measured feedback as its action history.",
        ),
        SafetySetting(
            name="lingbot_relay_status_guard",
            path="configs/system/a1.toml [relay.max_status_age_s]",
            default="1.0s",
            behavior="The bridge refuses to keep publishing if relay status is stale or no longer ACTIVE.",
            visibility="RuntimeError includes last relay state.",
            operator_note="Prevents the confusing case where the app prints publishes while the relay has stopped forwarding.",
        ),
        SafetySetting(
            name="gripper_path",
            path=GRIPPER_COMMAND_TOPIC,
            default="independent of arm relay",
            behavior="The LingBot bridge delays gripper publishes until the relay is confirmed ACTIVE.",
            visibility="Bridge publish log prints gripper_mm.",
            operator_note="Other manual gripper commands still bypass the arm relay by design.",
        ),
        SafetySetting(
            name="gripper_scale_mapping",
            path="configs/system/a1.toml [gripper]",
            default="continuous 0..1 -> system stroke range",
            behavior="Feedback outside the physical range is rejected; finite policy outputs are clipped to 0..1 and mapped linearly into the shared stroke range.",
            visibility="Bridge preview and publish log print gripper_norm and gripper_mm.",
            operator_note="Changing this range changes the data and checkpoint contract; collect and train a new run after changing it.",
        ),
    )


def build_architecture_findings() -> tuple[str, ...]:
    return (
        "The generic LeRobot ROS1 adapter now executes EEF translation/delta actions through the safe target path, but still rejects joint-space arm execution.",
        "Teleop joint-space control is implemented as an app runtime that stages jointTracker output through the relay.",
        "The relay no longer applies joint tracking-error or velocity clamps; staged tracker output is forwarded unchanged once validation passes.",
        "LingBot episode-relative targets are composed onto the startup pose before using the staged EEF target route.",
        "The direct-debug profile deliberately bypasses the relay and should remain isolated from normal app commands.",
    )


def safety_report_as_dict() -> dict[str, Any]:
    return {
        "settings": [asdict(item) for item in build_safety_settings()],
        "architecture_findings": list(build_architecture_findings()),
    }


def format_safety_report() -> str:
    lines = ["Galaxea A1 Runtime Safety Report", ""]
    lines.append("Controls that can change or block motion:")
    for item in build_safety_settings():
        lines.extend(
            [
                f"- {item.name}",
                f"  path: {item.path}",
                f"  default: {item.default}",
                f"  behavior: {item.behavior}",
                f"  visibility: {item.visibility}",
                f"  operator note: {item.operator_note}",
            ]
        )
    lines.extend(["", "Architecture findings:"])
    lines.extend(f"- {finding}" for finding in build_architecture_findings())
    return "\n".join(lines)
