"""Static safety disclosure for the Galaxea A1 runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.act.config import load_act_config
from galaxea_a1_runtime.teleop.config import load_teleop_config
from galaxea_a1_runtime.configuration.paths import (
    ACT_CONFIG,
    LINGBOT_CONFIG,
    SYSTEM_CONFIG,
    TELEOP_CONFIG,
)
from galaxea_a1_runtime.constants import IDLE_TIMEOUT_CODE, SAFE_RELAY_SCRIPT

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SafetySetting:
    name: str
    path: str
    default: str
    behavior: str
    visibility: str
    operator_note: str


def build_safety_settings(
    system_path: Path | None = None,
    *,
    repo_root: Path = ROOT,
    teleop_path: Path | None = None,
    lingbot_path: Path | None = None,
    act_path: Path | None = None,
) -> tuple[SafetySetting, ...]:
    """Return the runtime safety controls that can change execution behavior."""

    root = repo_root.resolve()
    system = load_system_config(
        system_path or root / SYSTEM_CONFIG,
        repo_root=root,
    )
    teleop = load_teleop_config(
        teleop_path or root / TELEOP_CONFIG,
        repo_root=root,
    )
    lingbot = load_lingbot_config(
        lingbot_path or root / LINGBOT_CONFIG,
        repo_root=root,
    )
    act = load_act_config(
        act_path or root / ACT_CONFIG,
        repo_root=root,
    )
    for owner, referenced_system in (
        ("teleop", teleop.system),
        ("lingbot", lingbot.system),
        ("act", act.system),
    ):
        if referenced_system.path != system.path:
            raise ValueError(
                f"{owner} config references {referenced_system.path}, "
                f"but the report system is {system.path}"
            )
    topics = system.topics

    return (
        SafetySetting(
            name="safe_command_path",
            path=f"{topics.eef_target} -> {topics.staged_command} -> relay -> {topics.host_command}",
            default="enabled for normal apps",
            behavior="App EE targets are staged and only forwarded by the relay after validation.",
            visibility=f"relay status on {topics.relay_status}",
            operator_note="Direct host command publishing exists only in the direct-debug profile.",
        ),
        SafetySetting(
            name="relay_lock",
            path=topics.motion_enable,
            default="LOCKED",
            behavior="The relay publishes nothing to the host command topic until an app explicitly enables motion.",
            visibility=f"LOCKED/ARMING/ACTIVE/FAULT JSON on {topics.relay_status}",
            operator_note="This is intentional fail-closed behavior, not a motion-planning feature.",
        ),
        SafetySetting(
            name="freshness_gate",
            path=SAFE_RELAY_SCRIPT,
            default=(
                f"command/feedback={system.relay.max_input_age_s:.2f}s, "
                f"status={system.relay.max_status_age_s:.2f}s"
            ),
            behavior="Joint feedback, staged tracker command, and motor status must all be fresh.",
            visibility="ARMING reason names the stale input; FAULT latches after arming timeout.",
            operator_note="A stale source looks like no motion unless the relay status is checked.",
        ),
        SafetySetting(
            name="arming_timeout",
            path=SAFE_RELAY_SCRIPT,
            default=f"{system.relay.arming_timeout_s:.1f}s",
            behavior="If enabled but validation cannot become healthy, the relay latches FAULT.",
            visibility=f"FAULT reason on {topics.relay_status}",
            operator_note="Stop/restart or explicitly re-enable only after inspecting the reason.",
        ),
        SafetySetting(
            name="initial_command_alignment",
            path=SAFE_RELAY_SCRIPT,
            default=f"{system.joint_safety.initial_alignment_tolerance_rad:.2f}rad",
            behavior="Only the first staged tracker command is checked against current joint feedback.",
            visibility="FAULT with initial command error values if exceeded.",
            operator_note="This check does not modify commands; after validation the relay forwards staged joint commands unchanged.",
        ),
        SafetySetting(
            name="joint_action_step_guard",
            path=f"{SYSTEM_CONFIG} [joint_safety.action_step_guard_enabled]",
            default=(
                f"enabled={str(system.joint_safety.action_step_guard_enabled).lower()}, "
                f"max_step={system.joint_safety.max_action_step_rad:g}rad"
            ),
            behavior=(
                "ACT joint actions are rejected when a step exceeds the tracked limit."
                if system.joint_safety.action_step_guard_enabled
                else "ACT joint step-jump rejection is disabled; finite values and absolute joint limits still apply."
            ),
            visibility="ACT action validation reports the exact rejected joint and step.",
            operator_note="Enable only through the system config; do not add an app-local threshold.",
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
            name="teleop_staged_joint_path",
            path=f"{topics.joint_target} -> jointTracker -> relay",
            default="enabled for just teleop <experiment>",
            behavior=(
                "The SO leader bridge publishes joint targets to the joint tracker; "
                "the tracker output is staged and forwarded by the same relay."
            ),
            visibility=f"Teleop bridge log plus relay status on {topics.relay_status}.",
            operator_note="This preserves the old teleop workflow without direct app publishing to host commands.",
        ),
        SafetySetting(
            name="teleop_relative_mapping",
            path=f"{TELEOP_CONFIG} [bridge]",
            default=(
                "relative leader delta from startup pose"
                if teleop.bridge.mapping.relative
                else "absolute leader mapping"
            ),
            behavior="A1 target starts from current A1 joints plus mapped leader delta, avoiding a startup jump.",
            visibility="Bridge log prints leader keys and A1 target joint names.",
            operator_note="Edit the tracked teleop config, then restart the bridge after repositioning the leader baseline.",
        ),
        SafetySetting(
            name="teleop_joint_limits",
            path=f"{SYSTEM_CONFIG} [joint_safety.lower_limits / joint_safety.upper_limits]",
            default=(
                f"lower={list(system.joint_safety.lower_limits)}, "
                f"upper={list(system.joint_safety.upper_limits)}"
            ),
            behavior="Mapped leader targets are clipped to explicit joint limits before publishing target positions.",
            visibility="Limits are tracked config values and covered by pure mapping/config tests.",
            operator_note="This changes the requested target if the leader moves beyond the configured A1 range.",
        ),
        SafetySetting(
            name="act_execution_gate",
            path=f"{ACT_CONFIG} [execution]",
            default=(
                f"execute={str(act.execution.execute).lower()}, "
                f"step_mode={str(act.execution.step_mode).lower()}, "
                f"control_hz={act.execution.control_hz:g}"
            ),
            behavior="ACT publishes staged joint targets only when execute is enabled; step mode requires operator confirmation per inference.",
            visibility="ACT startup prints the execution mode before loading the policy.",
            operator_note="A replacement checkpoint must be registered and deployment_ready before execution can be enabled.",
        ),
        SafetySetting(
            name="lingbot_workspace_clamp",
            path=f"{SYSTEM_CONFIG} [eef.xyz_min / eef.xyz_max]",
            default=(
                f"x=[{system.eef.xyz_min[0]:g},{system.eef.xyz_max[0]:g}], "
                f"y=[{system.eef.xyz_min[1]:g},{system.eef.xyz_max[1]:g}], "
                f"z=[{system.eef.xyz_min[2]:g},{system.eef.xyz_max[2]:g}]"
            ),
            behavior="Absolute LingBot targets outside the configured workspace are clamped.",
            visibility="Bridge preview prints clamp=workspace:<axis>.",
            operator_note="This can make a policy look pinned at the boundary; the preview should reveal it.",
        ),
        SafetySetting(
            name="lingbot_orientation_mode",
            path=f"{SYSTEM_CONFIG} [eef.orientation_mode]",
            default=system.eef.orientation_mode,
            behavior="LingBot quaternion channels are ignored unless orientation mode is model-quat.",
            visibility="Bridge preview prints orientation_mode.",
            operator_note="This is deliberate for translation-first EEF control.",
        ),
        SafetySetting(
            name="lingbot_execution_gate",
            path=f"{LINGBOT_CONFIG} [execution]",
            default=(
                f"execute={str(lingbot.execution.execute).lower()}, "
                f"step_mode={str(lingbot.execution.step_mode).lower()}, "
                f"step_actions={str(lingbot.execution.step_actions).lower()}"
            ),
            behavior="LingBot only enables the relay when execution is configured; step gates remain explicit deployment settings.",
            visibility="LingBot startup prints dry-run/live and step-gate state.",
            operator_note="Keep execute false until the new checkpoint, quantiles, prompt, and workspace have been reviewed.",
        ),
        SafetySetting(
            name="lingbot_eef_servo_compensation",
            path=f"{LINGBOT_CONFIG} [action.servo_gain]",
            default=(
                f"gain={lingbot.servo.gain:g}, max_extra={lingbot.servo.max_extra_m:g}m"
            ),
            behavior="When gain is greater than 1, the bridge sends an amplified tracker target toward the policy target.",
            visibility="Bridge preview prints tracker_cmd_xyz whenever it differs from the policy target.",
            operator_note="A gain at or below 1 disables compensation; raise it only for intentional tracker under-tracking compensation.",
        ),
        SafetySetting(
            name="lingbot_cache_actual_feedback",
            path=f"{LINGBOT_CONFIG} [action.cache_actual_feedback]",
            default=str(lingbot.servo.cache_actual_feedback).lower(),
            behavior=(
                "The LingBot KV cache records measured EEF feedback."
                if lingbot.servo.cache_actual_feedback
                else "The LingBot KV cache records the tracker command."
            ),
            visibility=(
                "Bridge startup prints cache_action_source=measured-feedback."
                if lingbot.servo.cache_actual_feedback
                else "Bridge startup prints cache_action_source=tracker-command."
            ),
            operator_note="Enable measured feedback only for a model trained with measured feedback as its action history.",
        ),
        SafetySetting(
            name="lingbot_relay_status_guard",
            path=f"{SYSTEM_CONFIG} [relay.max_status_age_s]",
            default=f"{system.relay.max_status_age_s:g}s",
            behavior="The bridge refuses to keep publishing if relay status is stale or no longer ACTIVE.",
            visibility="RuntimeError includes last relay state.",
            operator_note="Prevents the confusing case where the app prints publishes while the relay has stopped forwarding.",
        ),
        SafetySetting(
            name="teleop_gripper_mapping",
            path=f"{TELEOP_CONFIG} [gripper] -> {SYSTEM_CONFIG} [gripper]",
            default=(
                f"leader=[{teleop.gripper.source_min:g},"
                f"{teleop.gripper.source_max:g}], "
                f"invert={str(teleop.gripper.invert).lower()}"
            ),
            behavior=(
                "The configured continuous SO leader position maps linearly into "
                f"[{system.gripper.stroke_min_mm:g}, "
                f"{system.gripper.stroke_max_mm:g}]mm."
            ),
            visibility="Teleop config doctor and collected run metadata expose both ranges.",
            operator_note=(
                "Out-of-range leader feedback saturates only when the tracked "
                "Teleop compatibility policy explicitly enables it."
            ),
        ),
        SafetySetting(
            name="gripper_position_jump_compatibility",
            path=f"{SYSTEM_CONFIG} [relay.gripper_ignored_error_mask]",
            default=f"mask={system.relay.gripper_ignored_error_mask}",
            behavior=(
                "The relay removes only the configured gripper status bits before "
                "deciding whether gripper forwarding is healthy."
            ),
            visibility=(
                f"Raw motor_error_codes remain visible on {topics.relay_status}."
            ),
            operator_note=(
                "The tracked mask is 8 for Position Jump compatibility; every "
                "other non-idle gripper bit remains fatal."
            ),
        ),
        SafetySetting(
            name="gripper_path",
            path=f"{topics.gripper_target} -> relay -> {topics.gripper_command}",
            default="staged and fail-closed",
            behavior="Normal apps publish staged targets; only the ACTIVE relay may publish hardware commands.",
            visibility="Bridge publish log prints gripper_mm.",
            operator_note="Direct host-topic publishing is reserved for explicit hardware debugging.",
        ),
        SafetySetting(
            name="gripper_scale_mapping",
            path=f"{SYSTEM_CONFIG} [gripper]",
            default=(
                "continuous 0..1 -> "
                f"[{system.gripper.stroke_min_mm:g}, "
                f"{system.gripper.stroke_max_mm:g}]mm"
            ),
            behavior=(
                "Feedback outside the physical range is rejected. ACT rejects model "
                "gripper values outside 0..1; LingBot applies its named output bound "
                "before the strict physical unit conversion."
            ),
            visibility="Bridge preview and publish log print gripper_norm and gripper_mm.",
            operator_note="Changing this range changes the data and checkpoint contract; collect and train a new run after changing it.",
        ),
    )


def build_architecture_findings() -> tuple[str, ...]:
    return (
        "GalaxeaA1Robot is a schema/IO composition wrapper and has no implicit hardware adapter; managed apps own every live ROS path.",
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
