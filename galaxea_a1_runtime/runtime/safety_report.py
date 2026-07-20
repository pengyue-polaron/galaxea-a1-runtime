"""Static safety disclosure for the Galaxea A1 runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.apps.pi05.config import load_pi05_config
from galaxea_a1_runtime.configuration.paths import (
    LINGBOT_CONFIG,
    PI05_CONFIG,
    SYSTEM_CONFIG,
    TELEOP_CONFIG,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.constants import IDLE_TIMEOUT_CODE, SAFE_RELAY_SCRIPT
from galaxea_a1_runtime.teleop.config import load_teleop_config

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
    pi05_path: Path | None = None,
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
    pi05 = load_pi05_config(
        pi05_path or root / PI05_CONFIG,
        repo_root=root,
    )
    for owner, referenced_system in (
        ("teleop", teleop.system),
        ("lingbot", lingbot.system),
        ("pi05", pi05.system),
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
            path=(
                f"EEF policy -> bounded IK -> {topics.joint_target} -> "
                f"jointTracker -> {topics.staged_command} -> relay -> "
                f"{topics.host_command}"
            ),
            default="enabled for normal apps",
            behavior=(
                "App EEF targets become named joint targets through the tracked "
                "IK contract; only validated tracker output is forwarded by the relay."
            ),
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
            behavior=(
                "The relay checks the staged current-joint hold against current "
                "joint feedback before becoming ACTIVE."
            ),
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
            name="teleop_staged_joint_path",
            path=f"{topics.joint_target} -> jointTracker -> relay",
            default="enabled for just teleop <experiment>",
            behavior=(
                "The LeRobot Teleoperator and pair processor send canonical actions "
                "to the A1 Robot; its runtime backend publishes named joint targets "
                "to the staged tracker and relay."
            ),
            visibility=f"Teleop bridge log plus relay status on {topics.relay_status}.",
            operator_note="Neither out-of-tree plugin imports ROS or publishes host commands directly.",
        ),
        SafetySetting(
            name="teleop_relative_mapping",
            path=f"{TELEOP_CONFIG} [bridge]",
            default="relative leader delta from startup pose",
            behavior="A1 target starts from current A1 joints plus mapped leader delta, avoiding a startup jump.",
            visibility="Bridge becomes live only after the first processed hold is accepted by the Robot backend.",
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
            name="eef_policy_workspace_bounds",
            path=f"{SYSTEM_CONFIG} [eef.xyz_min / eef.xyz_max]",
            default=(
                f"x=[{system.eef.xyz_min[0]:g},{system.eef.xyz_max[0]:g}], "
                f"y=[{system.eef.xyz_min[1]:g},{system.eef.xyz_max[1]:g}], "
                f"z=[{system.eef.xyz_min[2]:g},{system.eef.xyz_max[2]:g}]"
            ),
            behavior="Absolute LingBot and pi0.5 targets outside the configured workspace are rejected without publication.",
            visibility="The bridge error names the offending axes, target, and configured bounds.",
            operator_note="Model outputs are never projected onto the workspace boundary.",
        ),
        SafetySetting(
            name="eef_policy_ik",
            path=f"{SYSTEM_CONFIG} [eef_ik] -> {topics.joint_target}",
            default=(
                f"position_tolerance={system.eef_ik.position_tolerance_m:g}m, "
                f"orientation_tolerance={system.eef_ik.orientation_tolerance_rad:g}rad, "
                f"max_joint_delta={system.eef_ik.max_solution_delta_rad:g}rad"
            ),
            behavior=(
                "The first-party URDF IK rejects non-convergence, non-finite values, "
                "joint-limit violations, and solutions beyond the configured joint delta."
            ),
            visibility=(
                "Verbose deployment logging reports IK iterations, Cartesian/orientation "
                "residuals, and maximum joint delta when enabled."
            ),
            operator_note=(
                "The bridge stages fresh named joint feedback as a hold; the relay "
                "is the sole owner of alignment validation and activation."
            ),
        ),
        SafetySetting(
            name="eef_policy_task_selection",
            path=str(lingbot.task_catalog.path.relative_to(root)),
            default=f"{len(lingbot.task_catalog.tasks)} tracked prompts",
            behavior=(
                "Live policy startup requires one tracked task selection before "
                "model, ROS, camera, or hardware processes start."
            ),
            visibility=(
                "The selector and bridge print the selected task id, train/OOD "
                "distribution, and exact prompt."
            ),
            operator_note=(
                "Cancelling selection starts nothing; unregistered prompts are rejected."
            ),
        ),
        SafetySetting(
            name="lingbot_execution_gate",
            path=f"{LINGBOT_CONFIG} [execution]",
            default=(
                f"execute={str(lingbot.execution.execute).lower()}, "
                f"step_mode={str(lingbot.execution.step_mode).lower()}, "
                f"step_actions={str(lingbot.execution.step_actions).lower()}, "
                f"max_model_calls={lingbot.execution.max_model_calls}"
            ),
            behavior="LingBot only enables the relay when execution is configured; step gates remain explicit deployment settings.",
            visibility="LingBot startup prints dry-run/live and step-gate state.",
            operator_note=(
                "The deployment owns rollout cadence; execution continues until its "
                "finite model-call cap is reached or the operator stops it."
            ),
        ),
        SafetySetting(
            name="pi05_execution_gate",
            path=f"{PI05_CONFIG} [execution]",
            default=(
                f"execute={str(pi05.execution.execute).lower()}, "
                f"step_mode={str(pi05.execution.step_mode).lower()}, "
                f"step_actions={str(pi05.execution.step_actions).lower()}, "
                f"max_model_calls={pi05.execution.max_model_calls}"
            ),
            behavior="Pi0.5 only enables the relay when execution is configured; inference and action-step gates remain explicit deployment settings.",
            visibility="Pi0.5 startup prints dry-run/live and step-gate state.",
            operator_note=(
                "The deployment owns rollout cadence; execution continues until its "
                "finite model-call cap is reached or the operator stops it."
            ),
        ),
        SafetySetting(
            name="eef_policy_relay_status_guard",
            path=f"{SYSTEM_CONFIG} [relay.max_status_age_s]",
            default=f"{system.relay.max_status_age_s:g}s",
            behavior="LingBot and pi0.5 bridges refuse to keep publishing if relay status is stale or no longer ACTIVE.",
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
            visibility="Teleop tracked config and collected run metadata expose both ranges.",
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
                "Feedback outside the physical range is rejected. EEF policy targets "
                "must satisfy the normalized bound before physical unit conversion."
            ),
            visibility="Bridge preview and publish log print gripper_norm and gripper_mm.",
            operator_note="Changing this range changes the data and checkpoint contract; collect and train a new run after changing it.",
        ),
    )


def build_architecture_findings() -> tuple[str, ...]:
    return (
        "Managed Teleop, LingBot, and pi0.5 apps own every supported live ROS path; no generic package adapter publishes implicitly.",
        "Teleop joint-space control is implemented as an app runtime that stages jointTracker output through the relay.",
        "The relay no longer applies joint tracking-error or velocity clamps; staged tracker output is forwarded unchanged once validation passes.",
        "LingBot and pi0.5 episode-relative targets are composed onto the startup pose, solved by bounded first-party IK, and sent through the staged joint route.",
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
