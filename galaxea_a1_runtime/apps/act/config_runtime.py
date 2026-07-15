"""Translate typed ACT configuration into process arguments."""

from __future__ import annotations

from galaxea_a1_runtime.apps.act.config_schema import ActConfig
from galaxea_a1_runtime.configuration.base import (
    bool_flag,
    number,
    shell_array,
    shell_assign,
)
from galaxea_a1_runtime.hardware.web_preview import web_preview_argv


def bridge_argv(config: ActConfig) -> list[str]:
    system = config.system
    topics = system.topics
    safety = system.joint_safety
    front = system.cameras.front
    wrist = system.cameras.wrist
    args = [
        "--checkpoint",
        str(config.policy.checkpoint),
        "--device",
        config.policy.device,
        bool_flag("disable-backbone-download", config.policy.disable_backbone_download),
        bool_flag("execute", config.execution.execute),
        bool_flag("step-mode", config.execution.step_mode),
        "--execute-steps-per-inference",
        str(config.execution.execute_steps_per_inference),
        "--control-hz",
        number(config.execution.control_hz),
        "--max-model-calls",
        str(config.execution.max_model_calls),
        bool_flag("print-actions", config.execution.print_actions),
        "--preview-steps",
        str(config.execution.preview_steps),
        "--joint-states-topic",
        topics.joint_states,
        "--target-topic",
        topics.joint_target,
        "--staged-command-topic",
        topics.staged_command,
        "--motion-enable-topic",
        topics.motion_enable,
        "--relay-status-topic",
        topics.relay_status,
        "--gripper-target-topic",
        topics.gripper_target,
        "--gripper-feedback-topic",
        topics.gripper_feedback,
        "--relay-enable-timeout",
        number(system.relay.enable_timeout_s),
        "--max-relay-status-age",
        number(system.relay.max_status_age_s),
        "--target-joint-names",
        *safety.names,
        "--lower-limits",
        *(number(value) for value in safety.lower_limits),
        "--upper-limits",
        *(number(value) for value in safety.upper_limits),
        bool_flag("action-step-guard-enabled", safety.action_step_guard_enabled),
        "--max-joint-action-step-rad",
        number(safety.max_action_step_rad),
        "--max-first-target-delta-rad",
        number(safety.max_first_target_delta_rad),
        "--initial-alignment-tolerance",
        number(safety.initial_alignment_tolerance_rad),
        "--state-timeout",
        number(safety.state_timeout_s),
        "--max-feedback-age",
        number(safety.max_feedback_age_s),
        "--max-camera-age",
        number(system.cameras.max_age_s),
        "--gripper-stroke-min",
        number(system.gripper.stroke_min_mm),
        "--gripper-stroke-max",
        number(system.gripper.stroke_max_mm),
        "--cam-width",
        str(front.width),
        "--cam-height",
        str(front.height),
        "--cam-fps",
        str(front.fps),
        "--camera-warmup-frames",
        str(system.cameras.warmup_frames),
        bool_flag("cam0-auto-exposure", front.auto_exposure),
        "--cam0-exposure",
        str(front.exposure),
        "--cam0-gain",
        str(front.gain),
        bool_flag("cam0-auto-white-balance", front.auto_white_balance),
        "--cam0-white-balance",
        str(front.white_balance),
        bool_flag("cam0-crop-enabled", front.crop is not None),
        "--cam1-device",
        wrist.device,
        "--cam1-backend",
        wrist.backend,
        "--cam1-serial",
        wrist.serial,
        "--cam1-backend-api",
        wrist.backend_api,
        "--cam1-pixel-format",
        wrist.pixel_format,
        *web_preview_argv(system.web_preview),
    ]
    if front.serial:
        args.extend(["--cam0-serial", front.serial])
    if front.crop is not None:
        args.extend(
            [
                "--cam0-crop-x",
                str(front.crop.x),
                "--cam0-crop-y",
                str(front.crop.y),
                "--cam0-crop-width",
                str(front.crop.width),
                "--cam0-crop-height",
                str(front.crop.height),
            ]
        )
    return args


def bash_config(config: ActConfig) -> str:
    return "\n".join(
        [
            shell_assign("CONFIG_PATH", str(config.path)),
            shell_assign("SYSTEM_CONFIG_PATH", str(config.system.path)),
            shell_assign("SESSION", config.session.tmux),
            shell_assign("PREFIX", config.runtime.prefix),
            shell_assign("CHECKPOINT", str(config.policy.checkpoint)),
            shell_assign(
                "DEPLOYMENT_READY", "1" if config.policy.deployment_ready else "0"
            ),
            shell_assign("WRIST_BACKEND", config.system.cameras.wrist.backend),
            shell_assign("WRIST_SERIAL", config.system.cameras.wrist.serial),
            shell_assign("WRIST_CAMERA", config.system.cameras.wrist.device),
            shell_array("BRIDGE_ARGS", bridge_argv(config)),
        ]
    )
