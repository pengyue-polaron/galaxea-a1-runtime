"""Translate typed LingBot configuration into process arguments."""

from __future__ import annotations

from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.configuration.base import (
    bool_flag,
    number,
    shell_array,
    shell_assign,
)
from galaxea_a1_runtime.hardware.web_preview import web_preview_argv


def bridge_argv(config: LingBotConfig) -> list[str]:
    system = config.system
    topics = system.topics
    eef = system.eef
    front = system.cameras.front
    wrist = system.cameras.wrist
    args = [
        "--host",
        config.server.host,
        "--port",
        str(config.server.port),
        "--prompt",
        config.server.prompt,
        bool_flag("step-mode", config.execution.step_mode),
        "--max-model-calls",
        str(config.execution.max_model_calls),
        "--execute-frames",
        str(config.execution.execute_frames),
        bool_flag("condition-on-ee-state", config.execution.condition_on_ee_state),
        "--lingbot-frame-chunk-size",
        str(config.execution.lingbot_frame_chunk_size),
        "--lingbot-action-per-frame",
        str(config.execution.lingbot_action_per_frame),
        "--exec-rate",
        number(config.execution.exec_rate),
        bool_flag("print-actions", config.execution.print_actions),
        "--review-deadband",
        number(config.execution.review_deadband_m),
        "--cam-width",
        str(front.width),
        "--cam-height",
        str(front.height),
        "--cam-fps",
        str(front.fps),
        "--max-camera-age",
        number(system.cameras.max_age_s),
        "--cam0-serial",
        front.serial,
        bool_flag("cam0-auto-exposure", front.auto_exposure),
        "--cam0-exposure",
        str(front.exposure),
        "--cam0-gain",
        str(front.gain),
        bool_flag("cam0-auto-white-balance", front.auto_white_balance),
        "--cam0-white-balance",
        str(front.white_balance),
        "--cam0-observation-key",
        config.observations.front_key,
        bool_flag("cam0-crop-enabled", front.crop is not None),
        "--cam1-device",
        wrist.device,
        "--cam1-backend",
        wrist.backend,
        "--cam1-serial",
        wrist.serial,
        "--cam1-backend-api",
        wrist.backend_api,
        "--cam1-observation-key",
        config.observations.wrist_key,
        *web_preview_argv(system.web_preview),
        "--state-pose-topic",
        topics.eef_pose,
        "--state-gripper-topic",
        topics.gripper_feedback,
        "--cmd-pose-topic",
        topics.eef_target,
        "--cmd-gripper-topic",
        topics.gripper_target,
        "--motion-enable-topic",
        topics.motion_enable,
        "--relay-status-topic",
        topics.relay_status,
        "--relay-enable-timeout",
        number(system.relay.enable_timeout_s),
        "--max-relay-status-age",
        number(system.relay.max_status_age_s),
        "--command-frame",
        eef.command_frame,
        "--action-pose-mode",
        config.action.pose_mode,
        "--orientation-mode",
        eef.orientation_mode,
        "--eef-servo-gain",
        number(config.servo.gain),
        "--eef-servo-max-extra",
        number(config.servo.max_extra_m),
        "--eef-servo-settle",
        number(config.servo.settle_s),
        "--eef-servo-tolerance",
        number(config.servo.tolerance_m),
        "--eef-servo-corrections",
        str(config.servo.corrections),
        bool_flag("cache-actual-feedback", config.servo.cache_actual_feedback),
        "--xyz-min",
        *(number(value) for value in eef.xyz_min),
        "--xyz-max",
        *(number(value) for value in eef.xyz_max),
        "--min-quat-norm",
        number(eef.min_quat_norm),
        "--max-feedback-age",
        number(eef.max_feedback_age_s),
        "--feedback-wait-timeout",
        number(eef.feedback_wait_timeout_s),
        "--gripper-stroke-min",
        number(system.gripper.stroke_min_mm),
        "--gripper-stroke-max",
        number(system.gripper.stroke_max_mm),
    ]
    for enabled, flag in (
        (config.execution.execute, "--execute"),
        (config.execution.step_actions, "--step-actions"),
        (config.execution.no_kv_update, "--no-kv-update"),
    ):
        if enabled:
            args.append(flag)
    if config.execution.initial_ee_pose is not None:
        args.extend(
            [
                "--initial-ee-pose",
                *(number(value) for value in config.execution.initial_ee_pose),
            ]
        )
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


def bash_config(config: LingBotConfig) -> str:
    system = config.system
    lines = [
        shell_assign("CONFIG_PATH", str(config.path)),
        shell_assign("SYSTEM_CONFIG_PATH", str(system.path)),
        shell_assign("SESSION", config.session.tmux),
        shell_assign("LINGBOT_HOST", config.server.host),
        shell_assign("LINGBOT_PORT", str(config.server.port)),
        shell_assign("PROMPT", config.server.prompt),
        shell_assign("MODEL_SESSION", config.policy_server.tmux),
        shell_assign("MODEL_CHECKOUT", str(config.policy_server.checkout)),
        shell_assign("MODEL_PYTHON", str(config.policy_server.python)),
        shell_assign("BASE_MODEL", str(config.policy_server.base_model)),
        shell_assign("MODEL_CHECKPOINT", str(config.policy_server.checkpoint)),
        shell_assign("MODEL_ROOT", str(config.policy_server.model_root)),
        shell_assign("MODEL_SAVE_ROOT", str(config.policy_server.save_root)),
        shell_assign("MODEL_MASTER_PORT", str(config.policy_server.master_port)),
        shell_assign(
            "MODEL_STARTUP_TIMEOUT", number(config.policy_server.startup_timeout_s)
        ),
        shell_assign(
            "MODEL_EXPECTED_WEIGHT_SIZE",
            str(config.policy_server.expected_weight_size_bytes),
        ),
        shell_assign(
            "DEPLOYMENT_READY", "1" if config.policy_server.deployment_ready else "0"
        ),
        shell_assign("WRIST_BACKEND", system.cameras.wrist.backend),
        shell_assign("WRIST_SERIAL", system.cameras.wrist.serial),
        shell_assign("WRIST_CAMERA", system.cameras.wrist.device),
        shell_assign("STAGED_TOPIC", system.topics.staged_command),
        shell_assign("RELAY_ENABLE_TOPIC", system.topics.motion_enable),
        shell_assign("RELAY_STATUS_TOPIC", system.topics.relay_status),
        shell_array("BRIDGE_ARGS", bridge_argv(config)),
    ]
    return "\n".join(lines)
