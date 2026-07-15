"""Translate typed teleop configuration into process arguments."""

from __future__ import annotations

from galaxea_a1_runtime.configuration.base import (
    bool_flag,
    number,
    shell_array,
    shell_assign,
)
from galaxea_a1_runtime.hardware.web_preview import web_preview_argv
from galaxea_a1_runtime.teleop.config_schema import TeleopConfig


def bridge_argv(config: TeleopConfig) -> list[str]:
    mapping = config.bridge.mapping
    system = config.system
    topics = system.topics
    args = [
        "--leader-port",
        config.leader.port,
        "--leader-id",
        config.leader.id,
        bool_flag("leader-use-degrees", config.leader.use_degrees),
        "--hz",
        number(config.bridge.hz),
        "--dof",
        str(config.bridge.dof),
        "--joint-states-topic",
        topics.joint_states,
        "--target-topic",
        topics.joint_target,
        "--staged-command-topic",
        topics.staged_command,
        "--target-joint-names",
        _csv(system.joint_safety.names),
        bool_flag("relative", mapping.relative),
        bool_flag("input-degrees", mapping.input_degrees),
        f"--scale={_csv(mapping.scale)}",
        f"--sign={_csv(mapping.sign)}",
        f"--bias-rad={_csv(mapping.bias_rad)}",
        f"--lower-limits={_csv(mapping.lower_limits)}",
        f"--upper-limits={_csv(mapping.upper_limits)}",
        "--motion-enable-topic",
        topics.motion_enable,
        "--relay-status-topic",
        topics.relay_status,
        "--relay-enable-timeout",
        number(system.relay.enable_timeout_s),
        "--max-relay-status-age",
        number(system.relay.max_status_age_s),
        "--a1-state-timeout",
        number(config.bridge.a1_state_timeout_s),
        "--initial-alignment-tolerance",
        number(system.joint_safety.initial_alignment_tolerance_rad),
        bool_flag("gripper-enabled", config.gripper.enabled),
        "--gripper-source-key",
        config.gripper.source_key,
        "--gripper-topic",
        topics.gripper_target,
        "--gripper-min-stroke-mm",
        number(system.gripper.stroke_min_mm),
        "--gripper-max-stroke-mm",
        number(system.gripper.stroke_max_mm),
    ]
    if config.gripper.invert:
        args.append("--gripper-invert")
    return args


def collect_argv(config: TeleopConfig) -> list[str]:
    system = config.system
    topics = system.topics
    front = system.cameras.front
    wrist = system.cameras.wrist
    args = [
        "--data-root",
        str(config.collection.data_root),
        "--state-mode",
        config.collection.state_mode.value,
        "--fps",
        number(config.collection.fps),
        "--max-duration-s",
        number(config.collection.max_duration_s),
        bool_flag("auto-reset-after-save", config.collection.auto_reset_after_save),
        "--jpeg-quality",
        str(config.collection.jpeg_quality),
        "--ready-timeout-s",
        number(config.collection.ready_timeout_s),
        "--max-camera-age-s",
        number(system.cameras.max_age_s),
        "--max-joint-feedback-age-s",
        number(system.joint_safety.max_feedback_age_s),
        "--max-eef-feedback-age-s",
        number(system.eef.max_feedback_age_s),
        "--max-action-age-s",
        number(system.joint_safety.max_feedback_age_s),
        "--max-gripper-age-s",
        number(system.joint_safety.max_feedback_age_s),
        "--max-joint-action-step-rad",
        number(config.collection.max_joint_action_step_rad),
        "--joint-topic",
        topics.joint_states,
        "--eef-topic",
        topics.eef_pose,
        "--action-topic",
        topics.joint_target,
        "--gripper-feedback-topic",
        topics.gripper_feedback,
        "--gripper-action-topic",
        topics.gripper_target,
        "--gripper-stroke-min",
        number(system.gripper.stroke_min_mm),
        "--gripper-stroke-max",
        number(system.gripper.stroke_max_mm),
        "--staged-command-topic",
        topics.staged_command,
        "--host-command-topic",
        topics.host_command,
        "--cam0-width",
        str(front.width),
        "--cam0-height",
        str(front.height),
        "--cam0-fps",
        str(front.fps),
        bool_flag("cam0-require-usb3", front.require_usb3),
        bool_flag("cam0-depth-enabled", front.depth),
        "--cam0-depth-width",
        str(front.depth_width or front.width),
        "--cam0-depth-height",
        str(front.depth_height or front.height),
        bool_flag("cam0-align-depth-to-color", front.align_depth_to_color),
        bool_flag("cam0-crop-enabled", front.crop is not None),
        "--cam1-device",
        wrist.device,
        "--cam1-backend",
        wrist.backend,
        "--cam1-serial",
        wrist.serial,
        "--cam1-width",
        str(wrist.width),
        "--cam1-height",
        str(wrist.height),
        "--cam1-fps",
        str(wrist.fps),
        "--cam1-pixel-format",
        wrist.pixel_format,
        *web_preview_argv(system.web_preview),
    ]
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
    args.extend(["--cam0-serial", front.serial])
    return args


def bash_config(config: TeleopConfig) -> str:
    system = config.system
    topics = system.topics
    return "\n".join(
        [
            shell_assign("CONFIG_PATH", str(config.path)),
            shell_assign("SYSTEM_CONFIG_PATH", str(system.path)),
            shell_assign("IMAGE", system.host.image),
            shell_assign("SERIAL", system.host.a1_serial),
            shell_assign("LEADER_PORT", config.leader.port),
            shell_assign("LEADER_ID", config.leader.id),
            shell_assign("PREFIX", config.runtime.prefix),
            shell_assign("RUN_DIR", config.runtime.run_dir),
            shell_assign("STAGED_TOPIC", topics.staged_command),
            shell_assign("RELAY_ENABLE_TOPIC", topics.motion_enable),
            shell_assign("RELAY_STATUS_TOPIC", topics.relay_status),
            shell_assign("JOINT_STATES_TOPIC", topics.joint_states),
            shell_assign("HOST_COMMAND_TOPIC", topics.host_command),
            shell_assign("MOTOR_STATUS_TOPIC", topics.motor_status),
            shell_assign("GRIPPER_TARGET_TOPIC", topics.gripper_target),
            shell_assign("GRIPPER_COMMAND_TOPIC", topics.gripper_command),
            shell_assign("EEF_POSE_TOPIC", topics.eef_pose),
            shell_assign("RELAY_MAX_INPUT_AGE_S", number(system.relay.max_input_age_s)),
            shell_assign(
                "RELAY_ARMING_TIMEOUT_S", number(system.relay.arming_timeout_s)
            ),
            shell_assign(
                "RELAY_MAX_INITIAL_ERROR_RAD",
                number(system.joint_safety.initial_alignment_tolerance_rad),
            ),
            shell_assign("TARGET_TOPIC", topics.joint_target),
            shell_assign("GRIPPER_MIN_STROKE_MM", number(system.gripper.stroke_min_mm)),
            shell_assign("GRIPPER_MAX_STROKE_MM", number(system.gripper.stroke_max_mm)),
            shell_assign("WEB_PREVIEW_PORT", str(system.web_preview.port)),
            shell_array("BRIDGE_ARGS", bridge_argv(config)),
            shell_array("COLLECT_ARGS", collect_argv(config)),
        ]
    )


def _csv(values: tuple[float, ...] | tuple[str, ...]) -> str:
    return ",".join(
        number(value) if isinstance(value, float) else str(value) for value in values
    )
