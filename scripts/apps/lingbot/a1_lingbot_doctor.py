#!/usr/bin/env python3
"""Configuration-driven layered health check for LingBot on A1."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.apps.lingbot.config import (  # noqa: E402
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.console import ArgumentParser  # noqa: E402
from galaxea_a1_runtime.constants import (  # noqa: E402
    EE_TRACKER_NODE,
    SAFE_RELAY_NODE,
)
from galaxea_a1_runtime.hardware.cameras import realsense_device_info  # noqa: E402
from galaxea_a1_runtime.runtime.health_checks import (  # noqa: E402
    Check,
    RosDoctorContext,
    add_check,
    arm_control_result,
    finish_checks,
    relay_status_result,
)


def websocket_open(host: str, port: int, *, timeout_s: float) -> bool:
    try:
        import websockets.sync.client

        with websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            ping_interval=None,
            close_timeout=timeout_s,
            open_timeout=timeout_s,
        ) as websocket:
            websocket.recv(timeout=timeout_s)
        return True
    except Exception:
        return False


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path(ROOT))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-execution", action="store_true")
    args = parser.parse_args()
    config = load_lingbot_config(args.config, repo_root=ROOT)
    system = config.system
    topics_config = system.topics
    checks: list[Check] = []

    serial = Path(system.host.a1_serial)
    serial_ok = serial.exists()
    serial_detail = "missing (expected while arm power is off)"
    if serial_ok:
        serial_detail = f"{serial} -> {serial.resolve()}"
    add_check(
        checks,
        "serial",
        serial_ok,
        serial_detail,
        required=args.require_execution,
    )
    wrist = system.cameras.wrist
    if wrist.backend == "realsense":
        try:
            wrist_info = realsense_device_info(wrist.serial)
        except Exception as exc:
            add_check(checks, "wrist_camera", False, str(exc))
        else:
            add_check(checks, "wrist_camera", wrist_info is not None, str(wrist_info))
    else:
        add_check(checks, "wrist_camera", Path(wrist.device).exists(), wrist.device)
    add_check(
        checks,
        "lingbot_server",
        websocket_open(
            config.server.host,
            config.server.port,
            timeout_s=config.server.connect_timeout_s,
        ),
        f"{config.server.host}:{config.server.port}",
    )

    try:
        import rosgraph
        import rosnode
        import rospy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from signal_arm.msg import arm_control, status_stamped
        from std_msgs.msg import String
    except Exception as exc:
        add_check(
            checks,
            "ros_python",
            False,
            repr(exc),
            required=args.require_execution,
        )
        return finish_checks(checks, json_output=args.json)

    master_ok = rosgraph.is_master_online()
    add_check(
        checks,
        "ros_master",
        master_ok,
        os.environ.get("ROS_MASTER_URI", "http://localhost:11311"),
        required=args.require_execution,
    )
    if not master_ok:
        return finish_checks(checks, json_output=args.json)

    rospy.init_node("a1_lingbot_doctor", anonymous=True, disable_signals=True)
    ros = RosDoctorContext(
        rospy=rospy,
        rosnode=rosnode,
        checks=checks,
        timeout_s=system.doctor.ros_topic_timeout_s,
        required=args.require_execution,
    )
    joints = ros.message(
        "joint_feedback",
        topics_config.joint_states,
        JointState,
        lambda msg: (
            len(msg.position) >= len(system.joint_safety.names),
            f"positions={len(msg.position)} "
            f"arm_values={list(msg.position[: len(system.joint_safety.names)])}",
        ),
    )
    ros.motor_status(topics_config.motor_status, status_stamped)
    ros.message(
        "staged_command",
        topics_config.staged_command,
        arm_control,
        lambda msg: arm_control_result(
            msg,
            arm_joints=len(system.joint_safety.names),
            allowed_modes=system.relay.allowed_control_modes,
        ),
    )
    ros.message(
        "ee_feedback",
        topics_config.eef_pose,
        PoseStamped,
        lambda msg: (
            True,
            "xyz="
            + str(
                [
                    round(msg.pose.position.x, 4),
                    round(msg.pose.position.y, 4),
                    round(msg.pose.position.z, 4),
                ]
            ),
        ),
    )

    for check_name, node_name in (
        ("tracker", EE_TRACKER_NODE),
        ("relay", SAFE_RELAY_NODE),
    ):
        ros.node(check_name, node_name)
    ros.message(
        "relay_state",
        topics_config.relay_status,
        String,
        lambda msg: relay_status_result(msg, require_execution=args.require_execution),
    )
    if joints is not None and not args.require_execution:
        add_check(
            checks,
            "power_off_consistency",
            True,
            "feedback exists although execution was not required",
            required=False,
        )
    return finish_checks(checks, json_output=args.json)


if __name__ == "__main__":
    sys.exit(main())
