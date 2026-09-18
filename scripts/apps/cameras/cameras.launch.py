"""Official RealSense drivers and one shared timestamp synchronizer."""

import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from galaxea_a1_runtime.apps.cameras.ros2_pipeline import (
    CAMERA_NAMESPACE,
    driver_parameters,
)
from galaxea_a1_runtime.configuration.system import load_system_config


def generate_launch_description():
    config_path = Path(os.environ["A1_SYSTEM_CONFIG_PATH"])
    system = load_system_config(config_path)
    processes = []
    for name, camera in (
        ("front", system.cameras.front),
        ("wrist", system.cameras.wrist),
    ):
        processes.append(
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace=CAMERA_NAMESPACE,
                name=name,
                parameters=[
                    driver_parameters(
                        camera, os.environ[f"A1_{name.upper()}_COLOR_MODULE"]
                    ),
                    {"camera_name": name},
                ],
                output="screen",
            )
        )
    processes.append(
        ExecuteProcess(
            cmd=[
                sys.executable,
                "-m",
                "galaxea_a1_runtime.apps.cameras.ros2_pipeline",
                "--config",
                str(config_path),
                "--front-usb-type",
                os.environ["A1_FRONT_USB_TYPE"],
            ],
            output="screen",
        )
    )
    handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=process,
                on_exit=[
                    EmitEvent(event=Shutdown(reason="camera pipeline component exited"))
                ],
            )
        )
        for process in processes
    ]
    return LaunchDescription([*handlers, *processes])
