"""Run the single ROS 2 camera owner in its isolated Python/ROS environment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
from pathlib import Path

from galaxea_a1_runtime.apps.cameras.ros2_pipeline import (
    ROS2_CAMERA_CONTAINER,
    ROS2_IMAGE,
)
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.hardware.cameras import (
    realsense_device_info,
    realsense_usb_is_superspeed,
)
from galaxea_a1_runtime.runtime.local_ipc import process_socket_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    system = load_system_config(args.config)
    root = Path(__file__).resolve().parents[3]
    if args.stop:
        result = subprocess.run(
            ["docker", "container", "inspect", ROS2_CAMERA_CONTAINER],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            # Distinguish an absent container from an unavailable Docker daemon.
            subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
            return 0
        container = json.loads(result.stdout)[0]
        if container["Config"]["Labels"].get("io.galaxea.a1-runtime.managed") != "true":
            raise RuntimeError("refusing to stop an unmanaged ROS 2 camera container")
        subprocess.run(
            ["docker", "stop", "--time", "5", ROS2_CAMERA_CONTAINER],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return 0
    subprocess.run(
        ["docker", "image", "inspect", ROS2_IMAGE],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    existing = subprocess.run(
        ["docker", "container", "inspect", ROS2_CAMERA_CONTAINER], capture_output=True
    )
    if existing.returncode == 0:
        raise RuntimeError(
            "ROS 2 camera container already exists; stop the camera workflow before retrying"
        )
    devices = []
    for camera in (system.cameras.front, system.cameras.wrist):
        device = realsense_device_info(camera.serial)
        if device is None:
            raise RuntimeError(f"RealSense {camera.serial} is unavailable")
        if camera.require_usb3 and not realsense_usb_is_superspeed(device.usb_type):
            raise RuntimeError(
                f"RealSense {camera.serial} requires USB3, got {device.usb_type}"
            )
        devices.append(device)
    state_root = process_socket_path("a1-camera-bridge.sock").parent
    state_root.mkdir(parents=True, exist_ok=True)
    device_args = []
    for device_path in sorted(Path("/dev").glob("video*")):
        device_args.extend(["--device", str(device_path)])
    for group in sorted(set(os.getgroups())):
        device_args.extend(["--group-add", str(group)])
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        ROS2_CAMERA_CONTAINER,
        "--label",
        "io.galaxea.a1-runtime.managed=true",
        "--network",
        "host",
        "--ipc",
        "host",
        "--init",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--device-cgroup-rule",
        "c 189:* rmw",
        *device_args,
        "-v",
        "/run/udev:/run/udev:ro",
        "-v",
        "/dev/bus/usb:/dev/bus/usb",
        "-v",
        f"{root}:/workspace:ro",
        "-v",
        f"{state_root}:/run/galaxea-a1-runtime",
        "-e",
        "A1_PROCESS_STATE_ROOT=/run/galaxea-a1-runtime",
        "-e",
        f"A1_FRONT_COLOR_MODULE={'depth_module' if 'D405' in devices[0].name else 'rgb_camera'}",
        "-e",
        f"A1_WRIST_COLOR_MODULE={'depth_module' if 'D405' in devices[1].name else 'rgb_camera'}",
        "-e",
        f"A1_SYSTEM_CONFIG_PATH=/workspace/{system.path.relative_to(root)}",
        "-e",
        f"A1_FRONT_USB_TYPE={devices[0].usb_type}",
        "-e",
        f"ROS_DOMAIN_ID={system.ros2.domain_id}",
        "-e",
        "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
        "-e",
        "ROS_LOG_DIR=/tmp/ros-log",
        ROS2_IMAGE,
        "ros2",
        "launch",
        "/workspace/scripts/apps/cameras/cameras.launch.py",
    ]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    process = subprocess.Popen(command)
    try:
        while process.poll() is None and not stop.wait(0.2):
            pass
        if not stop.is_set():
            raise RuntimeError(f"ROS 2 camera owner exited with {process.returncode}")
    finally:
        subprocess.run(
            ["docker", "stop", "--time", "5", ROS2_CAMERA_CONTAINER],
            capture_output=True,
        )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
