#!/usr/bin/env python3
"""Policy → ROS bridge: replaces the SO leader in the teleop pipeline.

Same services as `just teleop`, but the policy drives the arm instead of the SO leader arm.

Pipeline:
    camera_server (ZMQ port 5558) ─┐
    /joint_states_host (ROS)        ├→ policy (WebSocket) → /arm_joint_target_position (ROS)
                                    │                      → /gripper_position_control_host (ROS)

Required services (same as just teleop minus the SO leader bridge):
    just launch roscore
    just launch driver
    just joint-tracker
    just policy                 ← WebSocket policy server (port 8001 by default)
    just launch camera-server   ← for live camera frames

Usage:
    just policy-ros-bridge
    just policy-ros-bridge 127.0.0.1 8001 "pick up the marker"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import pathlib

_A1_SDK = pathlib.Path(__file__).parents[3] / "third_party" / "A1_SDK" / "install"
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import cv2
import numpy as np
import zmq

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openpi_client import websocket_client_policy as _ws_policy

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control

CAM_PORT = 5558
CAM_HOST = "127.0.0.1"

_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "arm_joint6",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _parse_cam_timestamp(raw: bytes) -> float | None:
    try:
        ts = float(raw.decode("ascii"))
        return ts / 1e9 if ts > 1e12 else ts
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class PolicyRosBridge:
    def __init__(self, args):
        self.args = args
        self.prompt = args.prompt
        self._state7: np.ndarray | None = None
        self._state_lock = threading.Lock()
        self._latest_images: dict = {}

        # ZMQ camera subscriber
        ctx = zmq.Context()
        self._cam_sub = ctx.socket(zmq.SUB)
        self._cam_sub.setsockopt(zmq.RCVHWM, 4)
        self._cam_sub.connect(f"tcp://{args.cam_host}:{args.cam_port}")
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # ROS
        rospy.init_node("policy_ros_bridge", anonymous=False)
        self._joint_pub = rospy.Publisher(
            "/arm_joint_target_position", JointState, queue_size=10
        )
        self._gripper_pub = rospy.Publisher(
            "/gripper_position_control_host", gripper_position_control, queue_size=10
        )
        rospy.Subscriber("/joint_states_host", JointState, self._joint_state_cb, queue_size=1)

        # Policy
        print(f"[Bridge] Connecting to policy at ws://{args.host}:{args.port} ...")
        self._policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        print(f"[Bridge] Connected. Metadata: {self._policy.get_server_metadata()}")

    def _joint_state_cb(self, msg: JointState):
        if not msg.position or not msg.name:
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        required = _JOINT_NAMES + ["gripper"]
        if not all(n in name_to_pos for n in required):
            # Fallback: use positional order
            if len(msg.position) >= 7:
                state7 = np.array(msg.position[:7], dtype=np.float32)
            else:
                return
        else:
            state7 = np.array([float(name_to_pos[n]) for n in required], dtype=np.float32)
        # Convert live gripper state from mm → training-data radians
        state7[6] = state7[6] * self.args.state_gripper_scale + self.args.state_gripper_offset
        with self._state_lock:
            self._state7 = state7

    def _poll_cameras(self):
        for _ in range(20):
            try:
                parts = self._cam_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) != 3:
                continue
            cam_id = parts[0].decode("utf-8", errors="replace")
            ts = _parse_cam_timestamp(parts[1])
            if ts is None:
                continue
            img = _decode_jpeg(parts[2])
            if img is None:
                continue
            self._latest_images[cam_id] = img

    def _get_cameras(self) -> dict | None:
        self._poll_cameras()
        if "cam_0" not in self._latest_images or "cam_1" not in self._latest_images:
            return None
        return {"cam_0": self._latest_images["cam_0"], "cam_1": self._latest_images["cam_1"]}

    def _publish_action(self, action7: np.ndarray):
        """Publish 6-DOF arm target + gripper, same topics as the SO leader bridge."""
        now = rospy.Time.now()

        js = JointState()
        js.header.stamp = now
        js.name = _JOINT_NAMES
        js.position = [float(v) for v in action7[:6]]
        self._joint_pub.publish(js)

        gripper_val = float(action7[6])
        stroke_mm = gripper_val * self.args.gripper_scale + self.args.gripper_offset
        gm = gripper_position_control()
        gm.header.stamp = now
        gm.gripper_stroke = stroke_mm
        self._gripper_pub.publish(gm)

    def run(self):
        step_dt = 1.0 / self.args.exec_rate if self.args.exec_rate > 0 else 0.0
        step_count = 0
        print("[Bridge] Waiting for joint state and camera frames ...")

        while not rospy.is_shutdown():
            # Get latest state
            with self._state_lock:
                state7 = self._state7
            if state7 is None:
                time.sleep(0.01)
                continue

            # Get latest cameras
            cameras = self._get_cameras()
            if cameras is None:
                time.sleep(0.01)
                continue

            obs = {
                "observation/image":       cameras["cam_0"],
                "observation/wrist_image": cameras["cam_1"],
                "observation/state":       state7,
                "prompt":                  self.prompt,
            }

            try:
                result = self._policy.infer(obs)
            except Exception as e:
                print(f"[Bridge] Inference error: {e}")
                continue

            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[np.newaxis, :]

            step_count += 1
            print(
                f"[Bridge] Step {step_count}: chunk={actions.shape[0]}  "
                f"action[0]={np.round(actions[0, :6], 3).tolist()}"
            )

            # Execute all actions in the chunk at exec_rate Hz
            for action in actions:
                if rospy.is_shutdown():
                    break
                t0 = time.monotonic()
                self._publish_action(action[:7])
                elapsed = time.monotonic() - t0
                remaining = step_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(
        description="Policy → ROS bridge (same pipeline as just teleop, policy replaces SO leader)."
    )
    parser.add_argument("--host",    default="127.0.0.1", help="WebSocket policy server host")
    parser.add_argument("--port",    type=int, default=8001,  help="WebSocket policy server port")
    parser.add_argument("--prompt",  default="pick up the marker and place it into the red plate")
    parser.add_argument("--cam-port", type=int, default=CAM_PORT, help="ZMQ camera port")
    parser.add_argument("--cam-host", default=CAM_HOST)
    parser.add_argument(
        "--exec-rate", type=float, default=10.0,
        help="Rate (Hz) at which each action in a chunk is sent to the robot.",
    )
    parser.add_argument(
        "--gripper-scale", type=float, default=-58.14,
        help="Multiply policy action[6] by this to get gripper stroke mm.",
    )
    parser.add_argument(
        "--gripper-offset", type=float, default=-18.84,
        help="Add this to (action[6] * scale) to get gripper stroke mm.",
    )
    parser.add_argument(
        "--state-gripper-scale", type=float, default=-0.01720,
        help="Multiply live gripper state (mm) by this to convert to training radians.",
    )
    parser.add_argument(
        "--state-gripper-offset", type=float, default=-0.324,
        help="Add this to (gripper_mm * scale) to convert to training radians.",
    )
    args = parser.parse_args()

    bridge = PolicyRosBridge(args)
    bridge.run()


if __name__ == "__main__":
    main()
