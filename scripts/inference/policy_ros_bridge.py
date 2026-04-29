#!/usr/bin/env python3
"""Policy → ROS bridge: replaces the SO leader in the teleop pipeline.

Pipeline:
    cameras (direct capture)        ─┐
    /joint_states_host (ROS)         ├→ policy (WebSocket) → /arm_joint_target_position (ROS)
                                     │                      → /gripper_position_control_host (ROS)

Required services:
    roscore          (via Docker: scripts/collect_data/a1_noetic_docker.sh roscore)
    driver           (via Docker: scripts/collect_data/a1_noetic_docker.sh driver)
    joint-tracker    (via Docker: scripts/collect_data/a1_noetic_docker.sh tracker)
    policy server    (remote or local WebSocket server on port 8001)

Usage:
    just policy-ros-bridge
    just policy-ros-bridge 10.208.2.251 8001 "pick up the banana"
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import threading
import pathlib

# Remove ROS2 Humble paths — they shadow ROS1 packages (e.g. rosgraph_msgs)
sys.path = [p for p in sys.path if "/opt/ros/humble" not in p]

_A1_SDK = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "A1_SDK" / "install"
_USER_SITE = pathlib.Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
    str(_A1_SDK / "lib" / "python3" / "dist-packages"),
    str(_USER_SITE),
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

import cv2
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openpi_client import websocket_client_policy as _ws_policy

import rospy
from sensor_msgs.msg import JointState
from signal_arm.msg import gripper_position_control

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


# ---------------------------------------------------------------------------
# Auto-start roscore if no master is running
# ---------------------------------------------------------------------------

def _ros_master_reachable(timeout: float = 1.0) -> bool:
    """Check if a ROS master is listening on ROS_MASTER_URI."""
    try:
        from urllib.parse import urlparse
        uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
        parsed = urlparse(uri)
        sock = socket.create_connection(
            (parsed.hostname or "localhost", parsed.port or 11311), timeout=timeout
        )
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


_roscore_proc: subprocess.Popen | None = None


def _ensure_roscore() -> None:
    """Start roscore automatically if no master is detected."""
    global _roscore_proc
    if _ros_master_reachable():
        return

    print("[Bridge] ROS master not found — starting roscore automatically ...")
    # Build a clean env so ROS2 Humble paths don't shadow ROS1 packages
    env = os.environ.copy()
    for var in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "CMAKE_PREFIX_PATH"):
        paths = env.get(var, "")
        env[var] = ":".join(p for p in paths.split(":") if "/opt/ros/humble" not in p)
    for var in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "ROS_DISTRO"):
        env.pop(var, None)

    _roscore_proc = subprocess.Popen(
        ["roscore"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    for i in range(100):  # wait up to 10 s
        if _ros_master_reachable(timeout=0.5):
            print("[Bridge] roscore is up.")
            return
        time.sleep(0.1)

    # If we get here, roscore failed to start
    stderr = ""
    if _roscore_proc.poll() is not None:
        stderr = _roscore_proc.stderr.read().decode(errors="replace")
    _roscore_proc = None
    raise RuntimeError(f"Failed to start roscore.\n{stderr}")


def _cleanup_roscore() -> None:
    global _roscore_proc
    if _roscore_proc is not None:
        print("[Bridge] Stopping roscore ...")
        _roscore_proc.send_signal(signal.SIGINT)
        _roscore_proc.wait(timeout=5)
        _roscore_proc = None

_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "arm_joint6",
]


# ---------------------------------------------------------------------------
# Direct camera capture (same as camera_server, no ZMQ)
# ---------------------------------------------------------------------------

class _RealSenseCamera:
    """RealSense camera — initialized EXACTLY like lerobot_a1_collect.py used during data
    collection. No exposure/gain/auto_exposure overrides, just open the pipeline.

    However, since RealSense retains settings from prior sessions, we first force
    auto-exposure ON (the camera's factory default) to ensure consistent behavior with
    the collection-time state. Then we wait for auto-exposure to settle.
    """
    def __init__(self, serial: str | None, width: int, height: int, fps: int,
                 auto_exposure: bool = True, exposure: int | None = None,
                 gain: int | None = None):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed")
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        # Match data collection exactly: 640x480 BGR @ 30fps
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self._pipeline.start(config)

        # Reset camera to its factory default state (matches data-collection assumption
        # that the camera was in default auto-exposure mode).
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        # Reset white balance to auto as well (data collection didn't set it manually).
        try:
            color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        except Exception:
            pass
        # Let auto-exposure / auto-white-balance settle.
        for _ in range(90):
            self._pipeline.wait_for_frames()

    def read(self):
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self):
        self._pipeline.stop()


class _OpenCVCamera:
    """OpenCV camera — initialized EXACTLY like lerobot_a1_collect.py used during data
    collection: V4L2 backend, 640x480 @ 30fps, no manual exposure/WB.

    Resets the camera to auto modes (default state) since prior processes may have left
    manual exposure/WB settings stuck.
    """
    def __init__(self, device, width: int, height: int, fps: int, backend_api: str = "auto"):
        source = int(device) if str(device).isdigit() else str(device)
        if backend_api == "v4l2":
            self._cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device={device}")
        # Match data collection exactly: 640x480 @ 30fps
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        # Reset to auto modes (matches data-collection assumption of default camera state).
        # CAP_PROP_AUTO_EXPOSURE: 3 = auto, 1 = manual (V4L2 convention)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        # Drain a few warm-up frames so auto modes settle.
        for _ in range(30):
            self._cap.read()

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        self._cap.release()


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class PolicyRosBridge:
    def __init__(self, args):
        self.args = args
        self.prompt = args.prompt
        self._state7: np.ndarray | None = None
        self._state_ts: float = 0.0
        self._state_lock = threading.Lock()

        # --- Direct camera capture ---
        self._cameras: dict[str, _RealSenseCamera | _OpenCVCamera] = {}
        print("[Bridge] Opening cameras ...")
        if args.cam0_backend == "realsense":
            self._cameras["cam_0"] = _RealSenseCamera(
                serial=args.cam0_serial, width=args.cam_width, height=args.cam_height,
                fps=args.cam_fps, auto_exposure=not args.cam0_manual_exposure,
                exposure=args.cam0_exposure, gain=args.cam0_gain,
            )
        else:
            self._cameras["cam_0"] = _OpenCVCamera(
                device=args.cam0_device, width=args.cam_width, height=args.cam_height,
                fps=args.cam_fps, backend_api=args.cam0_backend_api,
            )
        print("[Bridge]   cam_0 ready")

        self._cameras["cam_1"] = _OpenCVCamera(
            device=args.cam1_device, width=args.cam_width, height=args.cam_height,
            fps=args.cam_fps, backend_api=args.cam1_backend_api,
        )
        print("[Bridge]   cam_1 ready")

        # --- ROS ---
        print("[Bridge] Connecting to ROS master ...")
        _ensure_roscore()
        rospy.init_node("policy_ros_bridge", anonymous=False)
        self._joint_pub = rospy.Publisher(
            "/arm_joint_target_position", JointState, queue_size=10
        )
        self._gripper_pub = rospy.Publisher(
            "/gripper_position_control_host", gripper_position_control, queue_size=10
        )
        rospy.Subscriber("/joint_states_host", JointState, self._joint_state_cb, queue_size=1)
        print("[Bridge] ROS connected.")

        # --- Policy (WebSocket) ---
        print(f"[Bridge] Connecting to policy at ws://{args.host}:{args.port} ...")
        self._policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        print(f"[Bridge] Connected. Metadata: {self._policy.get_server_metadata()}")

    def _joint_state_cb(self, msg: JointState):
        if not msg.position or not msg.name:
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        required = _JOINT_NAMES + ["gripper"]
        if not all(n in name_to_pos for n in required):
            if len(msg.position) >= 7:
                state7 = np.array(msg.position[:7], dtype=np.float32)
            else:
                return
        else:
            state7 = np.array([float(name_to_pos[n]) for n in required], dtype=np.float32)
        state7[6] = state7[6] * self.args.state_gripper_scale + self.args.state_gripper_offset
        with self._state_lock:
            self._state7 = state7
            self._state_ts = time.monotonic()

    def _read_cameras(self) -> dict[str, np.ndarray] | None:
        images = {}
        for cam_id, cam in self._cameras.items():
            frame = cam.read()
            if frame is None:
                return None
            # BGR → RGB
            images[cam_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return images

    def _publish_action(self, action7: np.ndarray):
        now = rospy.Time.now()

        js = JointState()
        js.header.stamp = now
        js.name = _JOINT_NAMES
        js.position = [float(v) for v in action7[:6]]
        self._joint_pub.publish(js)

        gripper_val = float(action7[6])
        stroke_mm = gripper_val * self.args.gripper_scale + self.args.gripper_offset
        # Clamp to [0, 100] mm. Negative strokes force motor past mechanical limit
        # → continuous high current → motor heating. 0 mm matches training command range.
        stroke_mm = max(0.0, min(100.0, stroke_mm))
        # Debug print every ~30 calls (~1s at 30Hz)
        if not hasattr(self, "_grip_count"):
            self._grip_count = 0
        self._grip_count += 1
        if self._grip_count % 30 == 1:
            print(f"  [Gripper] rad={gripper_val:+.3f} → stroke_mm={stroke_mm:.1f}")
        gm = gripper_position_control()
        gm.header.stamp = now
        gm.gripper_stroke = stroke_mm
        self._gripper_pub.publish(gm)

    def run(self):
        step_dt = 1.0 / self.args.exec_rate if self.args.exec_rate > 0 else 0.0
        step_count = 0
        step_mode = self.args.step_mode
        if step_mode:
            print("[Bridge] Step mode ON — press Enter after each chunk, q to quit.")
        print("[Bridge] Waiting for joint state and camera frames ...")

        _stale_warned = False
        while not rospy.is_shutdown():
            with self._state_lock:
                state7 = self._state7
                state_age = time.monotonic() - self._state_ts if self._state7 is not None else float("inf")
            if state7 is None or state_age > 1.0:
                if state7 is not None and not _stale_warned:
                    print(f"[Bridge] Joint state stale ({state_age:.1f}s) — pausing inference. Is the arm powered on?")
                    _stale_warned = True
                time.sleep(0.05)
                continue
            if _stale_warned:
                print("[Bridge] Joint state fresh — resuming inference.")
                _stale_warned = False

            cameras = self._read_cameras()
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

            # Execute all actions in the chunk before re-inferring.
            actions_to_exec = actions
            for action in actions_to_exec:
                if rospy.is_shutdown():
                    break
                t0 = time.monotonic()
                self._publish_action(action[:7])
                elapsed = time.monotonic() - t0
                remaining = step_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

            if step_mode:
                try:
                    cmd = input(f"[Bridge] Chunk {step_count} done. Enter=next, q=quit: ").strip().lower()
                except EOFError:
                    cmd = "q"
                if cmd in {"q", "quit", "exit"}:
                    print("[Bridge] Stopped by user.")
                    break

    def close(self):
        for cam in self._cameras.values():
            cam.close()


def main():
    parser = argparse.ArgumentParser(
        description="Policy → ROS bridge (cameras captured directly, actions via ROS)."
    )
    parser.add_argument("--host",    default="127.0.0.1", help="WebSocket policy server host")
    parser.add_argument("--port",    type=int, default=8001,  help="WebSocket policy server port")
    parser.add_argument("--prompt",  default="pick up the marker and place it into the red plate")

    # Camera settings (defaults match configs/collect_data.yaml)
    parser.add_argument("--cam-width",  type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps",    type=int, default=30)

    # cam_0: RealSense by default
    parser.add_argument("--cam0-backend", default="realsense", choices=["realsense", "opencv"])
    parser.add_argument("--cam0-serial",  default="341522300456", help="RealSense serial number")
    parser.add_argument("--cam0-device",  default="0", help="OpenCV device (if backend=opencv)")
    parser.add_argument("--cam0-backend-api", default="auto")
    parser.add_argument("--cam0-manual-exposure", action="store_true", default=False)
    parser.add_argument("--cam0-exposure", type=int, default=None)
    parser.add_argument("--cam0-gain",     type=int, default=None)

    # cam_1: OpenCV by default
    parser.add_argument("--cam1-device",      default="/dev/video0")
    parser.add_argument("--cam1-backend-api", default="v4l2")

    parser.add_argument(
        "--exec-rate", type=float, default=30.0,
        help="Rate (Hz) at which each action in a chunk is sent to the robot. "
             "Training data was recorded at 30 Hz (fps_target=30), so 30 Hz matches training dynamics.",
    )
    parser.add_argument("--gripper-scale",  type=float, default=38.36,
        help="Multiply policy action[6] (radians) by this to convert to stroke mm. "
             "Forward mapping (matches training data + user's calibration): "
             "rad -2.602 → stroke 0 mm (closed), rad 0.005 → stroke 100 mm (open). "
             "Stroke clamped to [0, 100].")
    parser.add_argument("--gripper-offset", type=float, default=99.81,
        help="Add this to (action[6] * scale) to get gripper stroke mm.")
    parser.add_argument("--state-gripper-scale",  type=float, default=1.0)
    parser.add_argument("--state-gripper-offset", type=float, default=0.0)
    parser.add_argument("--step-mode", action="store_true", default=False,
        help="Pause after each chunk and wait for Enter before next inference.")
    args = parser.parse_args()

    bridge = PolicyRosBridge(args)
    try:
        bridge.run()
    finally:
        bridge.close()
        _cleanup_roscore()


if __name__ == "__main__":
    main()
