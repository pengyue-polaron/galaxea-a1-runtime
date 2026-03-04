"""
Zero-shot EEF bridge: A1 robot arm ↔ pi05_droid WebSocket server.

DROID action format (8D):
  [0:3] EEF position delta  (dx, dy, dz)  in metres, world frame
  [3:6] EEF rotation delta  (rx, ry, rz)  axis-angle, world frame
  [6]   wrist / extra dim   (ignored for A1)
  [7]   gripper absolute    [0 = closed, 1 = open]

ROS topics used
---------------
  SUB  /joint_states_host          → DROID observation/joint_position (7D padded)
  SUB  /end_effector_pose          → current EEF pose for delta accumulation
  PUB  /a1_ee_target               → target EEF PoseStamped (consumed by eeTracker MPC/IK)
  PUB  /gripper_position_control_host → gripper stroke in mm

ZMQ topics used
---------------
  SUB  port 5558  → camera frames  (cam_0, cam_1)

Requires
--------
  Terminal 1  just launch roscore
  Terminal 2  just launch ee-tracker      # provides /end_effector_pose + MPC/IK
  Terminal 3  just launch camera-server   # provides ZMQ cameras
  Terminal 4  just policy-droid           # pi05_droid WebSocket server on port 8000
  Terminal 5  just droid-eef-bridge       # this script

Usage
-----
  just droid-eef-bridge
  just droid-eef-bridge "pick up the red cup"
  just droid-eef-bridge "..." --pos-scale 0.5 --action-chunk-size 3
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path


# ── ROS path setup ─────────────────────────────────────────────────────────────
def _extend_ros_python_paths():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
        str(repo_root / "third_party" / "A1_SDK" / "install" / "lib" / "python3" / "dist-packages"),
    ]
    for c in candidates:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.append(c)


_extend_ros_python_paths()

try:
    import rospy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from signal_arm.msg import gripper_position_control
except Exception as exc:
    print(
        f"ROS import failed: {exc}\n"
        "Tip: source /opt/ros/noetic/setup.bash and third_party/A1_SDK/install/setup.bash",
        file=sys.stderr,
    )
    raise SystemExit(1)

import cv2
import numpy as np
import zmq
from openpi_client import websocket_client_policy as _ws_policy

from datacoach.constants import ZMQ_CAM_PORT

# ── Gripper mapping ────────────────────────────────────────────────────────────
# A1 gripper stroke in mm: 0 = fully closed, ~80 = fully open
A1_GRIPPER_CLOSED_MM = 0.0
A1_GRIPPER_OPEN_MM = 80.0


def droid_gripper_to_mm(g01: float) -> float:
    """DROID gripper [0=closed,1=open] → A1 gripper stroke in mm."""
    return float(np.clip(g01, 0.0, 1.0)) * (A1_GRIPPER_OPEN_MM - A1_GRIPPER_CLOSED_MM)


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _quat_multiply(q1, q2):
    """Hamilton product of two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector to quaternion [x, y, z, w]."""
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis_angle / angle
    s = np.sin(angle / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(angle / 2.0)])


def _apply_delta_rotation(q_current: np.ndarray, delta_axis_angle: np.ndarray) -> np.ndarray:
    """Apply a rotation delta (axis-angle) on top of a current quaternion [x,y,z,w]."""
    dq = _axis_angle_to_quat(delta_axis_angle)
    q_new = _quat_multiply(dq, q_current)
    norm = np.linalg.norm(q_new)
    if norm < 1e-9:
        return q_current
    return q_new / norm


# ── Camera helpers ─────────────────────────────────────────────────────────────

def _decode_jpeg_rgb_224(img_bytes: bytes) -> np.ndarray | None:
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)


# ── Main bridge ────────────────────────────────────────────────────────────────

class DroidEEFBridge:
    def __init__(
        self,
        zmq_host: str,
        server_host: str,
        server_port: int,
        prompt: str,
        action_chunk_size: int,
        pos_scale: float,
        rot_scale: float,
        flip_axes: str,
        max_camera_age_s: float,
        publish_rate_hz: float,
    ):
        self._prompt = prompt
        self._action_chunk_size = action_chunk_size
        self._pos_scale = pos_scale
        self._rot_scale = rot_scale
        # Per-axis sign for position AND rotation deltas: e.g. "y" flips Y.
        # When Y is mirrored, rotations around X and Z also flip sign.
        self._pos_sign = np.array([
            -1.0 if "x" in flip_axes else 1.0,
            -1.0 if "y" in flip_axes else 1.0,
            -1.0 if "z" in flip_axes else 1.0,
        ])
        # Rotation sign: flip the two axes orthogonal to each flipped axis.
        # Mirroring Y → negate rx, rz (axes that "cross" Y).
        rot_sign = np.ones(3)
        if "x" in flip_axes:
            rot_sign[1] *= -1; rot_sign[2] *= -1
        if "y" in flip_axes:
            rot_sign[0] *= -1; rot_sign[2] *= -1
        if "z" in flip_axes:
            rot_sign[0] *= -1; rot_sign[1] *= -1
        self._rot_sign = rot_sign
        self._max_camera_age_s = max_camera_age_s
        self._publish_rate_hz = publish_rate_hz
        self._publish_dt = 1.0 / publish_rate_hz

        # Shared ROS state (written by callbacks, read by main loop)
        self._lock = threading.Lock()
        self._joint_positions: np.ndarray | None = None   # (6,) A1 arm joints in rad
        self._gripper_joint: float | None = None          # A1 gripper joint (rad)
        self._current_eef: PoseStamped | None = None      # latest /end_effector_pose

        # Camera state
        self._latest_images: dict = {}
        self._warned_bad_cam = False
        self._cam_drop_count = 0

        # Action queue: list of (target_pose, gripper_mm)
        self._action_queue: deque = deque()

        # ── ROS init ──
        rospy.init_node("droid_eef_bridge", anonymous=True)

        rospy.Subscriber("/joint_states_host", JointState, self._joint_cb, queue_size=1)
        rospy.Subscriber("/end_effector_pose", PoseStamped, self._eef_cb, queue_size=1)

        self._eef_pub = rospy.Publisher("/a1_ee_target", PoseStamped, queue_size=1)
        self._grip_pub = rospy.Publisher(
            "/gripper_position_control_host", gripper_position_control, queue_size=1
        )

        # ── ZMQ cameras ──
        ctx = zmq.Context()
        self._cam_sub = ctx.socket(zmq.SUB)
        self._cam_sub.setsockopt(zmq.RCVHWM, 20)
        self._cam_sub.connect(f"tcp://{zmq_host}:{ZMQ_CAM_PORT}")
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # ── WebSocket policy client ──
        print(f"[Bridge] Connecting to pi05_droid at ws://{server_host}:{server_port} ...")
        self._policy = _ws_policy.WebsocketClientPolicy(host=server_host, port=server_port)
        print(f"[Bridge] Server metadata: {self._policy.get_server_metadata()}")
        print(f"[Bridge] Prompt: {self._prompt!r}")
        print(f"[Bridge] pos_scale={self._pos_scale}  rot_scale={self._rot_scale}  flip_axes={flip_axes!r}")
        print(f"[Bridge] action_chunk_size={self._action_chunk_size}  publish_rate={self._publish_rate_hz}Hz")

        time.sleep(0.3)

    # ── ROS callbacks ──────────────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        joints = [name_to_pos.get(f"arm_joint{i}", 0.0) for i in range(1, 7)]
        gripper = name_to_pos.get("gripper", 0.0)
        with self._lock:
            self._joint_positions = np.array(joints, dtype=np.float64)
            self._gripper_joint = float(gripper)

    def _eef_cb(self, msg: PoseStamped):
        with self._lock:
            self._current_eef = msg

    # ── Camera polling ─────────────────────────────────────────────────────────

    def _poll_cameras(self):
        while True:
            try:
                parts = self._cam_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) != 3:
                if not self._warned_bad_cam:
                    print("[Bridge] WARNING: camera message must be [cam_id, timestamp, jpeg]")
                    self._warned_bad_cam = True
                continue
            cam_id = parts[0].decode("utf-8", errors="replace")
            try:
                ts = float(parts[1].decode("ascii"))
                if ts > 1e12:
                    ts /= 1e9
            except Exception:
                continue
            img = _decode_jpeg_rgb_224(parts[2])
            if img is None:
                continue
            self._latest_images[cam_id] = {"image": img, "ts": ts}

    def _get_cameras(self) -> dict | None:
        self._poll_cameras()
        now = time.time()
        for cam_id in ("cam_0", "cam_1"):
            if cam_id not in self._latest_images:
                return None
            age = now - self._latest_images[cam_id]["ts"]
            if age > self._max_camera_age_s:
                self._cam_drop_count += 1
                if self._cam_drop_count % 50 == 1:
                    print(f"[Bridge] Stale camera {cam_id} age={age:.3f}s (#{self._cam_drop_count})")
                return None
        return {k: self._latest_images[k]["image"] for k in ("cam_0", "cam_1")}

    # ── Observation builder ────────────────────────────────────────────────────

    def _build_obs(self, joints: np.ndarray, gripper_rad: float, images: dict) -> dict:
        # Pad A1's 6-DOF arm to DROID's 7-DOF (zero-fill 7th joint)
        joint7 = np.zeros(7, dtype=np.float64)
        joint7[:6] = joints

        # Normalise A1 gripper rad → DROID [0=closed, 1=open]
        # A1 gripper rad range observed: -1.62 (open) to -0.55 (closed)
        g_closed, g_open = -0.55, -1.62
        gripper_01 = float(np.clip((gripper_rad - g_closed) / (g_open - g_closed), 0.0, 1.0))

        return {
            "observation/exterior_image_1_left": images["cam_0"],
            "observation/wrist_image_left":      images["cam_1"],
            "observation/joint_position":        joint7,
            "observation/gripper_position":      np.array([gripper_01], dtype=np.float64),
            "prompt": self._prompt,
        }

    # ── Action application ─────────────────────────────────────────────────────

    def _compute_action_queue(self, action_dict: dict, eef_pose: PoseStamped):
        """
        Convert DROID 8D actions into a queue of (PoseStamped, gripper_mm) targets.
        Deltas are accumulated from the current EEF pose.
        """
        actions = np.asarray(action_dict["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        n = min(actions.shape[0], self._action_chunk_size)

        # Unpack current pose
        p = eef_pose.pose.position
        o = eef_pose.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        quat = np.array([o.x, o.y, o.z, o.w])   # [x, y, z, w]

        self._action_queue.clear()
        for i in range(n):
            delta_pos = actions[i, 0:3] * self._pos_scale * self._pos_sign
            delta_rot = actions[i, 3:6] * self._rot_scale * self._rot_sign
            gripper_01 = float(actions[i, 7])

            pos = pos + delta_pos
            quat = _apply_delta_rotation(quat, delta_rot)

            target = PoseStamped()
            target.header.frame_id = eef_pose.header.frame_id or "world"
            target.header.stamp = rospy.Time.now()
            target.pose.position.x = float(pos[0])
            target.pose.position.y = float(pos[1])
            target.pose.position.z = float(pos[2])
            target.pose.orientation.x = float(quat[0])
            target.pose.orientation.y = float(quat[1])
            target.pose.orientation.z = float(quat[2])
            target.pose.orientation.w = float(quat[3])

            self._action_queue.append((target, droid_gripper_to_mm(gripper_01)))

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("[Bridge] Running. Waiting for joint states, EEF pose, and cameras...")
        infer_count = 0

        while not rospy.is_shutdown():
            loop_start = time.time()

            try:
                # ── Execute queued action ──
                if self._action_queue:
                    target_pose, gripper_mm = self._action_queue.popleft()
                    target_pose.header.stamp = rospy.Time.now()
                    self._eef_pub.publish(target_pose)

                    grip_msg = gripper_position_control()
                    grip_msg.gripper_stroke = float(gripper_mm)
                    self._grip_pub.publish(grip_msg)

                    p = target_pose.pose.position
                    print(
                        f"[chunk {len(self._action_queue)} left] "
                        f"eef=({p.x:.3f},{p.y:.3f},{p.z:.3f})  gripper={gripper_mm:.1f}mm"
                    )

                else:
                    # ── New inference ──
                    with self._lock:
                        joints = self._joint_positions
                        gripper_rad = self._gripper_joint
                        eef_pose = self._current_eef

                    if joints is None:
                        print("[Bridge] Waiting for /joint_states_host ...")
                        time.sleep(0.1)
                        continue
                    if eef_pose is None:
                        print("[Bridge] Waiting for /end_effector_pose ...")
                        time.sleep(0.1)
                        continue

                    images = self._get_cameras()
                    if images is None:
                        time.sleep(0.01)
                        continue

                    obs = self._build_obs(joints, gripper_rad, images)
                    t0 = time.time()
                    action_dict = self._policy.infer(obs)
                    infer_ms = (time.time() - t0) * 1000

                    with self._lock:
                        eef_pose_snap = self._current_eef

                    self._compute_action_queue(action_dict, eef_pose_snap)
                    infer_count += 1

                    target_pose, gripper_mm = self._action_queue.popleft()
                    target_pose.header.stamp = rospy.Time.now()
                    self._eef_pub.publish(target_pose)

                    grip_msg = gripper_position_control()
                    grip_msg.gripper_stroke = float(gripper_mm)
                    self._grip_pub.publish(grip_msg)

                    p = target_pose.pose.position
                    print(
                        f"[infer #{infer_count} {infer_ms:.0f}ms, chunk {len(self._action_queue)} left] "
                        f"eef=({p.x:.3f},{p.y:.3f},{p.z:.3f})  gripper={gripper_mm:.1f}mm"
                    )

            except KeyboardInterrupt:
                print("[Bridge] Stopped.")
                break
            except Exception:
                print("[Bridge] ERROR:")
                print(traceback.format_exc())

            # Rate limiting
            elapsed = time.time() - loop_start
            sleep_t = self._publish_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot DROID EEF bridge for A1")
    parser.add_argument("prompt",
                        nargs="?",
                        default="swap the position of the marker and the yellow block through the white plate",
                        help="Task prompt")
    parser.add_argument("--zmq-host",       default="localhost")
    parser.add_argument("--server-host",    default="localhost")
    parser.add_argument("--server-port",    type=int,   default=8000)
    parser.add_argument("--action-chunk-size", type=int, default=2,
                        help="Actions to execute before re-inferring")
    parser.add_argument("--pos-scale",      type=float, default=1.0,
                        help="Scale factor for EEF position delta (tune down if robot moves too fast)")
    parser.add_argument("--rot-scale",      type=float, default=1.0,
                        help="Scale factor for EEF rotation delta")
    parser.add_argument("--flip-axes",      default="y",
                        help="Axes to flip in DROID→A1 frame mapping, e.g. 'y', 'xy', '' for none")
    parser.add_argument("--max-camera-age-s", type=float, default=0.5)
    parser.add_argument("--publish-rate",   type=float, default=10.0,
                        help="Action publish rate in Hz")
    args = parser.parse_args()

    bridge = DroidEEFBridge(
        zmq_host=args.zmq_host,
        server_host=args.server_host,
        server_port=args.server_port,
        prompt=args.prompt,
        action_chunk_size=args.action_chunk_size,
        pos_scale=args.pos_scale,
        rot_scale=args.rot_scale,
        flip_axes=args.flip_axes.lower(),
        max_camera_age_s=args.max_camera_age_s,
        publish_rate_hz=args.publish_rate,
    )
    bridge.run()


if __name__ == "__main__":
    main()
