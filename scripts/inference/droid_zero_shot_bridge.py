"""
Zero-shot bridge: A1 robot arm ↔ pi05_droid WebSocket server.

Reads A1 ZMQ state/camera streams, reformats to DROID input format,
calls the pi05_droid WebSocket server, and publishes actions back to A1.

Format mapping
--------------
A1 state  : 7D  [j1..j6, gripper_rad]
DROID in  : joint_position=(7,) [j1..j6, 0.0 padding], gripper_position=(1,) [0..1]
DROID out : 8D  [j1..j6, j7_ignored, gripper_abs_01]
A1 action : 7D  [j1..j6 from DROID[:6], gripper_rad mapped back]

Cameras
-------
cam_0 (640×480 BGR) → 224×224 RGB → observation/exterior_image_1_left
cam_1 (640×480 BGR) → 224×224 RGB → observation/wrist_image_left

Usage
-----
    python scripts/inference/droid_zero_shot_bridge.py \
        --prompt "swap the marker and the block through the white plate"
"""

import argparse
import time
import traceback
from collections import deque

import cv2
import numpy as np
import zmq

# openpi_client must be on PYTHONPATH: packages/openpi-client/src
from openpi_client import websocket_client_policy as _ws_policy

from datacoach.constants import (
    ZMQ_STATE_PORT,
    ZMQ_CAM_PORT,
    ZMQ_POLICY_ACTION_PORT,
)

# ── Gripper mapping ────────────────────────────────────────────────────────────
# A1 gripper joint (radians) observed range from training data:
#   closed ≈ -0.55 rad,  open ≈ -1.62 rad
# DROID gripper output: 0.0 = closed, 1.0 = open
A1_GRIPPER_CLOSED_RAD = -0.55
A1_GRIPPER_OPEN_RAD = -1.62


def droid_gripper_to_a1(gripper_01: float) -> float:
    """Map DROID gripper [0=closed, 1=open] → A1 gripper radians."""
    return A1_GRIPPER_CLOSED_RAD + gripper_01 * (A1_GRIPPER_OPEN_RAD - A1_GRIPPER_CLOSED_RAD)


def a1_gripper_to_droid(gripper_rad: float) -> float:
    """Map A1 gripper radians → DROID gripper [0=closed, 1=open]."""
    span = A1_GRIPPER_OPEN_RAD - A1_GRIPPER_CLOSED_RAD
    val = (gripper_rad - A1_GRIPPER_CLOSED_RAD) / span
    return float(np.clip(val, 0.0, 1.0))


# ── Image helpers ──────────────────────────────────────────────────────────────

def decode_jpeg_to_rgb(img_bytes: bytes) -> np.ndarray | None:
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)


# ── Main bridge ────────────────────────────────────────────────────────────────

class DroidZeroShotBridge:
    def __init__(
        self,
        host: str,
        server_host: str,
        server_port: int,
        prompt: str,
        action_chunk_size: int,
        max_state_age_s: float,
        max_camera_age_s: float,
        max_cam_state_skew_s: float,
    ):
        self._prompt = prompt
        self._action_chunk_size = action_chunk_size
        self._max_state_age_s = max_state_age_s
        self._max_camera_age_s = max_camera_age_s
        self._max_cam_state_skew_s = max_cam_state_skew_s
        self._action_queue: deque = deque()
        self._latest_images: dict = {}
        self._warned_bad_cam = False
        self._stale_state_drops = 0
        self._stale_cam_drops = 0

        # ZMQ context
        ctx = zmq.Context()

        # SUB: robot state
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.RCVHWM, 1)
        self._sub.connect(f"tcp://{host}:{ZMQ_STATE_PORT}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # SUB: cameras
        self._cam_sub = ctx.socket(zmq.SUB)
        self._cam_sub.setsockopt(zmq.RCVHWM, 20)
        self._cam_sub.connect(f"tcp://{host}:{ZMQ_CAM_PORT}")
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # PUB: actions
        self._pub = ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://{host}:{ZMQ_POLICY_ACTION_PORT}")

        # WebSocket policy client
        print(f"[Bridge] Connecting to pi05_droid server at ws://{server_host}:{server_port} ...")
        self._policy = _ws_policy.WebsocketClientPolicy(host=server_host, port=server_port)
        print(f"[Bridge] Server metadata: {self._policy.get_server_metadata()}")

        print(f"[Bridge] State  SUB: tcp://{host}:{ZMQ_STATE_PORT}")
        print(f"[Bridge] Camera SUB: tcp://{host}:{ZMQ_CAM_PORT}")
        print(f"[Bridge] Action PUB: tcp://{host}:{ZMQ_POLICY_ACTION_PORT}")
        print(f"[Bridge] Prompt: {self._prompt!r}")
        time.sleep(0.3)

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
            img = decode_jpeg_to_rgb(parts[2])
            if img is None:
                continue
            self._latest_images[cam_id] = {"image": img, "ts": ts}

    def _get_cameras(self, state_ts: float) -> dict | None:
        self._poll_cameras()
        now = time.time()
        for cam_id in ("cam_0", "cam_1"):
            if cam_id not in self._latest_images:
                return None
            age = now - self._latest_images[cam_id]["ts"]
            if age > self._max_camera_age_s:
                self._stale_cam_drops += 1
                if self._stale_cam_drops % 100 == 1:
                    print(f"[Bridge] Stale camera {cam_id} age={age:.3f}s (#{self._stale_cam_drops})")
                return None
            if state_ts > 0.0:
                skew = abs(self._latest_images[cam_id]["ts"] - state_ts)
                if skew > self._max_cam_state_skew_s:
                    return None
        return {
            "cam_0": self._latest_images["cam_0"]["image"],
            "cam_1": self._latest_images["cam_1"]["image"],
        }

    def _build_obs(self, state_raw: dict, images: dict) -> dict:
        joints = np.array(state_raw["joints"], dtype=np.float64)  # 7D: [j1..j6, gripper_rad]
        # Pad arm joints to 7D for DROID (A1 has 6-DOF arm, DROID expects 7-DOF)
        joint7 = np.zeros(7, dtype=np.float64)
        joint7[:6] = joints[:6]
        gripper_01 = a1_gripper_to_droid(float(joints[6]))
        return {
            "observation/exterior_image_1_left": images["cam_0"],
            "observation/wrist_image_left": images["cam_1"],
            "observation/joint_position": joint7,
            "observation/gripper_position": np.array([gripper_01], dtype=np.float64),
            "prompt": self._prompt,
        }

    def _enqueue_actions(self, action_dict: dict):
        actions = np.asarray(action_dict["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        self._action_queue.clear()
        n = min(actions.shape[0], self._action_chunk_size)
        for i in range(n):
            # DROID output: [j1..j6, j7_padding, gripper_01]
            arm_joints = actions[i, :6].tolist()
            gripper_rad = droid_gripper_to_a1(float(np.clip(actions[i, 7], 0.0, 1.0)))
            self._action_queue.append({"joints": arm_joints + [gripper_rad]})

    def run(self):
        print("[Bridge] Running. Waiting for robot state and cameras...")
        infer_count = 0
        while True:
            try:
                obs_raw = self._sub.recv_json()
                state_ts = float(obs_raw.get("timestamp", 0.0))

                if state_ts > 0.0 and (time.time() - state_ts) > self._max_state_age_s:
                    self._stale_state_drops += 1
                    if self._stale_state_drops % 100 == 1:
                        print(f"[Bridge] Stale state age={time.time()-state_ts:.3f}s (#{self._stale_state_drops})")
                    continue

                # Drain queued actions from previous inference
                if self._action_queue:
                    action_out = self._action_queue.popleft()
                    action_out["timestamp"] = time.time()
                    self._pub.send_json(action_out)
                    print(f"[chunk {len(self._action_queue)} left] {action_out}")
                    continue

                # Need fresh inference
                images = self._get_cameras(state_ts)
                if images is None:
                    continue

                obs = self._build_obs(obs_raw, images)
                t0 = time.time()
                action_dict = self._policy.infer(obs)
                infer_ms = (time.time() - t0) * 1000

                self._enqueue_actions(action_dict)
                infer_count += 1

                action_out = self._action_queue.popleft()
                action_out["timestamp"] = time.time()
                self._pub.send_json(action_out)

                joints_str = ",".join(f"{v:.3f}" for v in action_out["joints"])
                print(f"[infer #{infer_count} {infer_ms:.0f}ms, chunk {len(self._action_queue)} left] joints=[{joints_str}]")

            except KeyboardInterrupt:
                print("[Bridge] Stopped.")
                break
            except Exception:
                print("[Bridge] ERROR:")
                print(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Zero-shot DROID bridge for A1 robot")
    parser.add_argument("--host", default="localhost", help="A1 ZMQ host")
    parser.add_argument("--server-host", default="localhost", help="pi05_droid server host")
    parser.add_argument("--server-port", type=int, default=8000, help="pi05_droid server port")
    parser.add_argument(
        "--prompt",
        default="swap the position of the marker and the yellow block through the white plate",
    )
    parser.add_argument("--action-chunk-size", type=int, default=2,
                        help="How many actions to execute before re-inferring")
    parser.add_argument("--max-state-age-s", type=float, default=0.5)
    parser.add_argument("--max-camera-age-s", type=float, default=0.5)
    parser.add_argument("--max-cam-state-skew-s", type=float, default=0.25)
    args = parser.parse_args()

    bridge = DroidZeroShotBridge(
        host=args.host,
        server_host=args.server_host,
        server_port=args.server_port,
        prompt=args.prompt,
        action_chunk_size=args.action_chunk_size,
        max_state_age_s=args.max_state_age_s,
        max_camera_age_s=args.max_camera_age_s,
        max_cam_state_skew_s=args.max_cam_state_skew_s,
    )
    bridge.run()


if __name__ == "__main__":
    main()
