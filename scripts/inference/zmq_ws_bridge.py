"""ZMQ ↔ WebSocket bridge for A1 robot inference.

Reads robot state + camera frames from ZMQ (published by a1_server + camera_server),
calls the openpi WebSocket policy server for inference, and publishes joint actions
back to ZMQ for the A1 server to consume.

Usage (alongside `just policy`):
    just zmq-bridge
    just zmq-bridge --prompt "pick up the cup" --action-chunk-size 3
"""

import argparse
import time

import cv2
import numpy as np
import zmq

from openpi_client import websocket_client_policy as _ws_policy

STATE_PORT = 5557
CAM_PORT = 5558
ACTION_PORT = 5559
HOST = "127.0.0.1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=HOST, help="WebSocket server host")
    p.add_argument("--port", type=int, default=8000, help="WebSocket server port")
    p.add_argument("--prompt", default="swap the position of the marker and the yellow block")
    p.add_argument("--action-chunk-size", type=int, default=2,
                   help="How many consecutive actions to execute before re-querying the policy")
    p.add_argument("--state-port", type=int, default=STATE_PORT)
    p.add_argument("--cam-port", type=int, default=CAM_PORT)
    p.add_argument("--action-port", type=int, default=ACTION_PORT)
    p.add_argument("--zmq-host", default=HOST)
    p.add_argument("--max-state-age-s", type=float, default=5.0)
    return p.parse_args()


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _parse_cam_timestamp(raw: bytes) -> float | None:
    try:
        ts = float(raw.decode("ascii"))
        if ts > 1e12:
            ts /= 1e9
        return ts
    except Exception:
        return None


class ZmqWsBridge:
    def __init__(self, args):
        self.args = args
        self.prompt = args.prompt
        self.action_chunk_size = args.action_chunk_size

        ctx = zmq.Context()

        self._state_sub = ctx.socket(zmq.SUB)
        self._state_sub.setsockopt(zmq.CONFLATE, 1)
        self._state_sub.setsockopt(zmq.RCVHWM, 1)
        self._state_sub.connect(f"tcp://{args.zmq_host}:{args.state_port}")
        self._state_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self._cam_sub = ctx.socket(zmq.SUB)
        self._cam_sub.setsockopt(zmq.RCVHWM, 4)
        self._cam_sub.connect(f"tcp://{args.zmq_host}:{args.cam_port}")
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        self._action_pub = ctx.socket(zmq.PUB)
        self._action_pub.bind(f"tcp://{args.zmq_host}:{args.action_port}")

        self._latest_images: dict = {}

        print(f"[Bridge] Connecting to WebSocket at ws://{args.host}:{args.port} ...")
        self._policy = _ws_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        print(f"[Bridge] Connected. Server metadata: {self._policy.get_server_metadata()}")

        time.sleep(0.3)

    def _poll_cameras(self):
        # Drain up to 10 frames so we get fresh entries for both cam_0 and cam_1.
        for _ in range(10):
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
            self._latest_images[cam_id] = {"image": img, "timestamp_s": ts}

    def _get_cameras(self, state_ts: float) -> dict | None:
        self._poll_cameras()
        for cam_id in ("cam_0", "cam_1"):
            if cam_id not in self._latest_images:
                return None
        return {k: v["image"] for k, v in self._latest_images.items() if k in ("cam_0", "cam_1")}

    def run(self):
        print("[Bridge] Running. Waiting for state + camera data ...")
        action_queue: list[np.ndarray] = []

        while True:
            # --- if we have queued actions, publish the next one ---
            if action_queue:
                joints = action_queue.pop(0)
                self._action_pub.send_json({
                    "timestamp": time.time(),
                    "joints": [float(j) for j in joints],
                })
                time.sleep(1.0 / 50.0)
                continue

            # --- read latest state ---
            try:
                state_msg = self._state_sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue

            state_ts = float(state_msg.get("timestamp", 0.0))
            now = time.time()
            if (now - state_ts) > self.args.max_state_age_s:
                continue

            joints = state_msg.get("joints")
            if joints is None:
                continue

            # --- get camera frames ---
            cameras = self._get_cameras(state_ts)
            if cameras is None:
                time.sleep(0.01)
                continue

            # --- build observation for pi0_a1_single_arm ---
            obs = {
                "observation/image": cameras["cam_0"],          # HWC uint8
                "observation/wrist_image": cameras["cam_1"],    # HWC uint8
                "observation/state": np.array(joints, dtype=np.float32),
                "prompt": self.prompt,
            }

            # --- call WebSocket policy server ---
            try:
                result = self._policy.infer(obs)
            except Exception as e:
                print(f"[Bridge] Inference error: {e}")
                continue

            actions = np.asarray(result["actions"])  # [action_horizon, 7]

            # --- queue up the first action_chunk_size actions ---
            for i in range(min(self.action_chunk_size, len(actions))):
                action_queue.append(actions[i])


def main():
    args = parse_args()
    bridge = ZmqWsBridge(args)
    bridge.run()


if __name__ == "__main__":
    main()
