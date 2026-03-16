"""ZMQ ↔ WebSocket bridge for A1 robot inference.

Reads robot state + camera frames from ZMQ (published by a1_server + camera_server),
calls the openpi WebSocket policy server for inference, and publishes the first
action of the returned sequence to ZMQ for the A1 server to forward to the robot.

One inference call → one action published → robot moves to that joint target.

Usage (alongside `just policy`):
    just zmq-bridge
    just zmq-bridge nyushrobo5090 8000 "pick up the cup"
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
    p.add_argument(
        "--prompt", default="swap the position of the marker and the yellow block"
    )
    p.add_argument(
        "--step-mode",
        action="store_true",
        help="Manual stepping: press Enter to run one infer->publish step",
    )
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
        self.step_mode = bool(args.step_mode)

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
        print(
            f"[Bridge] Connected. Server metadata: {self._policy.get_server_metadata()}"
        )

        time.sleep(0.3)

    def _poll_cameras(self):
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

    def _get_cameras(self) -> dict | None:
        self._poll_cameras()
        for cam_id in ("cam_0", "cam_1"):
            if cam_id not in self._latest_images:
                return None
        return {
            k: v["image"]
            for k, v in self._latest_images.items()
            if k in ("cam_0", "cam_1")
        }

    def run(self):
        print("[Bridge] Running. Waiting for state + camera data ...")
        step_count = 0

        if self.step_mode:
            print("[Bridge] Step mode enabled.")
            print("[Bridge] Press Enter to infer one step. Type 'q' + Enter to quit.")

        while True:
            if self.step_mode:
                try:
                    user_cmd = input("[Bridge] Next step (Enter/q): ").strip().lower()
                except EOFError:
                    user_cmd = "q"
                if user_cmd in {"q", "quit", "exit"}:
                    print("[Bridge] Exiting.")
                    break

            # --- read latest state ---
            try:
                state_msg = self._state_sub.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue

            if not isinstance(state_msg, dict):
                continue

            try:
                state_ts = float(state_msg.get("timestamp", 0.0))
            except (TypeError, ValueError):
                continue

            if (time.time() - state_ts) > self.args.max_state_age_s:
                continue

            joints = state_msg.get("joints")
            if not isinstance(joints, (list, tuple, np.ndarray)):
                continue

            # --- get camera frames ---
            cameras = self._get_cameras()
            if cameras is None:
                time.sleep(0.01)
                continue

            # --- build observation ---
            obs = {
                "observation/image": cameras["cam_0"],
                "observation/wrist_image": cameras["cam_1"],
                "observation/state": np.array(joints, dtype=np.float32),
                "prompt": self.prompt,
            }

            # --- call policy ---
            try:
                result = self._policy.infer(obs)
            except Exception as e:
                print(f"[Bridge] Inference error: {e}")
                continue

            if not isinstance(result, dict) or "actions" not in result:
                print("[Bridge] Inference output missing 'actions'.")
                continue

            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[np.newaxis, :]
            if actions.ndim != 2 or actions.shape[0] == 0:
                print(f"[Bridge] Unexpected action shape: {actions.shape}")
                continue

            # --- publish first action only ---
            action = actions[0]
            self._action_pub.send_json(
                {
                    "timestamp": time.time(),
                    "joints": [float(j) for j in action],
                }
            )

            step_count += 1
            print(f"[Bridge] Step {step_count}: published action {np.round(action, 3).tolist()}")


def main():
    args = parse_args()
    bridge = ZmqWsBridge(args)
    bridge.run()


if __name__ == "__main__":
    main()
