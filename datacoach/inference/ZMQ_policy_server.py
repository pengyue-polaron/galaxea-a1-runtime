import cv2
import numpy as np
import time
import traceback
import zmq


class ZMQPolicyServer:
    """
    ZMQ-based Policy Server

    Architecture:
        A1 ROS → ZMQ PUB (state)
                 ↓
        ZMQPolicyServer (SUB)
                 ↓ convert
        policy.infer
                 ↓ convert
        ZMQ PUB (action)
                 ↓
        A1 ROS subscriber
    """

    def __init__(
        self,
        policy,
        host,
        state_port,
        action_port,
        camera_port,
        metadata=None,
    ):
        self._policy = policy
        self._host = host
        self._state_port = state_port
        self._action_port = action_port
        self._camera_port = camera_port
        self._metadata = metadata or {}
        self._last_gripper = 0.0
        self._required_cam_ids = ("cam_0", "cam_1")
        self._latest_images = {}
        self._warned_bad_cam_message = False

        self._context = zmq.Context()

        # -------- SUB: receive state --------
        self._sub = self._context.socket(zmq.SUB)
        self._sub.connect(f"tcp://{host}:{state_port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # -------- PUB: publish action --------
        self._pub = self._context.socket(zmq.PUB)
        self._pub.bind(f"tcp://{host}:{action_port}")
        
        # -------- SUB: receive camera --------
        self._cam_sub = self._context.socket(zmq.SUB)
        self._cam_sub.connect(f"tcp://{host}:{camera_port}")
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")


        print(f"[ZMQPolicyServer] SUB connected to tcp://{host}:{state_port}")
        print(f"[ZMQPolicyServer] PUB bound to tcp://{host}:{action_port}")

        # Avoid first message drop
        time.sleep(0.3)

    
    def _poll_camera_frames(self):
        while True:
            try:
                parts = self._cam_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            # Strict dual-camera protocol: [camera_id, timestamp, jpeg_bytes].
            if len(parts) != 3:
                if not self._warned_bad_cam_message:
                    print(
                        "[ZMQPolicyServer] WARNING: camera message must have 3 parts "
                        "[cam_id, timestamp, jpeg_bytes]."
                    )
                    self._warned_bad_cam_message = True
                continue

            cam_id = parts[0].decode("utf-8", errors="replace")
            img_bytes = parts[2]

            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                continue
            self._latest_images[cam_id] = image

    def _get_camera_images(self):
        self._poll_camera_frames()
        for cam_id in self._required_cam_ids:
            if cam_id not in self._latest_images:
                return None
        return {
            "cam_0": self._latest_images["cam_0"],
            "cam_1": self._latest_images["cam_1"],
        }

    def _convert_obs(self, data, images):
        pos = data["pos"]
        ori = data["ori"]
        gripper = data["gripper"]
        self._last_gripper = float(gripper)

        state = np.array(
            [
                pos[0],
                pos[1],
                pos[2],
                ori[0],
                ori[1],
                ori[2],
                ori[3],
                gripper,
            ],
            dtype=np.float32,
        )

        timestamp = np.float32(data.get("timestamp", time.time()))
        obs = {
            "cam_0": images["cam_0"],
            "cam_1": images["cam_1"],
            "state": state,
            "action": np.zeros(state.shape, dtype=np.float32),
            "prompt": "pick twice",
            "observation.timestamp": timestamp,
            "observation.timestamp_is_pad": np.bool_(False),
            "state_is_pad": np.bool_(False),
        }

        if state.shape != (8,):
            raise ValueError(f"State shape must be (8,), got {state.shape}")

        return obs

    def _convert_action(self, action_dict):
        actions = np.asarray(action_dict["actions"], dtype=np.float32)

        if actions.ndim == 2:
            action = actions[0]
        else:
            action = actions

        if action.shape[0] == 7:
            action = np.concatenate([action, np.array([self._last_gripper], dtype=np.float32)])
        elif action.shape[0] != 8:
            raise ValueError(f"Expected 7 or 8-dim action, got {action.shape}")

        x, y, z = action[:3]
        qx, qy, qz, qw = action[3:7]
        q = np.array([qx, qy, qz, qw], dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 1e-8:
            q = q / q_norm
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        qx, qy, qz, qw = q.tolist()
        gripper = action[7]

        return {
            "timestamp": time.time(),
            "pos": (float(x), float(y), float(z)),
            "ori": (float(qx), float(qy), float(qz), float(qw)),
            "gripper": float(gripper),
        }

    def run(self):
        print("[ZMQPolicyServer] Running...")
        print("[ZMQPolicyServer] Expecting camera ids: cam_0, cam_1")
        while True:
            try:
                obs_raw = self._sub.recv_json()

                images = self._get_camera_images()
                if images is None:
                    continue

                obs = self._convert_obs(obs_raw, images)
                action_dict = self._policy.infer(obs)
                action_out = self._convert_action(action_dict)
                self._pub.send_json(action_out)
                print(action_out)
            except Exception:
                print("[ZMQPolicyServer] ERROR:")
                print(traceback.format_exc())
