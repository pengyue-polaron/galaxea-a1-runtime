import cv2
import zmq
import numpy as np
import time
import traceback


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
        self._cam_sub.connect(f"tcp://{host}:{camera_port}")  # camera port
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")


        print(f"[ZMQPolicyServer] SUB connected to tcp://{host}:{state_port}")
        print(f"[ZMQPolicyServer] PUB bound to tcp://{host}:{action_port}")

        # Avoid first message drop
        time.sleep(0.3)

    
    def _get_camera_image(self):
        try:
            topic, img_bytes = self._cam_sub.recv_multipart(flags=zmq.NOBLOCK)

            # decode jpeg
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            return image

        except zmq.Again:
            # no frame available
            return None

    # ============================================================
    # Observation Conversion
    # ============================================================
    def _convert_obs(self, data, image):
        """
        Convert ZMQ state dict into policy input format.

        Incoming:
            {
                'timestamp': ...,
                'pos': (x,y,z),
                'ori': (qx,qy,qz,qw),
                'gripper': value
            }

        Required policy format:
            {
                "image/cam_0_rgb": HWC uint8,
                "state": (8,),
                "action":, # place holder, for repack transform
                "prompt": str,
            }
        """

        pos = data["pos"]
        ori = data["ori"]
        gripper = data["gripper"]

        # Build state vector (8,)
        state = np.array([
            pos[0], pos[1], pos[2],
            ori[0], ori[1], ori[2], ori[3],
            gripper
        ], dtype=np.float32)


        obs = {
            "cam_0": image,
            "state": state,
            "action": np.zeros(state.shape, dtype=np.float32),
            "prompt": "do something",
        }

        # Debug check
        if state.shape != (8,):
            raise ValueError(f"State shape must be (8,), got {state.shape}")

        return obs

    # ============================================================
    # Action Conversion
    # ============================================================
    def _convert_action(self, action_dict):
        actions = np.asarray(action_dict["actions"], dtype=np.float32)

        if actions.ndim == 2:
            action = actions[0]
        else:
            action = actions

        if action.shape[0] != 8:
            raise ValueError(f"Expected 8-dim action, got {action.shape}")

        x, y, z = action[:3]
        qx, qy, qz, qw = action[3:7]
        gripper = action[7]

        return {
            "timestamp": time.time(),
            "pos": (float(x), float(y), float(z)),
            "ori": (float(qx), float(qy), float(qz), float(qw)),
            "gripper": float(gripper),
        }


 # ============================================================
    # Main Loop
    # ============================================================
    def run(self):
        print("[ZMQPolicyServer] Running...")
        while True:
            try:
                start_time = time.time()

                # 1️⃣ Receive state
                obs_raw = self._sub.recv_json()

                # 2️⃣ Get latest image
                image = self._get_camera_image()

                if image is None:
                    continue  # skip until image ready

                # 3️⃣ Convert
                obs = self._convert_obs(obs_raw, image)
                # 3️⃣ Inference
                infer_start = time.time()
                action_dict = self._policy.infer(obs)
                infer_time = (time.time() - infer_start) * 1000.0

                # 4️⃣ Convert action for A1
                action_out = self._convert_action(action_dict)

               
                # 5️⃣ Publish action
                self._pub.send_json(action_out)
                print(3)
                print(action_out)

            except Exception:
                print("[ZMQPolicyServer] ERROR:")
                print(traceback.format_exc())
