from collections import deque
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
        norm_stats=None,
        use_quantile_norm=False,
        deterministic_inference=True,
        prompt="pick_twice",
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
        # State normalization for proprio_seq history (must match training pipeline).
        self._norm_stats = norm_stats
        self._use_quantile_norm = use_quantile_norm
        # Use deterministic (zero) noise to reduce stochastic trembling.
        self._deterministic_inference = deterministic_inference
        self._prompt = prompt
        model = getattr(self._policy, "_model", None)
        self._model_action_dim = int(getattr(model, "action_dim", 32))
        self._model_action_horizon = int(getattr(model, "action_horizon", 10))
        self._ltc_bptt_len = int(getattr(model, "bptt_len", 8))
        self._ltc_dt_s = float(self._metadata.get("ltc_dt_s", 1.0 / 50.0))
        # Keep inference dt aligned with training by default.
        # Set `ltc_use_measured_dt=true` in policy_metadata only if the model was
        # trained to be robust to variable dt.
        self._ltc_use_measured_dt = bool(self._metadata.get("ltc_use_measured_dt", False))
        self._ltc_dt_min_s = float(self._metadata.get("ltc_dt_min_s", self._ltc_dt_s * 0.5))
        self._ltc_dt_max_s = float(self._metadata.get("ltc_dt_max_s", self._ltc_dt_s * 2.0))
        self._ltc_episode_id = str(self._metadata.get("ltc_episode_id", "a1_zmq"))
        self._ltc_history_len = max(1, self._ltc_bptt_len)
        self._ltc_history_reset_gap_s = float(self._metadata.get("ltc_history_reset_gap_s", 1.0))
        self._ltc_state_history = deque(maxlen=self._ltc_history_len)
        self._ltc_time_history = deque(maxlen=self._ltc_history_len)
        # Drop stale buffered states from previous runs.
        self._max_state_age_s = 0.5
        self._stale_state_drop_count = 0
        # Require camera frames to be fresh and synchronized with state timestamps.
        self._max_camera_age_s = 0.5
        self._max_cam_state_skew_s = 0.25
        self._max_inter_cam_skew_s = 0.08
        self._stale_camera_drop_count = 0
        self._misaligned_camera_drop_count = 0

        self._context = zmq.Context()

        # -------- SUB: receive state --------
        self._sub = self._context.socket(zmq.SUB)
        # Keep only the freshest robot state and prevent backlog replay after reconnect.
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.RCVHWM, 1)
        self._sub.connect(f"tcp://{host}:{state_port}")
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # -------- PUB: publish action --------
        self._pub = self._context.socket(zmq.PUB)
        self._pub.bind(f"tcp://{host}:{action_port}")
        
        # -------- SUB: receive camera --------
        self._cam_sub = self._context.socket(zmq.SUB)
        # Bound camera backlog to reduce stale frame accumulation.
        self._cam_sub.setsockopt(zmq.RCVHWM, 20)
        self._cam_sub.connect(f"tcp://{host}:{camera_port}")
        self._cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")


        print(f"[ZMQPolicyServer] SUB connected to tcp://{host}:{state_port}")
        print(f"[ZMQPolicyServer] PUB bound to tcp://{host}:{action_port}")

        # Avoid first message drop
        time.sleep(0.3)

    
    def _normalize_state_vec(self, state: np.ndarray) -> np.ndarray:
        """Normalize a 1-D state vector using the policy's state norm stats.

        This must mirror the normalization applied to the state sequence during
        training so that ``proprio_seq`` is on the same scale the LTC backbone
        was trained with.
        """
        if self._norm_stats is None:
            return state
        stats = self._norm_stats.get("state", None)
        if stats is None:
            return state
        n = min(state.shape[0], len(stats.q01) if stats.q01 is not None else len(stats.mean))
        x = state.copy()
        if self._use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = np.asarray(stats.q01[:n], dtype=np.float32)
            q99 = np.asarray(stats.q99[:n], dtype=np.float32)
            x[:n] = (x[:n] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            mean = np.asarray(stats.mean[:n], dtype=np.float32)
            std = np.asarray(stats.std[:n], dtype=np.float32)
            x[:n] = (x[:n] - mean) / (std + 1e-6)
        return x

    def _parse_camera_timestamp_s(self, raw_ts: bytes):
        try:
            ts = float(raw_ts.decode("ascii", errors="strict"))
        except Exception:
            return None
        # Camera server publishes time.time_ns(); convert to seconds.
        if ts > 1e12:
            ts = ts / 1e9
        return ts

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
            cam_ts_s = self._parse_camera_timestamp_s(parts[1])
            if cam_ts_s is None:
                continue
            img_bytes = parts[2]

            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._latest_images[cam_id] = {"image": image, "timestamp_s": cam_ts_s}

    def _get_camera_images(self, state_ts: float):
        self._poll_camera_frames()
        now = time.time()
        camera_entries = {}
        for cam_id in self._required_cam_ids:
            if cam_id not in self._latest_images:
                return None
            entry = self._latest_images[cam_id]
            cam_ts = float(entry["timestamp_s"])
            if (now - cam_ts) > self._max_camera_age_s:
                self._stale_camera_drop_count += 1
                if self._stale_camera_drop_count % 100 == 1:
                    print(
                        "[ZMQPolicyServer] Dropping stale camera frame "
                        f"(cam={cam_id}, age={now - cam_ts:.3f}s, count={self._stale_camera_drop_count})"
                    )
                return None
            camera_entries[cam_id] = entry

        cam0_ts = float(camera_entries["cam_0"]["timestamp_s"])
        cam1_ts = float(camera_entries["cam_1"]["timestamp_s"])
        inter_cam_skew = abs(cam0_ts - cam1_ts)
        if inter_cam_skew > self._max_inter_cam_skew_s:
            self._misaligned_camera_drop_count += 1
            if self._misaligned_camera_drop_count % 100 == 1:
                print(
                    "[ZMQPolicyServer] Dropping unsynced camera pair "
                    f"(cam_skew={inter_cam_skew:.3f}s, count={self._misaligned_camera_drop_count})"
                )
            return None

        if state_ts > 0.0:
            for cam_id in self._required_cam_ids:
                cam_ts = float(camera_entries[cam_id]["timestamp_s"])
                skew = abs(cam_ts - state_ts)
                if skew > self._max_cam_state_skew_s:
                    self._misaligned_camera_drop_count += 1
                    if self._misaligned_camera_drop_count % 100 == 1:
                        print(
                            "[ZMQPolicyServer] Dropping state/camera mismatch "
                            f"(cam={cam_id}, skew={skew:.3f}s, count={self._misaligned_camera_drop_count})"
                        )
                    return None

        return {"cam_0": camera_entries["cam_0"]["image"], "cam_1": camera_entries["cam_1"]["image"]}

    def _build_ltc_history(self, padded_state: np.ndarray, timestamp: float):
        # Reset history on large time gaps or non-monotonic timestamps.
        reset_ltc_state = not self._ltc_time_history
        ltc_dt_s = np.float32(self._ltc_dt_s)
        if self._ltc_time_history:
            last_ts = float(self._ltc_time_history[-1])
            dt = float(timestamp - last_ts)
            if timestamp < last_ts or dt > self._ltc_history_reset_gap_s:
                self._ltc_state_history.clear()
                self._ltc_time_history.clear()
                reset_ltc_state = True
            elif self._ltc_use_measured_dt and dt > 0.0:
                ltc_dt_s = np.float32(np.clip(dt, self._ltc_dt_min_s, self._ltc_dt_max_s))

        self._ltc_state_history.append(np.asarray(padded_state, dtype=np.float32).copy())
        self._ltc_time_history.append(float(timestamp))

        states = list(self._ltc_state_history)
        times = list(self._ltc_time_history)

        if len(states) < self._ltc_history_len:
            pad_count = self._ltc_history_len - len(states)
            states = [states[0]] * pad_count + states
            times = [times[0]] * pad_count + times

        proprio_seq = np.stack(states[-self._ltc_history_len :], axis=0).astype(np.float32, copy=False)
        ts = np.asarray(times[-self._ltc_history_len :], dtype=np.float32)
        deltas = np.diff(ts, prepend=ts[0]).astype(np.float32)
        deltas = np.maximum(deltas, 0.0)

        if np.all(deltas <= 0.0):
            deltas[:] = np.float32(self._ltc_dt_s)
        elif deltas[0] <= 0.0:
            positive = deltas[deltas > 0.0]
            deltas[0] = positive[0] if positive.size > 0 else np.float32(self._ltc_dt_s)

        ltc_dt = np.asarray([ltc_dt_s], dtype=np.float32)
        return proprio_seq, deltas[:, None], ltc_dt, bool(reset_ltc_state)

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
        # Normalize state before storing in proprio history so that proprio_seq
        # matches the normalized state sequences seen during training.
        normalized_state = self._normalize_state_vec(state)
        padded_normalized = np.zeros((self._model_action_dim,), dtype=np.float32)
        padded_normalized[: normalized_state.shape[0]] = normalized_state
        timestamp = float(data.get("timestamp", time.time()))
        proprio_seq, time_deltas, ltc_dt, reset_ltc_state = self._build_ltc_history(padded_normalized, timestamp)
        obs = {
            "cam_0": images["cam_0"],
            "cam_1": images["cam_1"],
            "state": state,
            "proprio_seq": proprio_seq,
            "time_deltas": time_deltas,
            "ltc_dt": ltc_dt,
            "episode_id": self._ltc_episode_id,
            "reset": np.bool_(reset_ltc_state),
            "action": np.zeros(state.shape, dtype=np.float32),
            "prompt": self._prompt,
            "observation.timestamp": np.float32(timestamp),
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
        print(f"[ZMQPolicyServer] deterministic_inference={self._deterministic_inference}, prompt='{self._prompt}'")
        print(
            "[ZMQPolicyServer] Training start reference: "
            "pos≈(0.078, -0.028, 0.238) ori≈(0.03, 0.72, -0.05, 0.69) gripper≈57.6"
        )
        # Pre-allocate deterministic zero noise to suppress stochastic trembling.
        _det_noise = (
            np.zeros((self._model_action_horizon, self._model_action_dim), dtype=np.float32)
            if self._deterministic_inference
            else None
        )
        _infer_step = 0
        _DIAG_STEPS = 20  # Log detailed diagnostics for first N inference steps.
        while True:
            try:
                obs_raw = self._sub.recv_json()
                state_ts = float(obs_raw.get("timestamp", 0.0))
                if state_ts > 0.0 and (time.time() - state_ts) > self._max_state_age_s:
                    self._stale_state_drop_count += 1
                    if self._stale_state_drop_count % 100 == 1:
                        print(
                            "[ZMQPolicyServer] Dropping stale state "
                            f"(age={time.time() - state_ts:.3f}s, count={self._stale_state_drop_count})"
                        )
                    continue

                images = self._get_camera_images(state_ts)
                if images is None:
                    continue

                obs = self._convert_obs(obs_raw, images)

                # Diagnostic logging for the first N steps to debug divergence.
                if _infer_step < _DIAG_STEPS:
                    _s = obs.get("state", None)
                    _ps = obs.get("proprio_seq", None)
                    _reset = obs.get("reset", None)
                    print(
                        f"[DIAG step={_infer_step}] "
                        f"state={np.array2string(_s, precision=4, separator=', ') if _s is not None else 'N/A'} "
                        f"reset={_reset}"
                    )
                    if _ps is not None:
                        print(f"  proprio_seq[-1]={np.array2string(_ps[-1, :8], precision=4, separator=', ')}")

                action_dict = self._policy.infer(obs, noise=_det_noise)
                action_out = self._convert_action(action_dict)
                self._pub.send_json(action_out)

                if _infer_step < _DIAG_STEPS:
                    print(
                        f"  → action: pos=({action_out['pos'][0]:.4f}, {action_out['pos'][1]:.4f}, {action_out['pos'][2]:.4f}) "
                        f"ori=({action_out['ori'][0]:.4f}, {action_out['ori'][1]:.4f}, {action_out['ori'][2]:.4f}, {action_out['ori'][3]:.4f}) "
                        f"gripper={action_out['gripper']:.2f}"
                    )
                else:
                    print("action out", action_out)
                    print()
                breakpoint()

                _infer_step += 1
            except Exception:
                print("[ZMQPolicyServer] ERROR:")
                print(traceback.format_exc())
