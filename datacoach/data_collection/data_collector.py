import pickle
import select
import signal
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import zmq
from omegaconf import DictConfig

from datacoach.constants import CAM_FPS, ZMQ_CAM_PORT, ZMQ_CMD_PORT, ZMQ_STATE_PORT


class ReplayAction(Enum):
    """Action to take after replay completes."""
    CLOSE = "close"
    REPLAY = "replay"


class DataCollector:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Delay output directory creation until recording actually starts.
        self.base_dir = None

        # === Camera setup ===
        camera_fps_value = self._cfg_get(cfg, "camera_fps", CAM_FPS)
        self.camera_fps = int(camera_fps_value) if camera_fps_value is not None else CAM_FPS
        self.wait_for_enter = bool(self._cfg_get(cfg, "wait_for_enter", True))
        self.min_camera_coverage_ratio = float(self._cfg_get(cfg, "min_camera_coverage_ratio", 0.9))
        self.max_camera_start_lag_s = float(self._cfg_get(cfg, "max_camera_start_lag_s", 1.0))
        self.max_camera_end_lag_s = float(self._cfg_get(cfg, "max_camera_end_lag_s", 1.0))
        self.camera_ids = self._parse_camera_ids(cfg)
        print(f"🎥 Expecting camera topics: {', '.join(self.camera_ids)}")

        # === Buffers ===
        self.state_data = []
        self.cmd_data = []
        self.image_buffers = {cam_id: [] for cam_id in self.camera_ids}

        self.lock = threading.Lock()
        self.recording = False
        self.exit_flag = False
        self.abort_reason = None
        self.save_on_stop = False

        # === ZMQ Setup ===
        context = zmq.Context()

        # Robot state
        self.zmq_state_sub = context.socket(zmq.SUB)
        self.zmq_state_sub.connect(f"tcp://127.0.0.1:{ZMQ_STATE_PORT}")
        self.zmq_state_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Commanded state
        self.zmq_cmd_sub = context.socket(zmq.SUB)
        self.zmq_cmd_sub.connect(f"tcp://127.0.0.1:{ZMQ_CMD_PORT}")
        self.zmq_cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Camera images
        self.zmq_cam_sub = context.socket(zmq.SUB)
        self.zmq_cam_sub.connect(f"tcp://127.0.0.1:{ZMQ_CAM_PORT}")
        self.zmq_cam_sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # === Threads ===
        self.state_thread = threading.Thread(target=self.collect_state, daemon=True)
        self.cmd_thread = threading.Thread(target=self.collect_commanded_state, daemon=True)
        self.image_thread = threading.Thread(target=self.collect_images, daemon=True)

        # === Signal handler ===
        signal.signal(signal.SIGINT, self.stop_recording)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _build_base_dir(self, cfg):
        storage_root = Path(cfg.storage_path) / cfg.task_name
        demo_index = self._cfg_get(cfg, "demo_index", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"demo_{demo_index}_{timestamp}"
        base_dir = storage_root / base_name

        suffix = 1
        while base_dir.exists():
            base_dir = storage_root / f"{base_name}_{suffix}"
            suffix += 1

        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir

    def _parse_camera_ids(self, cfg):
        cameras = self._cfg_get(cfg, "cameras", None)
        if not cameras:
            return ["cam_0"]

        ids = []
        for camera in cameras:
            if isinstance(camera, str):
                cam_id = camera
            else:
                cam_id = self._cfg_get(camera, "id", None)
            if cam_id:
                ids.append(str(cam_id))

        if not ids:
            return ["cam_0"]

        # Keep order while removing duplicates.
        return list(dict.fromkeys(ids))

    @staticmethod
    def _decode_timestamp(ts_bytes):
        try:
            ts = float(ts_bytes.decode())
        except Exception:
            return None

        # Support ns timestamps from the new camera server format.
        if ts > 1e12:
            ts = ts / 1e9
        return ts

    def _ensure_camera(self, cam_id: str):
        if cam_id not in self.image_buffers:
            self.image_buffers[cam_id] = []
            print(f"[Collector] Detected new camera topic: {cam_id}")

    def _estimate_video_fps(self, timestamps):
        if len(timestamps) < 2:
            return float(self.camera_fps)
        duration = float(timestamps[-1]) - float(timestamps[0])
        if duration <= 1e-6:
            return float(self.camera_fps)
        fps = (len(timestamps) - 1) / duration
        if not np.isfinite(fps) or fps <= 0:
            return float(self.camera_fps)
        # Guard against pathological timestamps.
        return float(min(max(fps, 1.0), 240.0))

    def _validate_recording_or_raise(self):
        if not self.state_data:
            raise RuntimeError("No robot states captured; aborting save.")
        if not self.cmd_data:
            raise RuntimeError("No commanded states captured; aborting save.")

        record_start = min(
            float(self.state_data[0]["timestamp"]),
            float(self.cmd_data[0]["timestamp"]),
        )
        record_end = max(
            float(self.state_data[-1]["timestamp"]),
            float(self.cmd_data[-1]["timestamp"]),
        )
        record_duration = max(0.0, record_end - record_start)

        missing_cameras = [cam_id for cam_id in self.camera_ids if not self.image_buffers.get(cam_id)]
        if missing_cameras:
            raise RuntimeError(
                f"Missing camera frames for {', '.join(missing_cameras)}; aborting save."
            )

        for cam_id in self.camera_ids:
            frames = self.image_buffers[cam_id]
            timestamps = [float(ts) for ts, _ in frames]
            if len(timestamps) < 2:
                raise RuntimeError(
                    f"Camera {cam_id} captured too few frames ({len(timestamps)}); aborting save."
                )

            cam_start = timestamps[0]
            cam_end = timestamps[-1]
            cam_duration = max(0.0, cam_end - cam_start)
            start_lag = max(0.0, cam_start - record_start)
            end_lag = max(0.0, record_end - cam_end)
            coverage_ratio = 1.0 if record_duration <= 1e-6 else cam_duration / record_duration

            if start_lag > self.max_camera_start_lag_s:
                raise RuntimeError(
                    f"Camera {cam_id} started too late ({start_lag:.2f}s > "
                    f"{self.max_camera_start_lag_s:.2f}s); aborting save."
                )
            if end_lag > self.max_camera_end_lag_s:
                raise RuntimeError(
                    f"Camera {cam_id} ended too early ({end_lag:.2f}s before replay end); "
                    "possible camera disconnect, aborting save."
                )
            if coverage_ratio < self.min_camera_coverage_ratio:
                raise RuntimeError(
                    f"Camera {cam_id} coverage too short ({coverage_ratio:.1%} < "
                    f"{self.min_camera_coverage_ratio:.1%}); aborting save."
                )

    def _cleanup_empty_base_dir(self):
        if self.base_dir is None:
            return
        try:
            self.base_dir.rmdir()
        except OSError:
            pass

    def _request_stop(self, *, save_recording: bool, reason: str | None = None, announce: str | None = None):
        if announce:
            print(announce)
        self.recording = False
        self.save_on_stop = self.save_on_stop or save_recording
        if reason and self.abort_reason is None:
            self.abort_reason = reason
        self.exit_flag = True

    def abort(self, reason: str, *, save_recording: bool = False):
        self._request_stop(save_recording=save_recording, reason=reason)

    def _join_worker_threads(self):
        for thread in (self.state_thread, self.cmd_thread, self.image_thread):
            if thread.is_alive():
                thread.join(timeout=1.0)

    def _wait_for_start_signal(self):
        if not self.wait_for_enter:
            print("Auto start enabled. Starting recording immediately...")
            return

        print("Press ENTER to start collecting data...")
        while not self.exit_flag:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.readline()
                return

    def _prompt_replay_action(self) -> ReplayAction:
        """Prompt user to choose action after replay completes."""
        print("\n" + "=" * 50)
        print("Replay finished! Choose next action:")
        print("  [1] Close")
        print("  [2] Replay again")
        print("=" * 50)

        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                user_input = sys.stdin.readline().strip()
                if user_input == "1":
                    return ReplayAction.CLOSE
                elif user_input == "2":
                    return ReplayAction.REPLAY
                else:
                    print("Invalid input. Please enter 1 or 2:")

    def collect_state(self):
        print(f"📡 Subscribing to robot state at tcp://127.0.0.1:{ZMQ_STATE_PORT}")
        while not self.exit_flag:
            try:
                msg = self.zmq_state_sub.recv_json(flags=zmq.NOBLOCK)
                if self.recording:
                    with self.lock:
                        ts = float(msg.get("timestamp", time.time())) if isinstance(msg, dict) else time.time()
                        self.state_data.append({"timestamp": ts, "data": msg})
            except zmq.Again:
                time.sleep(0.005)

    def collect_commanded_state(self):
        print(f"📡 Subscribing to commanded state at tcp://127.0.0.1:{ZMQ_CMD_PORT}")
        while not self.exit_flag:
            try:
                msg = self.zmq_cmd_sub.recv_json(flags=zmq.NOBLOCK)
                if self.recording:
                    with self.lock:
                        ts = float(msg.get("timestamp", time.time())) if isinstance(msg, dict) else time.time()
                        self.cmd_data.append({"timestamp": ts, "data": msg})
            except zmq.Again:
                time.sleep(0.005)

    def collect_images(self):
        print(f"📡 Subscribing to camera stream at tcp://127.0.0.1:{ZMQ_CAM_PORT}")
        print("📡 Expected multipart format: [camera_id, timestamp, jpeg_bytes]")

        while not self.exit_flag:
            try:
                parts = self.zmq_cam_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.005)
                continue

            # Backward compatibility: [timestamp, jpeg_bytes]
            if len(parts) == 2:
                cam_id = "cam_0"
                ts_bytes, img_bytes = parts
            elif len(parts) == 3:
                cam_id = parts[0].decode("utf-8", errors="replace")
                ts_bytes, img_bytes = parts[1], parts[2]
            else:
                continue

            timestamp = self._decode_timestamp(ts_bytes)
            if timestamp is None:
                continue

            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            if self.recording:
                with self.lock:
                    self._ensure_camera(cam_id)
                    self.image_buffers[cam_id].append((timestamp, img.copy()))

    def save_all(self):
        if self.base_dir is None:
            print("ℹ️ No recording session started, nothing to save.")
            return

        with self.lock:
            self._validate_recording_or_raise()

        print("💾 Saving data...")

        with self.lock:
            with open(self.base_dir / "states.pkl", "wb") as f:
                pickle.dump(self.state_data, f)

            with open(self.base_dir / "commanded_states.pkl", "wb") as f:
                pickle.dump(self.cmd_data, f)

            saved_cameras = []
            for cam_idx, cam_id in enumerate(sorted(self.image_buffers.keys())):
                frames = self.image_buffers[cam_id]
                if not frames:
                    continue

                video_path = self.base_dir / f"{cam_id}_rgb_video.mp4"
                timestamps = [float(ts) for ts, _ in frames]
                video_fps = self._estimate_video_fps(timestamps)

                first_img = frames[0][1]
                h, w = first_img.shape[:2]
                fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open video writer for {video_path}")

                for _, img in frames:
                    if img is None:
                        continue
                    ih, iw = img.shape[:2]
                    if ih != h or iw != w:
                        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                    writer.write(img)
                writer.release()

                metadata = {
                    "cam_idx": cam_idx,
                    "cam_name": cam_id,
                    "cam_fps": video_fps,
                    "cam_fps_configured": self.camera_fps,
                    "num_image_frames": len(timestamps),
                    "timestamps": timestamps,
                    "record_start_time": timestamps[0],
                    "record_end_time": timestamps[-1],
                    "filename": str(video_path),
                }

                with open(self.base_dir / f"{cam_id}_rgb_video.metadata", "wb") as f:
                    pickle.dump(metadata, f)

                print(f"  - {cam_id}: write_fps={video_fps:.2f}, frames={len(timestamps)}")
                saved_cameras.append(cam_id)

        print("✅ Saved:")
        print(f"  - states.pkl ({len(self.state_data)})")
        print(f"  - commanded_states.pkl ({len(self.cmd_data)})")
        if saved_cameras:
            for cam_id in saved_cameras:
                print(f"  - {cam_id}_rgb_video.mp4")
                print(f"  - {cam_id}_rgb_video.metadata")
        else:
            print("  - no camera frames captured")

    def stop_recording(self, *args):
        self._request_stop(save_recording=True, announce="\n⏹️ Ctrl+C detected, stopping...")

    def run(self) -> ReplayAction:
        """Run the data collection loop.

        Returns:
            ReplayAction: The action chosen by the user after completion.
        """
        self.state_thread.start()
        self.cmd_thread.start()
        self.image_thread.start()

        self._wait_for_start_signal()

        if self.abort_reason is not None:
            self._join_worker_threads()
            raise RuntimeError(self.abort_reason)

        with self.lock:
            self.state_data.clear()
            self.cmd_data.clear()
            for frames in self.image_buffers.values():
                frames.clear()
            self.base_dir = self._build_base_dir(self.cfg)

        print(f"📂 Saving data to: {self.base_dir}")
        self.recording = True
        print("▶️ Recording started. Press Ctrl+C to stop.")

        while not self.exit_flag:
            time.sleep(0.1)

        self._join_worker_threads()

        if self.save_on_stop:
            try:
                self.save_all()
            except Exception as exc:
                print(f"❌ Validation failed, recording discarded: {exc}")
                self._cleanup_empty_base_dir()
                raise
        else:
            self._cleanup_empty_base_dir()

        print("🧹 Collector run() exiting")
        if self.abort_reason is not None:
            raise RuntimeError(self.abort_reason)

        # Prompt user for next action
        return self._prompt_replay_action()


def main(cfg: DictConfig) -> ReplayAction:
    collector = DataCollector(cfg)
    return collector.run()


if __name__ == "__main__":
    raise SystemExit("Use scripts/collect_data/run_data_collection.py to run this module.")
