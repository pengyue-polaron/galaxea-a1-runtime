import pickle
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
from omegaconf import DictConfig

from datacoach.constants import CAM_FPS, ZMQ_CAM_PORT, ZMQ_CMD_PORT, ZMQ_STATE_PORT


class DataCollector:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # === Directory setup ===
        self.base_dir = Path(cfg.storage_path) / cfg.task_name / f"demo_{cfg.demo_index}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Saving data to: {self.base_dir}")

        # === Camera setup ===
        camera_fps_value = self._cfg_get(cfg, "camera_fps", CAM_FPS)
        self.camera_fps = int(camera_fps_value) if camera_fps_value is not None else CAM_FPS
        self.camera_ids = self._parse_camera_ids(cfg)
        print(f"🎥 Expecting camera topics: {', '.join(self.camera_ids)}")

        # === Buffers ===
        self.state_data = []
        self.cmd_data = []
        self.image_buffers = {cam_id: [] for cam_id in self.camera_ids}
        self.video_writers = {}

        self.lock = threading.Lock()
        self.recording = False
        self.exit_flag = False

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

    def collect_state(self):
        print(f"📡 Subscribing to robot state at tcp://127.0.0.1:{ZMQ_STATE_PORT}")
        while not self.exit_flag:
            try:
                msg = self.zmq_state_sub.recv_json(flags=zmq.NOBLOCK)
                if self.recording:
                    with self.lock:
                        self.state_data.append({"timestamp": time.time(), "data": msg})
            except zmq.Again:
                time.sleep(0.005)

    def collect_commanded_state(self):
        print(f"📡 Subscribing to commanded state at tcp://127.0.0.1:{ZMQ_CMD_PORT}")
        while not self.exit_flag:
            try:
                msg = self.zmq_cmd_sub.recv_json(flags=zmq.NOBLOCK)
                if self.recording:
                    with self.lock:
                        self.cmd_data.append({"timestamp": time.time(), "data": msg})
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

                    writer = self.video_writers.get(cam_id)
                    if writer is None:
                        h, w, _ = img.shape
                        video_path = self.base_dir / f"{cam_id}_rgb_video.mp4"
                        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
                        writer = cv2.VideoWriter(str(video_path), fourcc, self.camera_fps, (w, h))
                        self.video_writers[cam_id] = writer

                    writer.write(img)

    def save_all(self):
        print("💾 Saving data...")

        with self.lock:
            with open(self.base_dir / "states.pkl", "wb") as f:
                pickle.dump(self.state_data, f)

            with open(self.base_dir / "commanded_states.pkl", "wb") as f:
                pickle.dump(self.cmd_data, f)

            for writer in self.video_writers.values():
                writer.release()

            saved_cameras = []
            for cam_idx, cam_id in enumerate(sorted(self.image_buffers.keys())):
                frames = self.image_buffers[cam_id]
                if not frames:
                    continue

                video_path = self.base_dir / f"{cam_id}_rgb_video.mp4"
                timestamps = [ts for ts, _ in frames]

                metadata = {
                    "cam_idx": cam_idx,
                    "cam_name": cam_id,
                    "cam_fps": self.camera_fps,
                    "num_image_frames": len(timestamps),
                    "timestamps": timestamps,
                    "record_start_time": timestamps[0],
                    "record_end_time": timestamps[-1],
                    "filename": str(video_path),
                }

                with open(self.base_dir / f"{cam_id}_rgb_video.metadata", "wb") as f:
                    pickle.dump(metadata, f)

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
        print("\n⏹️ Ctrl+C detected, stopping...")
        self.recording = False
        self.exit_flag = True

        self.state_thread.join(timeout=1.0)
        self.cmd_thread.join(timeout=1.0)
        self.image_thread.join(timeout=1.0)

        self.save_all()

    def run(self):
        print("Press ENTER to start collecting data...")
        input()
        self.recording = True
        print("▶️ Recording started. Press Ctrl+C to stop.")

        self.state_thread.start()
        self.cmd_thread.start()
        self.image_thread.start()

        while not self.exit_flag:
            time.sleep(0.1)
        print("🧹 Collector run() exiting")


def main(cfg: DictConfig):
    collector = DataCollector(cfg)
    collector.run()


if __name__ == "__main__":
    raise SystemExit("Use scripts/collect_data/run_data_collection.py to run this module.")
