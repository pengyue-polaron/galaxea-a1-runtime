"""Read-only rollout telemetry journal; never creates a command publisher."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


def message_fields(value: Any) -> Any:
    """Retain ROS fields, including source header stamps and original ordering."""
    if hasattr(value, "__slots__"):
        return {name: message_fields(getattr(value, name)) for name in value.__slots__}
    if isinstance(value, (list, tuple)):
        return [message_fields(item) for item in value]
    return value


class MotionRecording:
    """Drain bounded callback events off the ROS/control threads without sampling."""

    def __init__(self, directory: Path, *, run_id: str, joint_names: tuple[str, ...]):
        directory.mkdir()
        self.directory = directory
        self.run_id = run_id
        self.joint_names = joint_names
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=8192)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._error: BaseException | None = None
        self._sequence = 0
        self._counts: Counter[str] = Counter()
        self._received: Counter[str] = Counter()
        self._source_sequences: dict[str, dict[str, int]] = {}
        self._subscribers: list[Any] = []
        self._stream = (directory / "events.jsonl").open("x", buffering=1)
        self._thread = threading.Thread(target=self._write, name="motion-recording")
        self._thread.start()
        self.record("recording_started", {"run_id": run_id, "joint_names": joint_names})

    def snapshot(self, source: Path, filename: str) -> None:
        data = source.read_bytes()
        (self.directory / filename).write_bytes(data)
        self.record(
            "source_snapshot",
            {
                "source": str(source),
                "file": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )

    def record(self, kind: str, data: dict) -> None:
        with self._lock:
            self.check()
            if self._closed:
                raise RuntimeError("Motion recording is already closed")
            event = {
                "schema_version": 1,
                "sequence": self._sequence,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
                "kind": kind,
                "data": data,
            }
            try:
                self._queue.put_nowait(event)
            except queue.Full as exc:
                self._error = RuntimeError(
                    "Motion recording queue overflow; recording incomplete"
                )
                raise self._error from exc
            self._sequence += 1
            self._received[kind] += 1

    def subscribe(self, rospy: Any, topic: str, message_type: Any, kind: str) -> None:
        def receive(message: Any) -> None:
            try:
                self.record(kind, {"topic": topic, "message": message_fields(message)})
            except Exception as exc:
                # rospy otherwise only logs callback exceptions. The rollout must
                # see a recorder failure at its next health check and stop.
                if not self._closed and self._error is None:
                    self._error = exc

        self._subscribers.append(rospy.Subscriber(topic, message_type, receive))

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("Motion recording failed") from self._error

    def require_streams(self, kinds: tuple[str, ...]) -> None:
        self.check()
        with self._lock:
            missing = [kind for kind in kinds if not self._received[kind]]
        if missing:
            raise RuntimeError(
                f"Motion recording has not received required streams: {missing}"
            )

    def _write(self) -> None:
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._stream.write(
                    json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n"
                )
                self._counts[event["kind"]] += 1
                header = event["data"].get("message", {}).get("header", {})
                source_sequence = header.get("seq")
                if isinstance(source_sequence, int):
                    stats = self._source_sequences.setdefault(
                        event["kind"],
                        {
                            "last": source_sequence - 1,
                            "discontinuities": 0,
                            "missing": 0,
                        },
                    )
                    delta = source_sequence - stats["last"]
                    if delta != 1:
                        stats["discontinuities"] += 1
                        stats["missing"] += max(0, delta - 1)
                    stats["last"] = source_sequence
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except Exception as exc:
            self._error = exc
        finally:
            self._stream.close()

    def close(self) -> None:
        if self._closed:
            return
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except Exception as exc:
                if self._error is None:
                    self._error = exc
        with self._lock:
            self._closed = True
        self._stop.set()
        self._thread.join()
        metadata = {
            "schema_version": 1,
            "run_id": self.run_id,
            "joint_names": self.joint_names,
            "complete": self._error is None
            and sum(self._counts.values()) == self._sequence,
            "error": None if self._error is None else str(self._error),
            "received_events": self._sequence,
            "written_counts": dict(self._counts),
            "source_sequence_stats": self._source_sequences,
            "completion_meaning": "All received events were written; inspect source_sequence_stats for upstream/ROS receive losses or publisher restarts",
            "scope": "bridge startup after batch reset through relay disable at rollout shutdown",
            "clock": "monotonic_ns / 1e9 shares the camera_timeline source monotonic clock; ROS source stamps remain in message headers",
            "command_order": "arm_control p_des/v_des follow joint_names; JointState retains its own name array",
        }
        (self.directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        self.check()
