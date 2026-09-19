"""Read-only rosbag2 acquisition with durable operator episode boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from embodied_ops.artifacts import atomic_write_text
from galaxea_a1_runtime.apps.cameras.ros2_tools import recording_topics
from galaxea_a1_runtime.apps.teleop.bag_contract import (
    CAPTURE_CONTRACT,
    MANIFEST_NAME,
    config_fingerprint,
    source_topics,
)
from galaxea_a1_runtime.observability import legacy_observation_topics
from galaxea_a1_runtime.runtime.ros2 import ROS2_IMAGE

FINALIZE_TIMEOUT_S = 30
STORAGE_CONFIG_NAME = "storage-config.yaml"
# Lossless MCAP chunk compression; measured ~2x on recorded raw camera streams.
STORAGE_CONFIG = "compression: Zstd\ncompressionLevel: Fastest\n"
PREFLIGHT_ATTEMPTS = 3
PREFLIGHT_TIMEOUT_S = 6.0
PREFLIGHT_RETRY_S = 3.0


def require_source_topics_delivered(system) -> None:
    """Fail before collection when ROS 2 robot telemetry is not being delivered."""

    repo = Path(__file__).resolve().parents[3]
    topics = source_topics(system)
    details = "probe did not run"
    for attempt in range(PREFLIGHT_ATTEMPTS):
        with tempfile.TemporaryDirectory(prefix="a1-telemetry-preflight-") as scratch:
            manifest = Path(scratch) / "preflight.json"
            manifest.write_text(json.dumps({"topics": topics}))
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "host",
                    "--ipc",
                    "host",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "-v",
                    f"{scratch}:/check",
                    "-v",
                    f"{repo}:/workspace:ro",
                    "-e",
                    f"ROS_DOMAIN_ID={system.ros2.domain_id}",
                    "-e",
                    "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
                    "-e",
                    "ROS_LOG_DIR=/tmp/ros-log",
                    ROS2_IMAGE,
                    "python3",
                    "-m",
                    "galaxea_a1_runtime.apps.teleop.bag_probe",
                    "/check/preflight.json",
                    str(PREFLIGHT_TIMEOUT_S),
                ],
                capture_output=True,
                text=True,
                timeout=PREFLIGHT_TIMEOUT_S + 30,
            )
        if result.returncode == 0:
            return
        output = (result.stderr or result.stdout).strip().splitlines()
        details = output[-1] if output else f"probe exited {result.returncode}"
        if attempt + 1 < PREFLIGHT_ATTEMPTS:
            time.sleep(PREFLIGHT_RETRY_S)
    raise RuntimeError(
        "robot telemetry topics are not delivered to ROS 2; restart the observation "
        f"stack (just cameras stop, then just cameras start) and retry ({details})"
    )


class BagCapture:
    """Own one recorder; all outcomes retain the raw files and manifest."""

    def __init__(self, config, *, experiment: str, task: str):
        self.config = config
        self.run_id = uuid.uuid4().hex
        self.root = (
            config.collection.dataset_root.parent
            / "recordings"
            / experiment
            / self.run_id
        )
        self.container = f"a1-collection-bag-{self.run_id}"
        self.process = None
        self.log = None
        self.manifest = {
            "capture_contract": CAPTURE_CONTRACT,
            "bag_id": self.run_id,
            "experiment": experiment,
            "task": task,
            "config_sha256": config_fingerprint(config),
            "config_snapshots": {
                "teleop": config.path.read_text(),
                "system": config.system.path.read_text(),
            },
            "topics": source_topics(config.system),
            "clock": "ros_system_time",
            "status": "preparing",
            "disposition": "interrupted",
        }

    def _write_manifest(self):
        atomic_write_text(
            self.root / MANIFEST_NAME,
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
        )

    def start(self):
        self.root.mkdir(parents=True, exist_ok=False)
        atomic_write_text(self.root / STORAGE_CONFIG_NAME, STORAGE_CONFIG)
        self._write_manifest()
        repo = Path(__file__).resolve().parents[3]
        required = source_topics(self.config.system)
        topics = sorted(
            set(required.values())
            | set(recording_topics(self.config.system))
            | {t for _, t, _ in legacy_observation_topics(self.config.system)}
        )
        self.log = (self.root / "recorder.log").open("w")
        self.process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                self.container,
                "--label",
                "io.galaxea.a1-runtime.managed=true",
                "--network",
                "host",
                "--ipc",
                "host",
                "--init",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{self.root}:/recordings",
                "-v",
                f"{repo}:/workspace:ro",
                "-e",
                f"ROS_DOMAIN_ID={self.config.system.ros2.domain_id}",
                "-e",
                "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
                "-e",
                "ROS_LOG_DIR=/tmp/ros-log",
                ROS2_IMAGE,
                "ros2",
                "bag",
                "record",
                "--storage",
                "mcap",
                "--storage-config-file",
                f"/recordings/{STORAGE_CONFIG_NAME}",
                "--output",
                "/recordings/bag",
                "--disable-keyboard-controls",
                "--topics",
                *topics,
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.config.collection.ready_timeout_s
        while time.monotonic() < deadline:
            self.check()
            log = (self.root / "recorder.log").read_text()
            if all(
                f"Subscribed to topic '{topic}'" in log for topic in required.values()
            ):
                try:
                    subprocess.run(
                        [
                            "docker",
                            "exec",
                            self.container,
                            "/ros_entrypoint.sh",
                            "python3",
                            "-m",
                            "galaxea_a1_runtime.apps.teleop.bag_probe",
                            "/recordings/episode.json",
                            str(self.config.collection.ready_timeout_s),
                        ],
                        check=True,
                        stdout=self.log,
                        stderr=subprocess.STDOUT,
                        timeout=self.config.collection.ready_timeout_s + 5,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        "robot telemetry topics are not being delivered to the recorder; "
                        "restart the observation stack (just cameras stop, then "
                        "just cameras start) and retry"
                    ) from exc
                time.sleep(
                    max(
                        self.config.system.cameras.max_age_s,
                        self.config.system.joint_safety.max_feedback_age_s,
                    )
                )
                self.check()
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"rosbag2 required subscriptions not ready; see {self.root / 'recorder.log'}"
        )

    def begin(self):
        self.check()
        self.manifest.update(status="recording", start_ns=time.time_ns())
        self._write_manifest()

    def check(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(
                f"rosbag2 recorder is not running; raw data retained at {self.root}"
            )

    def stop(self, disposition="interrupted"):
        if "end_ns" in self.manifest:
            return
        self.manifest.update(end_ns=time.time_ns(), disposition=str(disposition))
        finalized = False
        try:
            if self.process is not None:
                # Preserve a bounded tail for interpolation beyond the last grid point.
                if disposition == "save" and self.process.poll() is None:
                    time.sleep(
                        max(
                            self.config.system.joint_safety.max_feedback_age_s,
                            self.config.system.eef.max_feedback_age_s,
                        )
                    )
                subprocess.run(
                    ["docker", "kill", "--signal", "SIGINT", self.container],
                    capture_output=True,
                )
                try:
                    self.process.wait(timeout=FINALIZE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    subprocess.run(
                        ["docker", "stop", "--time", "5", self.container],
                        capture_output=True,
                    )
                    self.process.wait(timeout=10)
                finalized = (
                    self.process.returncode == 0
                    and (self.root / "bag/metadata.yaml").is_file()
                )
        finally:
            if self.log is not None:
                self.log.close()
            self.manifest["status"] = "finalized" if finalized else "incomplete"
            self._write_manifest()
        if not finalized:
            raise RuntimeError(
                f"rosbag2 did not finalize; raw data retained at {self.root}"
            )
