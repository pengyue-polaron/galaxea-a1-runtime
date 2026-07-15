#!/usr/bin/env python3
# ruff: noqa: E402
"""Orchestrate the tracked A1 and SO leader collection reset."""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from galaxea_a1_runtime.runtime.ros1_env import configure_ros1_python

configure_ros1_python(ROOT_DIR)

from galaxea_a1_runtime.apps.teleop.reset_a1 import A1HomeRunner
from galaxea_a1_runtime.apps.teleop.reset_config import load_home_pose
from galaxea_a1_runtime.apps.teleop.reset_leader import reset_leader_home
from galaxea_a1_runtime.apps.teleop.reset_progress import ResetProgress


DEFAULT_CONFIG = ROOT_DIR / "configs" / "poses" / "a1_so100_collection_start.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset A1 and leader to the tracked start pose."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    pose = load_home_pose(parse_args().config)
    devices = (
        ("A1", "Leader") if pose.leader is not None and pose.leader.enabled else ("A1",)
    )
    progress = ResetProgress(devices)
    a1 = A1HomeRunner(pose, progress)
    jobs = {"A1": a1.run}
    if pose.leader is not None and pose.leader.enabled:
        jobs["leader"] = lambda: reset_leader_home(pose, progress)

    errors: list[tuple[str, BaseException]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(job): name for name, job in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except BaseException as exc:
                errors.append((name, exc))
    if errors:
        progress.finish(success=False)
        details = "; ".join(f"{name}: {exc}" for name, exc in errors)
        raise RuntimeError(f"Reset failed ({details})") from errors[0][1]
    progress.finish(success=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
