"""SO leader implementation of the tracked collection reset."""

from __future__ import annotations

import time

from galaxea_a1_runtime.apps.teleop.reset_config import HomePose, LeaderMotion
from galaxea_a1_runtime.apps.reset.progress import ResetProgress
from galaxea_a1_runtime.console import warning
from lerobot_teleoperator_galaxea_a1_so_leader import (
    GalaxeaA1SOLeader,
    GalaxeaA1SOLeaderConfig,
)


def reset_leader_home(home: HomePose, progress: ResetProgress) -> None:
    if not home.leader.enabled:
        progress.update("Leader", 100)
        return
    leader_home = home.leader
    motion = home.leader_motion
    leader_config = leader_home.config
    leader = GalaxeaA1SOLeader(
        GalaxeaA1SOLeaderConfig(
            id=leader_config.id,
            port=leader_config.port,
            motor_write_retries=leader_config.motor_write_retries,
        )
    )
    leader.connect(calibrate=False)
    try:
        current = {key: float(value) for key, value in leader.get_action().items()}
        target = leader_home.action
        missing = sorted(key for key in target if key not in current)
        if missing:
            raise RuntimeError(f"leader action missing keys: {missing}")
        start = {key: current[key] for key in target}
        progress.update("Leader", 0)
        leader.enable_torque()
        move_leader_to_goal(leader, start, target, motion, progress)
        progress.update("Leader", 100)
    finally:
        try:
            leader.disable_torque()
        finally:
            leader.disconnect()


def move_leader_to_goal(
    leader: GalaxeaA1SOLeader,
    start: dict[str, float],
    target: dict[str, float],
    motion: LeaderMotion,
    progress: ResetProgress,
) -> None:
    current = start
    failures: dict[str, tuple[float, float]] = {}
    for attempt in range(1, motion.max_goal_attempts + 1):
        move_leader_smooth(leader, current, target, motion, progress)
        final = {
            key: float(value)
            for key, value in leader.get_action().items()
            if key in target
        }
        errors = mapping_errors(final, target)
        failures = goal_failures(errors, motion)
        if not failures:
            return
        if attempt < motion.max_goal_attempts:
            warning(
                f"Leader reset attempt {attempt}/{motion.max_goal_attempts} "
                f"needs correction ({format_goal_failures(failures)})"
            )
            current = final

    raise RuntimeError(
        f"Leader reset did not converge after {motion.max_goal_attempts} attempts "
        f"({format_goal_failures(failures)})"
    )


def goal_failures(
    errors: dict[str, float], motion: LeaderMotion
) -> dict[str, tuple[float, float]]:
    failures = {}
    for key, error in errors.items():
        tolerance = (
            motion.gripper_goal_tolerance_units
            if key == "gripper.pos"
            else motion.goal_tolerance_units
        )
        if error > tolerance:
            failures[key] = (error, tolerance)
    return failures


def format_goal_failures(failures: dict[str, tuple[float, float]]) -> str:
    return ", ".join(
        f"{key} error {error:.3f} > {tolerance:.3f}"
        for key, (error, tolerance) in sorted(failures.items())
    )


def move_leader_smooth(
    leader: GalaxeaA1SOLeader,
    start: dict[str, float],
    target: dict[str, float],
    motion: LeaderMotion,
    progress: ResetProgress,
) -> None:
    max_delta = max(abs(target[key] - start[key]) for key in target)
    duration_s = max(motion.min_duration_s, max_delta / motion.max_velocity_units_s)
    steps = max(1, int(duration_s * motion.hz))
    for step in range(steps + 1):
        alpha = step / steps
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        command = {
            key: start[key] + (target[key] - start[key]) * smooth for key in target
        }
        leader.send_feedback(command)
        progress.update("Leader", alpha * 100.0)
        time.sleep(1.0 / motion.hz)
    deadline = time.monotonic() + motion.hold_s
    while time.monotonic() < deadline:
        leader.send_feedback(target)
        time.sleep(1.0 / motion.hz)


def mapping_errors(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    missing = sorted(key for key in right if key not in left)
    if missing:
        raise ValueError(f"missing keys: {missing}")
    return {key: abs(left[key] - right[key]) for key in right}
