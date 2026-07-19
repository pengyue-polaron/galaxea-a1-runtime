"""SO leader implementation of the tracked collection reset."""

from __future__ import annotations

import time

from galaxea_a1_runtime.apps.teleop.reset_config import HomePose, LeaderMotion
from galaxea_a1_runtime.apps.reset.progress import ResetProgress
from galaxea_a1_runtime.teleop.a1_so_leader import A1SOLeader, SOLeaderTeleopConfig


def reset_leader_home(home: HomePose, progress: ResetProgress) -> None:
    if not home.leader.enabled:
        progress.update("Leader", 100)
        return
    leader_home = home.leader
    motion = home.leader_motion
    leader_config = leader_home.config
    leader = A1SOLeader(
        SOLeaderTeleopConfig(
            id=leader_config.id,
            port=leader_config.port,
            use_degrees=leader_config.use_degrees,
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
        move_leader_smooth(leader, start, target, motion, progress)
        final = {
            key: float(value)
            for key, value in leader.get_action().items()
            if key in target
        }
        errors = mapping_errors(final, target)
        body_error = max(
            (error for key, error in errors.items() if key != "gripper.pos"),
            default=0.0,
        )
        gripper_error = errors.get("gripper.pos", 0.0)
        if body_error > motion.goal_tolerance_units:
            raise RuntimeError(
                f"Leader body reset error {body_error:.3f} exceeds tolerance "
                f"{motion.goal_tolerance_units:.3f}"
            )
        if gripper_error > motion.gripper_goal_tolerance_units:
            raise RuntimeError(
                f"Leader gripper reset error {gripper_error:.3f} exceeds tolerance "
                f"{motion.gripper_goal_tolerance_units:.3f}"
            )
        progress.update("Leader", 100)
    finally:
        try:
            leader.disable_torque()
        finally:
            leader.disconnect()


def move_leader_smooth(
    leader: A1SOLeader,
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
