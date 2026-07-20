"""Shared fail-closed EEF policy command execution.

The executor is ROS-free: bridge modules inject their relay monitor, command
publisher, shutdown predicate, and clocks. This keeps the activation sequence
identical across policy backends and directly unit-testable without hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from galaxea_a1_runtime.runtime.staged_motion import StagedMotionGate


class EefPolicyExecutor(StagedMotionGate):
    """Publish one validated EEF action through the locked command relay."""

    def __init__(
        self,
        *,
        relay: Any,
        commander: Any,
        staged_monitor: Any,
        relay_enable_timeout_s: float,
        staged_wait_timeout_s: float,
        staged_max_age_s: float,
        staged_alignment_tolerance_rad: float,
        is_shutdown: Callable[[], bool],
        policy_label: str,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            relay=relay,
            commander=commander,
            staged_monitor=staged_monitor,
            relay_enable_timeout_s=relay_enable_timeout_s,
            staged_wait_timeout_s=staged_wait_timeout_s,
            staged_max_age_s=staged_max_age_s,
            staged_alignment_tolerance_rad=staged_alignment_tolerance_rad,
            is_shutdown=is_shutdown,
            owner_label=policy_label,
            monotonic=monotonic,
            sleep=sleep,
        )
        self.policy_label = policy_label

    def publish(self, policy_action: np.ndarray) -> np.ndarray:
        """Publish one validated action after deterministic hold activation."""

        if not self.motion_enabled:
            raise RuntimeError(
                f"{self.policy_label} current-joint hold is not ACTIVE; "
                "refusing to publish a policy action"
            )
        self.enable_motion()

        target = np.asarray(policy_action, dtype=np.float64).reshape(8).copy()
        self.commander.publish_action(target, publish_gripper=False)
        self.commander.publish_gripper(float(target[7]))
        return target


def close_policy_resources(
    *,
    policy_label: str,
    executor: EefPolicyExecutor,
    timer: Any | None,
    cameras: Any | None,
    client: Any | None,
) -> None:
    """Close bridge-owned resources in the fail-closed shutdown order."""

    operations = [executor.disable_motion]
    if timer is not None:
        operations.append(timer.shutdown)
    if cameras is not None:
        operations.append(cameras.close)
    if client is not None:
        operations.append(client.close)

    errors: list[BaseException] = []
    for operation in operations:
        try:
            operation()
        except BaseException as exc:  # Cleanup must always continue.
            errors.append(exc)
    if errors:
        raise BaseExceptionGroup(f"{policy_label} bridge cleanup failed", errors)
