"""Shared fail-closed EEF policy command execution.

The executor is ROS-free: bridge modules inject their state cache, relay monitor,
command publisher, shutdown predicate, and clocks. This keeps the safety sequence
identical across policy backends and directly unit-testable without hardware.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from galaxea_a1_runtime.console import info, success


class EefPolicyExecutor:
    """Publish one validated EEF action through the locked command relay."""

    def __init__(
        self,
        *,
        state: Any,
        relay: Any,
        commander: Any,
        relay_enable_timeout_s: float,
        settle_s: float,
        tolerance_m: float,
        corrections: int,
        is_shutdown: Callable[[], bool],
        policy_label: str,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        numeric = {
            "relay_enable_timeout_s": relay_enable_timeout_s,
            "settle_s": settle_s,
            "tolerance_m": tolerance_m,
        }
        invalid = [name for name, value in numeric.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"non-finite EEF executor settings: {invalid}")
        if relay_enable_timeout_s <= 0 or settle_s < 0 or tolerance_m <= 0:
            raise ValueError("invalid EEF executor timeout/tolerance settings")
        if (
            isinstance(corrections, bool)
            or not isinstance(corrections, int)
            or corrections < 0
        ):
            raise ValueError("EEF executor corrections must be a non-negative integer")
        self.state = state
        self.relay = relay
        self.commander = commander
        self.relay_enable_timeout_s = relay_enable_timeout_s
        self.settle_s = settle_s
        self.tolerance_m = tolerance_m
        self.corrections = corrections
        self.is_shutdown = is_shutdown
        self.policy_label = policy_label
        self.monotonic = monotonic
        self.sleep = sleep
        self.motion_enabled = False

    def publish_active_pose_target(self, _event: object = None) -> None:
        """Refresh the staged pose target; safe for use as a ROS timer callback."""

        self.commander.publish_active_pose_target()

    def publish(self, policy_action: np.ndarray) -> np.ndarray:
        """Stage, unlock, publish, and track one already-sanitized action."""

        if not self.state.pose_is_fresh() or not self.state.gripper_is_fresh():
            raise RuntimeError(
                f"{self.policy_label} EEF or gripper feedback is missing or stale; "
                "refusing to publish"
            )

        started = self.monotonic()
        last_command = self.state.tracker_command(policy_action)

        # Pose is staged while the relay remains locked. Gripper publication is
        # deliberately delayed until ACTIVE so no pre-gate target can pass through.
        self.commander.publish_action(last_command, publish_gripper=False)
        self.enable_motion()
        self.commander.publish_action(last_command, publish_gripper=True)

        error = self._wait_for_target_tracking(policy_action, started)
        for correction_index in range(self.corrections):
            if error <= self.tolerance_m:
                break
            command = self.state.tracker_command(policy_action)
            if np.allclose(command[:3], last_command[:3], atol=1e-4):
                break
            last_command = command
            info(
                f"{self.policy_label} tracking correction "
                f"{correction_index + 1}/{self.corrections}: "
                f"command_xyz={np.round(command[:3], 4).tolist()} "
                f"policy_xyz={np.round(policy_action[:3], 4).tolist()}"
            )
            self.commander.publish_action(last_command, publish_gripper=False)
            error = self._wait_for_target_tracking(policy_action, started)
        return last_command

    def enable_motion(self) -> None:
        """Open the relay only after a fresh ACTIVE acknowledgement."""

        if self.motion_enabled:
            if self.relay.is_active():
                return
            self.motion_enabled = False
            self.commander.publish_motion_enable(False)
            raise RuntimeError(
                "A1 relay is no longer confirmed ACTIVE; refusing to publish "
                f"{self.policy_label} commands. Last relay state: "
                f"{self.relay.summary()}"
            )

        self.commander.publish_motion_enable(True)
        deadline = self.monotonic() + self.relay_enable_timeout_s
        last_state = "no status"
        while not self.is_shutdown() and self.monotonic() < deadline:
            status, _ = self.relay.status()
            last_state = self.relay.summary()
            if self.relay.is_active():
                self.motion_enabled = True
                success("Real arm command relay is ACTIVE.")
                return
            if status is not None and status.state == "FAULT":
                break
            self.sleep(0.05)
        self.commander.publish_motion_enable(False)
        raise RuntimeError(f"A1 relay did not become ACTIVE: {last_state}")

    def disable_motion(self) -> None:
        """Lock the relay, including after partial initialization."""

        try:
            self.commander.publish_motion_enable(False)
        finally:
            self.motion_enabled = False

    def _wait_for_target_tracking(
        self, policy_action: np.ndarray, started: float
    ) -> float:
        if self.settle_s <= 0:
            return float("nan")
        deadline = self.monotonic() + self.settle_s
        error = float("inf")
        while not self.is_shutdown() and self.monotonic() < deadline:
            current = self.state.current_xyz()
            if current is not None:
                error = float(
                    np.linalg.norm(
                        np.asarray(policy_action[:3], dtype=np.float64) - current
                    )
                )
                if error <= self.tolerance_m:
                    break
            self.sleep(0.03)
        current = self.state.current_xyz()
        if current is not None:
            error = float(
                np.linalg.norm(
                    np.asarray(policy_action[:3], dtype=np.float64) - current
                )
            )
            info(
                f"{self.policy_label} tracking: "
                f"waited={self.monotonic() - started:.2f}s "
                f"actual_xyz={np.round(current, 4).tolist()} "
                f"target_err_cm={error * 100.0:.2f}"
            )
        return error


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
