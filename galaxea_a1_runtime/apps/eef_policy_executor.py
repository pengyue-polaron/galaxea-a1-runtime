"""Shared fail-closed EEF policy command execution.

The executor is ROS-free: bridge modules inject their relay monitor, command
publisher, shutdown predicate, and clocks. This keeps the activation sequence
identical across policy backends and directly unit-testable without hardware.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from galaxea_a1_runtime.console import success


class EefPolicyExecutor:
    """Publish one validated EEF action through the locked command relay."""

    def __init__(
        self,
        *,
        relay: Any,
        commander: Any,
        relay_enable_timeout_s: float,
        is_shutdown: Callable[[], bool],
        policy_label: str,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(relay_enable_timeout_s) or relay_enable_timeout_s <= 0:
            raise ValueError("EEF executor relay timeout must be finite and positive")
        self.relay = relay
        self.commander = commander
        self.relay_enable_timeout_s = relay_enable_timeout_s
        self.is_shutdown = is_shutdown
        self.policy_label = policy_label
        self.monotonic = monotonic
        self.sleep = sleep
        self.motion_enabled = False

    def publish_active_target(self, _event: object = None) -> None:
        """Refresh the staged target; safe for use as a ROS timer callback."""

        self.commander.publish_active_target()

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

    def activate_current_hold(self) -> None:
        """Stage current named joints and activate the relay on that hold."""

        if self.motion_enabled:
            self.enable_motion()
            return
        self.commander.hold_current_target()
        self.commander.publish_active_target()
        self.enable_motion()

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
