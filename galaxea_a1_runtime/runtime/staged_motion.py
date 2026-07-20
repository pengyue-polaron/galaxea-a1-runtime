"""App-agnostic activation of the locked staged-command relay."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from galaxea_a1_runtime.console import success
from galaxea_a1_runtime.runtime.ros_feedback import wait_for_staged_joint_alignment


class StagedMotionGate:
    """Activate motion only after a fresh current-joint hold is staged."""

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
        owner_label: str,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(relay_enable_timeout_s) or relay_enable_timeout_s <= 0:
            raise ValueError("staged motion relay timeout must be finite and positive")
        staged_settings = (
            staged_wait_timeout_s,
            staged_max_age_s,
            staged_alignment_tolerance_rad,
        )
        if not all(math.isfinite(value) for value in staged_settings):
            raise ValueError("staged-hold settings must be finite")
        if (
            staged_wait_timeout_s <= 0
            or staged_max_age_s <= 0
            or staged_alignment_tolerance_rad < 0
        ):
            raise ValueError(
                "staged-hold timeouts must be positive and alignment "
                "tolerance must be non-negative"
            )
        self.relay = relay
        self.commander = commander
        self.staged_monitor = staged_monitor
        self.relay_enable_timeout_s = relay_enable_timeout_s
        self.staged_wait_timeout_s = staged_wait_timeout_s
        self.staged_max_age_s = staged_max_age_s
        self.staged_alignment_tolerance_rad = staged_alignment_tolerance_rad
        self.is_shutdown = is_shutdown
        self.owner_label = owner_label
        self.monotonic = monotonic
        self.sleep = sleep
        self.motion_enabled = False
        self.motion_requested = False

    def publish_active_target(self, _event: object = None) -> None:
        """Refresh the staged target; safe for a ROS timer callback."""

        self.commander.publish_active_target()

    def activate_current_hold(self) -> None:
        """Stage current named joints and activate the relay on that hold."""

        if self.motion_enabled:
            self.enable_motion()
            return
        hold = self.commander.hold_current_target()
        self.commander.publish_active_target()
        wait_for_staged_joint_alignment(
            self.staged_monitor,
            hold,
            dof=len(hold),
            timeout_s=self.staged_wait_timeout_s,
            max_age_s=self.staged_max_age_s,
            tolerance_rad=self.staged_alignment_tolerance_rad,
            is_shutdown=self.is_shutdown,
            sleep=self.sleep,
            monotonic=self.monotonic,
        )
        self.enable_motion()

    def enable_motion(self) -> None:
        """Open the relay only after a fresh ACTIVE acknowledgement."""

        if self.motion_enabled:
            if self.relay.is_active():
                return
            self.motion_enabled = False
            self.commander.publish_motion_enable(False)
            self.motion_requested = False
            raise RuntimeError(
                "A1 relay is no longer confirmed ACTIVE; refusing to publish "
                f"{self.owner_label} commands. Last relay state: {self.relay.summary()}"
            )

        self.motion_requested = True
        self.commander.publish_motion_enable(True)
        deadline = self.monotonic() + self.relay_enable_timeout_s
        last_state = "no status"
        try:
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
        except BaseException:
            self.commander.publish_motion_enable(False)
            self.motion_requested = False
            raise
        self.commander.publish_motion_enable(False)
        self.motion_requested = False
        raise RuntimeError(f"A1 relay did not become ACTIVE: {last_state}")

    def disable_motion(self) -> None:
        """Lock the relay, including after partial initialization."""

        try:
            self.commander.publish_motion_enable(False)
        finally:
            self.motion_enabled = False
            self.motion_requested = False
