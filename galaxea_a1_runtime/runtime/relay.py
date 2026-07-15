"""Shared, ROS-message-agnostic relay status monitoring."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from galaxea_a1_runtime.hardware.freshness import LatestMessageCache


@dataclass(frozen=True)
class RelayStatus:
    state: str
    reason: str = ""
    valid: bool = True


RELAY_STATES = frozenset({"LOCKED", "ARMING", "ACTIVE", "FAULT"})


def decode_relay_status(data: str) -> RelayStatus:
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return RelayStatus(
            state="FAULT", reason=f"invalid relay status JSON: {data!r}", valid=False
        )
    if not isinstance(payload, dict):
        return RelayStatus(
            state="FAULT", reason="relay status payload must be an object", valid=False
        )
    state = payload.get("state")
    reason = payload.get("reason", "")
    if not isinstance(state, str) or state not in RELAY_STATES:
        return RelayStatus(
            state="FAULT", reason=f"invalid relay state: {state!r}", valid=False
        )
    if not isinstance(reason, str):
        return RelayStatus(
            state="FAULT", reason="relay status reason must be a string", valid=False
        )
    return RelayStatus(state=state, reason=reason)


def relay_status_is_fresh(
    updated_monotonic: float | None,
    *,
    max_age_s: float,
    now: float | None = None,
) -> bool:
    if updated_monotonic is None:
        return False
    current = time.monotonic() if now is None else now
    return current - updated_monotonic <= max_age_s


def relay_state_summary(
    status: RelayStatus | None,
    updated_monotonic: float | None,
    *,
    max_age_s: float,
    now: float | None = None,
) -> str:
    relay = status or RelayStatus(state="UNKNOWN")
    freshness = (
        "fresh"
        if relay_status_is_fresh(updated_monotonic, max_age_s=max_age_s, now=now)
        else "stale/no status"
    )
    return f"{relay.state}: {relay.reason} ({freshness})"


class RelayMonitor:
    """Decode relay callbacks and expose only fresh ACTIVE status."""

    def __init__(self, max_status_age_s: float):
        self.max_status_age_s = max_status_age_s
        self.cache: LatestMessageCache[RelayStatus] = LatestMessageCache()

    def callback(self, msg: Any) -> None:
        self.cache.set(decode_relay_status(str(msg.data)))

    def status(self) -> tuple[RelayStatus | None, float | None]:
        return self.cache.snapshot()

    def summary(self) -> str:
        status, updated = self.status()
        return relay_state_summary(status, updated, max_age_s=self.max_status_age_s)

    def is_active(self) -> bool:
        status, updated = self.status()
        return (
            relay_status_is_fresh(updated, max_age_s=self.max_status_age_s)
            and (status or RelayStatus("UNKNOWN")).state == "ACTIVE"
        )
