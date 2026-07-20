"""Shared lifecycle handling for EEF policy bridge entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from galaxea_a1_runtime.console import info, warning


class EefPolicyBridge(Protocol):
    def run(self) -> None: ...

    def close(self) -> None: ...


def run_eef_policy_bridge(
    bridge: EefPolicyBridge,
    *,
    break_status_line: Callable[[], None] | None = None,
    record_outcome: Callable[[str, str], None] | None = None,
) -> int:
    """Run one bridge and cleanly end expected operator or safety stops."""

    outcome: tuple[str, str] | None = None
    try:
        bridge.run()
    except KeyboardInterrupt:
        if break_status_line is not None:
            break_status_line()
        message = "Ctrl+C received; locking the arm and closing policy resources."
        info(message)
        outcome = ("operator_interrupted", message)
    except Exception as exc:
        # Keep optional numeric/URDF dependencies out of static CLI startup.
        from galaxea_a1_runtime.apps.eef_policy_actions import (
            EefPolicyWorkspaceRejected,
        )
        from galaxea_a1_runtime.hardware.eef_ik import A1EefIkTargetRejected

        if isinstance(exc, A1EefIkTargetRejected):
            kind = "ik_target_rejected"
            boundary = "IK"
        elif isinstance(exc, EefPolicyWorkspaceRejected):
            kind = "workspace_target_rejected"
            boundary = "workspace"
        else:
            raise
        if break_status_line is not None:
            break_status_line()
        message = (
            f"Policy target was rejected by the tracked {boundary} bounds; this attempt "
            f"ended safely. {exc}"
        )
        warning(message)
        outcome = (kind, message)
    finally:
        bridge.close()
    if outcome is not None and record_outcome is not None:
        record_outcome(*outcome)
    return 0
