from galaxea_a1_runtime.apps import eef_policy_cli as cli
from galaxea_a1_runtime.policies.eef_actions import EefPolicyWorkspaceRejected
from galaxea_a1_runtime.hardware.eef_ik import A1EefIkTargetRejected


class _InterruptedBridge:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def run(self) -> None:
        self.events.append("run")
        raise KeyboardInterrupt

    def close(self) -> None:
        self.events.append("close")


class _RejectedBridge(_InterruptedBridge):
    def run(self) -> None:
        self.events.append("run")
        raise A1EefIkTargetRejected("solution delta 1.62 > 1.50 rad")


class _WorkspaceRejectedBridge(_InterruptedBridge):
    def run(self) -> None:
        self.events.append("run")
        raise EefPolicyWorkspaceRejected("target x=0.472 exceeds max=0.47")


def test_operator_interrupt_is_a_clean_exit_after_bridge_cleanup(monkeypatch):
    events: list[str] = []
    outcomes: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "info", lambda message: events.append(message))

    assert (
        cli.run_eef_policy_bridge(
            _InterruptedBridge(events),
            break_status_line=lambda: events.append("break-line"),
            record_outcome=lambda kind, message: outcomes.append((kind, message)),
        )
        == 0
    )

    assert events[:2] == ["run", "break-line"]
    assert "Ctrl+C received" in events[2]
    assert events[3] == "close"
    assert outcomes == [("operator_interrupted", events[2])]


def test_ik_target_rejection_safely_ends_attempt_without_traceback(monkeypatch):
    events: list[str] = []
    outcomes: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "warning", lambda message: events.append(message))

    assert (
        cli.run_eef_policy_bridge(
            _RejectedBridge(events),
            break_status_line=lambda: events.append("break-line"),
            record_outcome=lambda kind, message: outcomes.append((kind, message)),
        )
        == 0
    )

    assert events[:2] == ["run", "break-line"]
    assert "ended safely" in events[2]
    assert events[3] == "close"
    assert outcomes == [("ik_target_rejected", events[2])]


def test_workspace_rejection_safely_ends_attempt_without_traceback(monkeypatch):
    events: list[str] = []
    outcomes: list[tuple[str, str]] = []
    monkeypatch.setattr(cli, "warning", lambda message: events.append(message))

    assert (
        cli.run_eef_policy_bridge(
            _WorkspaceRejectedBridge(events),
            break_status_line=lambda: events.append("break-line"),
            record_outcome=lambda kind, message: outcomes.append((kind, message)),
        )
        == 0
    )

    assert events[:2] == ["run", "break-line"]
    assert "tracked workspace bounds" in events[2]
    assert "ended safely" in events[2]
    assert events[3] == "close"
    assert outcomes == [("workspace_target_rejected", events[2])]


def test_unexpected_bridge_error_still_fails_after_cleanup():
    events: list[str] = []
    bridge = _InterruptedBridge(events)
    bridge.run = lambda: (_ for _ in ()).throw(RuntimeError("serial disconnected"))

    try:
        cli.run_eef_policy_bridge(bridge)
    except RuntimeError as exc:
        assert str(exc) == "serial disconnected"
    else:
        raise AssertionError("unexpected runtime errors must not be hidden")

    assert events == ["close"]
