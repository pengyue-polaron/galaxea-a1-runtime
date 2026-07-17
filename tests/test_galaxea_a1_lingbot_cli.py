from galaxea_a1_runtime.apps.lingbot import cli


class _LiveStatus:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def break_line(self) -> None:
        self.events.append("break-line")


class _InterruptedBridge:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.live_status = _LiveStatus(events)

    def run(self) -> None:
        self.events.append("run")
        raise KeyboardInterrupt

    def close(self) -> None:
        self.events.append("close")


def test_operator_interrupt_is_a_clean_exit_after_bridge_cleanup(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(cli, "info", lambda message: events.append(message))

    assert cli.run_bridge(_InterruptedBridge(events)) == 0

    assert events[:2] == ["run", "break-line"]
    assert "Ctrl+C received" in events[2]
    assert events[3] == "close"
