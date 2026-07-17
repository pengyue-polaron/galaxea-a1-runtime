import io

from galaxea_a1_runtime.console import LiveStatusLine


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_live_status_replaces_one_tty_line_without_newlines():
    stream = FakeTty()
    status = LiveStatusLine(stream=stream)

    status.update("inference 1")
    status.update("execute 1/16")
    status.close()

    output = stream.getvalue()
    assert "\n" not in output
    assert output.count("\r") == 4
    assert "[RUN] inference 1" in output
    assert "[RUN] execute 1/16" in output


def test_live_status_throttles_redirected_output():
    stream = io.StringIO()
    now = [0.0]
    status = LiveStatusLine(stream=stream, monotonic=lambda: now[0])

    status.update("first")
    now[0] = 1.0
    status.update("hidden")
    now[0] = 5.0
    status.update("second")

    assert stream.getvalue().splitlines() == ["[RUN] first", "[RUN] second"]
