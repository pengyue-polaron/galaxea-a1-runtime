from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TMUX = REPO / "scripts/runtime/a1_tmux.sh"


def _bash(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{TMUX}"\n{body}'],
        text=True,
        capture_output=True,
        check=False,
    )


def test_tmux_http_health_wait_returns_on_first_success():
    result = _bash(
        """
curl() { return 0; }
a1_tmux_has_session() { return 1; }
a1_tmux_wait_for_http_health model http://127.0.0.1/healthz EXIT= model 2
"""
    )

    assert result.returncode == 0, result.stderr


def test_tmux_http_health_wait_fails_on_process_exit_marker():
    result = _bash(
        """
curl() { return 1; }
a1_tmux_has_session() { return 0; }
a1_tmux_capture() { echo SERVER_EXIT=1; }
a1_tmux_wait_for_http_health model http://127.0.0.1/healthz SERVER_EXIT= model 2
"""
    )

    assert result.returncode == 2
    assert "process exited during startup" in result.stderr


def test_tmux_http_health_wait_rejects_non_integer_timeout():
    result = _bash(
        "a1_tmux_wait_for_http_health model http://127.0.0.1/healthz EXIT= model 1.5"
    )

    assert result.returncode == 2
    assert "positive integer" in result.stderr
