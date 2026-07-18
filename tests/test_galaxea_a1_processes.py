from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROCESSES = REPO / "scripts/runtime/a1_processes.sh"


def _bash(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            f'export A1_PROCESS_STATE_ROOT="{tmp_path / "state"}"\n'
            f'source "{PROCESSES}"\n{body}',
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_marked_process_group_starts_reports_and_stops(tmp_path: Path):
    result = _bash(
        tmp_path,
        f"""
set -euo pipefail
a1_process_start test-server "{tmp_path}" "{tmp_path / "server.log"}" \\
  bash -c 'trap "exit 0" TERM; while :; do sleep 0.05; done'
a1_process_is_running test-server
a1_process_status test-server
a1_process_stop test-server 2
! a1_process_is_running test-server
[[ ! -e "$(a1_process_state_file test-server)" ]]
""",
    )

    assert result.returncode == 0, result.stderr
    assert "test-server: running" in result.stdout


def test_stop_refuses_pid_without_repository_marker(tmp_path: Path):
    result = _bash(
        tmp_path,
        """
set -u
mkdir -p "${A1_PROCESS_STATE_ROOT}"
sleep 30 &
unmarked_pid=$!
printf '%s\n' "${unmarked_pid}" >"$(a1_process_state_file unmarked)"
a1_process_stop unmarked 1
rc=$?
kill "${unmarked_pid}" 2>/dev/null || true
wait "${unmarked_pid}" 2>/dev/null || true
exit "${rc}"
""",
    )

    assert result.returncode == 2
    assert "Refusing to stop unmarked PID" in result.stderr


def test_emergency_cleanup_stops_every_marked_process(tmp_path: Path):
    result = _bash(
        tmp_path,
        f"""
set -euo pipefail
for name in first second; do
  a1_process_start "${{name}}" "{tmp_path}" "{tmp_path}/$name.log" \\
    bash -c 'trap "exit 0" TERM; while :; do sleep 0.05; done'
done
a1_process_stop_all_managed 2
! a1_process_is_running first
! a1_process_is_running second
""",
    )

    assert result.returncode == 0, result.stderr
