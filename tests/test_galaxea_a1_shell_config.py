import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOADER = REPO / "scripts/runtime/a1_config.sh"


def run_bash(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'set -e; source "{LOADER}"; {body}'],
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_config_loader_propagates_renderer_failure():
    result = run_bash("a1_load_shell_config bash -c 'exit 17'")

    assert result.returncode == 17
    assert "Configuration renderer failed" in result.stderr


def test_shell_config_loader_rejects_empty_output():
    result = run_bash("a1_load_shell_config true")

    assert result.returncode == 2


def test_shell_config_loader_applies_assignments_in_calling_shell():
    result = run_bash(
        "a1_load_shell_config printf 'A1_TEST_VALUE=%s\\n' ready; "
        'test "${A1_TEST_VALUE}" = ready'
    )

    assert result.returncode == 0
