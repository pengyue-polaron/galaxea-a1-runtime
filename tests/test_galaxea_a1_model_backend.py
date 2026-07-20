import ast
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from galaxea_a1_runtime.models.backend import (
    parse_code_backend,
    verify_backend_checkout,
    verify_backend_environment,
)


REPO = Path(__file__).resolve().parents[1]


def test_runtime_source_remains_python311_parseable_for_openpi_backend():
    failures = []
    for path in sorted((REPO / "galaxea_a1_runtime").rglob("*.py")):
        try:
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(REPO)}:{exc.lineno}: {exc.msg}")

    assert not failures, "OpenPI Python 3.11 syntax failures:\n" + "\n".join(failures)


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _backend(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "config", "user.email", "test@example.com")
    (checkout / "source.py").write_text("VALUE = 1\n")
    _git(checkout, "add", "source.py")
    _git(checkout, "commit", "-q", "-m", "test")
    revision = _git(checkout, "rev-parse", "HEAD")
    repository = "https://example.com/owner/backend.git"
    _git(checkout, "remote", "add", "origin", repository)
    lock = tmp_path / "requirements.lock"
    lock.write_text("example==1\n")
    return parse_code_backend(
        backend={"schema_version": 1, "id": "test", "adapter": "test"},
        source={
            "repository": repository,
            "revision": revision,
            "checkout": str(checkout),
        },
        environment={
            "manager": "requirements-lock",
            "python_version": "3.12",
            "python": str(tmp_path / "environment/bin/python"),
            "lock": str(lock),
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        },
        repo_root=tmp_path,
    )


def test_backend_checkout_verifies_origin_revision_cleanliness_and_lock(tmp_path):
    config = _backend(tmp_path)

    verify_backend_checkout(config)

    _git(
        config.source.checkout,
        "remote",
        "set-url",
        "origin",
        "https://example.com/wrong.git",
    )
    with pytest.raises(ValueError, match="origin mismatch"):
        verify_backend_checkout(config)


def test_backend_checkout_rejects_lock_drift(tmp_path):
    config = _backend(tmp_path)
    config.environment.lock.write_text("example==2\n")

    with pytest.raises(ValueError, match="lock SHA256 mismatch"):
        verify_backend_checkout(config)


def test_backend_environment_verifies_the_configured_python_version(tmp_path):
    config = _backend(tmp_path)
    config.environment.python.parent.mkdir(parents=True)
    config.environment.python.symlink_to(sys.executable)

    verify_backend_environment(config)

    wrong_version = replace(
        config,
        environment=replace(config.environment, python_version="0.0"),
    )
    with pytest.raises(ValueError, match="Python version mismatch"):
        verify_backend_environment(wrong_version)
