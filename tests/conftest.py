from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def short_socket_dir() -> Iterator[Path]:
    """Provide a real, private directory below the portable AF_UNIX limit."""

    path = Path(tempfile.mkdtemp(prefix="galaxea-a1-", dir="/tmp")).resolve()
    try:
        yield path
    finally:
        shutil.rmtree(path)
