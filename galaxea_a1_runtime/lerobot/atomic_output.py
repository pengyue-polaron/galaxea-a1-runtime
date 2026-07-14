"""Fail-safe local output transactions for dataset conversion."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


@contextmanager
def atomic_output_directory(
    target: Path,
    *,
    overwrite: bool,
) -> Iterator[Path]:
    """Build beside ``target`` and install it only after the body succeeds.

    When overwriting, the previous target is renamed to a private backup during
    the final swap and restored if installing the staged directory fails.
    """

    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if _exists(target) and not overwrite:
        raise FileExistsError(f"target root exists: {target}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        yield staging
        _install_staged_directory(staging, target, overwrite=overwrite)
    except BaseException:
        if _exists(staging):
            _remove(staging)
        raise


def _install_staged_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
    if not _exists(target):
        os.replace(staging, target)
        return
    if not overwrite:
        raise FileExistsError(f"target root appeared during conversion: {target}")

    backup = target.parent / f".{target.name}.backup-{uuid4().hex}"
    os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        try:
            os.replace(backup, target)
        except BaseException as restore_error:
            raise RuntimeError(
                f"failed to install {target} and failed to restore backup {backup}"
            ) from restore_error
        raise
    _remove(backup)


@contextmanager
def atomic_output_file(target: Path) -> Iterator[Path]:
    """Write a sibling temporary file and atomically replace ``target``."""

    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.staging-",
        dir=target.parent,
    )
    os.close(descriptor)
    staging = Path(temporary_name)
    try:
        yield staging
        os.replace(staging, target)
    except BaseException:
        if _exists(staging):
            _remove(staging)
        raise


def atomic_write_text(target: Path, text: str, *, encoding: str) -> None:
    with atomic_output_file(target) as staging:
        staging.write_text(text, encoding=encoding)
