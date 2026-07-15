"""Fail-safe local file and directory output transactions."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


class OutputDirectoryTransaction:
    """Explicitly commit a staged directory or remove it on scope exit."""

    def __init__(
        self,
        target: Path,
        *,
        overwrite: bool = False,
        precreate_staging: bool = True,
    ):
        self.target = target.expanduser().resolve()
        self.overwrite = overwrite
        self.precreate_staging = precreate_staging
        self.path: Path | None = None
        self._committed = False

    def __enter__(self) -> "OutputDirectoryTransaction":
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if _exists(self.target) and not self.overwrite:
            raise FileExistsError(f"target root exists: {self.target}")
        self.path = Path(
            tempfile.mkdtemp(
                prefix=f".{self.target.name}.staging-", dir=self.target.parent
            )
        )
        if not self.precreate_staging:
            self.path.rmdir()
        return self

    def commit(self) -> Path:
        if self.path is None:
            raise RuntimeError("output transaction has not started")
        if self._committed:
            raise RuntimeError("output transaction was already committed")
        if not _exists(self.path):
            raise RuntimeError(f"staged directory was not created: {self.path}")
        _install_staged_directory(self.path, self.target, overwrite=self.overwrite)
        self._committed = True
        return self.target

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if not self._committed and self.path is not None and _exists(self.path):
            _remove(self.path)


@contextmanager
def atomic_output_directory(
    target: Path, *, overwrite: bool, precreate_staging: bool = True
) -> Iterator[Path]:
    """Build beside ``target`` and install it only after the body succeeds."""

    with OutputDirectoryTransaction(
        target,
        overwrite=overwrite,
        precreate_staging=precreate_staging,
    ) as transaction:
        assert transaction.path is not None
        yield transaction.path
        transaction.commit()


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


def atomic_write_text(target: Path, text: str, *, encoding: str = "utf-8") -> None:
    with atomic_output_file(target) as staging:
        staging.write_text(text, encoding=encoding)
