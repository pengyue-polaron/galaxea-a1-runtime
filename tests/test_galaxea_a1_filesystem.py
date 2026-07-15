from pathlib import Path

import pytest

from galaxea_a1_runtime.filesystem import (
    OutputDirectoryTransaction,
    atomic_output_directory,
    atomic_output_file,
)


def _temporary_outputs(parent: Path) -> list[Path]:
    return sorted(path for path in parent.iterdir() if ".staging-" in path.name)


def test_atomic_directory_failure_preserves_existing_output(tmp_path):
    target = tmp_path / "dataset"
    target.mkdir()
    (target / "old.txt").write_text("old")

    with pytest.raises(RuntimeError, match="conversion failed"):
        with atomic_output_directory(target, overwrite=True) as staging:
            (staging / "new.txt").write_text("new")
            raise RuntimeError("conversion failed")

    assert (target / "old.txt").read_text() == "old"
    assert not (target / "new.txt").exists()
    assert _temporary_outputs(tmp_path) == []


def test_atomic_directory_success_replaces_existing_output(tmp_path):
    target = tmp_path / "dataset"
    target.mkdir()
    (target / "old.txt").write_text("old")

    with atomic_output_directory(target, overwrite=True) as staging:
        (staging / "new.txt").write_text("new")

    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text() == "new"
    assert _temporary_outputs(tmp_path) == []


def test_atomic_directory_rejects_existing_output_without_overwrite(tmp_path):
    target = tmp_path / "dataset"
    target.mkdir()

    with pytest.raises(FileExistsError, match="target root exists"):
        with atomic_output_directory(target, overwrite=False):
            raise AssertionError("must not enter")


def test_explicit_directory_transaction_only_installs_after_commit(tmp_path):
    target = tmp_path / "episode_000"

    with OutputDirectoryTransaction(target) as transaction:
        assert transaction.path is not None
        (transaction.path / "frames.csv").write_text("frame_index\n")
        assert not target.exists()
        transaction.commit()

    assert (target / "frames.csv").is_file()


def test_uncommitted_directory_transaction_removes_staging(tmp_path):
    target = tmp_path / "episode_000"

    with OutputDirectoryTransaction(target) as transaction:
        assert transaction.path is not None
        (transaction.path / "partial.jpg").write_bytes(b"partial")

    assert not target.exists()
    assert _temporary_outputs(tmp_path) == []


def test_atomic_file_failure_preserves_existing_output(tmp_path):
    target = tmp_path / "dataset.tar.gz"
    target.write_bytes(b"old")

    with pytest.raises(RuntimeError, match="archive failed"):
        with atomic_output_file(target) as staging:
            staging.write_bytes(b"new")
            raise RuntimeError("archive failed")

    assert target.read_bytes() == b"old"
    assert _temporary_outputs(tmp_path) == []
