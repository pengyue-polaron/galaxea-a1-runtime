import hashlib
import json
from pathlib import Path

import pytest

from galaxea_a1_runtime.models.config import load_model_config
from galaxea_a1_runtime.models.store import fetch_artifact, validate_artifact


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_config(tmp_path: Path, files: dict[str, bytes], *, materialize: bool = True):
    revision = "1" * 40
    contract = tmp_path / "contract.toml"
    contract.write_text("[test]\nvalue = 1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "test/example",
                "source": {
                    "provider": "huggingface",
                    "repo_id": "owner/example",
                    "revision": revision,
                },
                "files": {
                    path: {"size": len(value), "sha256": _sha256(value)}
                    for path, value in files.items()
                },
            }
        )
    )
    descriptor = tmp_path / "model.toml"
    descriptor.write_text(
        "\n".join(
            (
                "[model]",
                "schema_version = 1",
                'id = "test/example"',
                'backend = "test"',
                'artifact_format = "test-format"',
                "checkpoint_step = 12",
                f'manifest = "{manifest}"',
                f'contract = "{contract}"',
                "",
                "[source]",
                'provider = "huggingface"',
                'repo_id = "owner/example"',
                f'revision = "{revision}"',
                'revision_label = "step-12"',
            )
        )
    )
    config = load_model_config(descriptor, repo_root=tmp_path)
    if materialize:
        for relative, value in files.items():
            path = config.artifact_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    return config


def test_model_artifact_path_is_derived_and_content_verified(tmp_path):
    config = _model_config(
        tmp_path,
        {"params/model.bin": b"immutable weights", "config.json": b"{}\n"},
    )

    result = validate_artifact(config, verify_hashes=True)

    assert config.artifact_root == (
        tmp_path / "models/artifacts/test/example" / ("1" * 40)
    )
    assert result.files == 2
    assert result.bytes == len(b"immutable weights") + len(b"{}\n")


def test_model_artifact_rejects_unexpected_or_drifted_files(tmp_path):
    config = _model_config(tmp_path, {"params/model.bin": b"immutable weights"})
    (config.artifact_root / "unexpected").write_text("drift")

    with pytest.raises(ValueError, match="file set mismatch"):
        validate_artifact(config, verify_hashes=True)

    (config.artifact_root / "unexpected").unlink()
    (config.artifact_root / "params/model.bin").write_bytes(b"x" * 17)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_artifact(config, verify_hashes=True)


def test_model_fetch_validates_hidden_staging_before_atomic_publication(
    tmp_path, monkeypatch
):
    files = {"params/model.bin": b"immutable weights", "config.json": b"{}\n"}
    config = _model_config(tmp_path, files, materialize=False)

    def snapshot_download(**kwargs):
        assert kwargs["revision"] == "1" * 40
        assert kwargs["max_workers"] == 4
        local_dir = Path(kwargs["local_dir"])
        assert local_dir.name == f".{config.source.revision}.staging"
        for relative, value in files.items():
            path = local_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    result = fetch_artifact(config)

    assert result.root == config.artifact_root
    assert result.root.is_dir()
    assert not result.root.with_name(f".{config.source.revision}.staging").exists()
