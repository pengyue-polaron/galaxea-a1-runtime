"""Content-verified local storage for immutable model artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from galaxea_a1_runtime.models.config import ModelArtifactConfig


@dataclass(frozen=True)
class ArtifactValidation:
    root: Path
    files: int
    bytes: int
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(
    config: ModelArtifactConfig,
    *,
    verify_hashes: bool,
    root: Path | None = None,
) -> ArtifactValidation:
    root = config.artifact_root if root is None else root
    if not root.is_dir():
        raise FileNotFoundError(f"model artifact is missing: {root}")
    expected_paths = {Path(*item.path.parts) for item in config.manifest.files}
    actual_paths = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(map(str, missing)))
        if unexpected:
            details.append("unexpected=" + ", ".join(map(str, unexpected)))
        raise ValueError(
            f"model artifact file set mismatch at {root}: {'; '.join(details)}"
        )
    total_bytes = 0
    for expected in config.manifest.files:
        path = root.joinpath(*expected.path.parts)
        size = path.stat().st_size
        if size != expected.size:
            raise ValueError(
                f"model artifact size mismatch for {expected.path}: "
                f"expected {expected.size}, got {size}"
            )
        total_bytes += size
        if verify_hashes:
            digest = sha256_file(path)
            if digest != expected.sha256:
                raise ValueError(
                    f"model artifact SHA256 mismatch for {expected.path}: "
                    f"expected {expected.sha256}, got {digest}"
                )
    return ArtifactValidation(
        root=root,
        files=len(config.manifest.files),
        bytes=total_bytes,
        manifest_sha256=config.manifest.sha256,
    )


def fetch_artifact(config: ModelArtifactConfig) -> ArtifactValidation:
    """Fetch into a hidden sibling and publish only after full validation."""

    root = config.artifact_root
    if root.exists():
        return validate_artifact(config, verify_hashes=True)
    staging = root.with_name(f".{root.name}.staging")
    if staging.exists():
        raise FileExistsError(
            f"model artifact staging path already exists; inspect it before retrying: {staging}"
        )
    if config.source.provider != "huggingface":
        raise ValueError(f"unsupported model provider: {config.source.provider}")
    from huggingface_hub import snapshot_download

    staging.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.source.repo_id,
        revision=config.source.revision,
        local_dir=staging,
        allow_patterns=[item.path.as_posix() for item in config.manifest.files],
        max_workers=4,
    )
    result = validate_artifact(config, verify_hashes=True, root=staging)
    staging.rename(root)
    return ArtifactValidation(
        root=root,
        files=result.files,
        bytes=result.bytes,
        manifest_sha256=result.manifest_sha256,
    )
