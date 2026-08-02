"""Compose the generic verified artifact store with the A1 model layout."""

from embodied_ops.artifact_store import (
    ArtifactValidation,
    fetch_huggingface_artifact,
    validate_artifact,
)

from galaxea_a1_runtime.models.config import ModelArtifactConfig


def fetch_artifact(config: ModelArtifactConfig) -> ArtifactValidation:
    return fetch_huggingface_artifact(
        config,
        cache_root=config.repo_root / "models" / "artifacts",
    )


__all__ = ["ArtifactValidation", "fetch_artifact", "validate_artifact"]
