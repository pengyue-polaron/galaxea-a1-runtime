"""Discover and resolve strict tracked model registrations."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.models.config import ModelArtifactConfig, load_model_config


def registered_models(
    repo_root: Path,
    *,
    backend: str | None = None,
) -> tuple[ModelArtifactConfig, ...]:
    """Load every tracked model descriptor, optionally for one backend."""

    root = repo_root.resolve()
    paths = sorted((root / "configs/models").glob("**/*.toml"))
    descriptors = [path for path in paths if not path.name.endswith(".contract.toml")]
    models = tuple(load_model_config(path, repo_root=root) for path in descriptors)
    identities: dict[tuple[str, str], Path] = {}
    for model in models:
        identity = (model.model_id, model.source.revision)
        if previous := identities.get(identity):
            raise ValueError(
                "duplicate configured model identity "
                f"{model.model_id}@{model.source.revision}: {previous}, {model.path}"
            )
        identities[identity] = model.path
    if backend is None:
        return models
    return tuple(model for model in models if model.backend == backend)


def resolve_registered_model(
    selector: str,
    *,
    repo_root: Path,
    backend: str,
) -> ModelArtifactConfig:
    """Resolve an exact registered id or an unambiguous descriptor name."""

    value = selector.strip()
    if not value or value != selector:
        raise ValueError("model selector must be non-empty without surrounding space")
    models = registered_models(repo_root, backend=backend)
    matches = [model for model in models if value in _selectors(model)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(_pinned_id(model) for model in matches)
        raise ValueError(
            f"ambiguous registered model {value!r}; choose one of: {choices}"
        )
    available = ", ".join(_display_id(model) for model in models) or "none"
    raise ValueError(
        f"unknown registered {backend} model {value!r}; available: {available}"
    )


def _selectors(model: ModelArtifactConfig) -> frozenset[str]:
    return frozenset(
        {
            model.model_id,
            model.path.stem,
            f"{model.model_id}@{model.source.revision}",
            f"{model.model_id}@{model.source.revision_label}",
        }
    )


def _pinned_id(model: ModelArtifactConfig) -> str:
    return f"{model.model_id}@{model.source.revision}"


def _display_id(model: ModelArtifactConfig) -> str:
    return f"{model.path.stem} ({model.model_id}@{model.source.revision_label})"
