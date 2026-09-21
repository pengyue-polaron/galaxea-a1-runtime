"""Register every model of one tracked release plan from verified content.

The registrar pins one Hugging Face revision, proves every local or downloaded
file against that revision's tree metadata, and only then publishes an
immutable artifact plus its descriptor, manifest, contract, and deployment.
Configuration writes are create-only: an edited file is a conflict, never a
silent overwrite.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from embodied_ops import file_sha256, write_json_once, write_text_once

from galaxea_a1_runtime.apps.diffusion2one.contract import (
    validate_checkpoint_metadata,
)
from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.console import ArgumentParser, emit, failure, info, success
from galaxea_a1_runtime.models.config import load_model_config
from galaxea_a1_runtime.models.release import (
    RELEASE_MODEL_FILES,
    ModelRelease,
    ReleaseModel,
    load_model_release,
)
from galaxea_a1_runtime.models.store import validate_artifact

RECEIPT_ROOT = Path("outputs/model_registration")
SEED_ROOT = Path("models/imports")


@dataclass(frozen=True)
class HubTreeEntry:
    path: str
    size: int
    sha256: str | None
    git_sha1: str | None


def register_release(
    plan: ModelRelease,
    *,
    local_root: Path | None = None,
    only: tuple[str, ...] = (),
) -> tuple[int, dict[str, Any]]:
    """Register every selected model; one model failure does not hide the rest."""

    if local_root is not None:
        local_root = local_root.expanduser().resolve()
        if not local_root.is_dir():
            raise FileNotFoundError(f"local release root is missing: {local_root}")
    known = {model.key for model in plan.models}
    if only:
        unknown = sorted(set(only) - known)
        if unknown:
            raise ValueError(
                "unknown --only keys: "
                + ", ".join(unknown)
                + "; available: "
                + ", ".join(sorted(known))
            )
    selected = set(only)
    info(
        f"Registering release {plan.release_id} from "
        f"{plan.source.repo_id}@{plan.source.revision} "
        f"({len(selected) if selected else len(plan.models)} model(s))"
    )
    tree = _hub_tree(plan)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    receipt_root = plan.repo_root / RECEIPT_ROOT / f"{plan.release_id}_{stamp}"
    receipt_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    failures = 0
    for index, model in enumerate(plan.models):
        if selected and model.key not in selected:
            continue
        try:
            entry = _register_model(plan, model, index, tree, local_root=local_root)
        except (OSError, RuntimeError, ValueError) as exc:
            failures += 1
            emit("FAIL", f"{model.key:<40} {exc}", stream=sys.stderr)
            entries.append(
                {"key": model.key, "model_id": model.model_id, "error": str(exc)}
            )
        else:
            entries.append(entry)
            emit(
                "PASS",
                f"{model.key:<40} {entry['artifact_state']:<10} "
                f"{entry['files']} files {entry['bytes']} bytes",
            )

    receipt = {
        "release_id": plan.release_id,
        "plan": str(plan.path),
        "source": {
            "provider": plan.source.provider,
            "repo_id": plan.source.repo_id,
            "revision": plan.source.revision,
            "revision_label": plan.source.revision_label,
        },
        "local_root": str(local_root) if local_root is not None else "",
        "models": entries,
        "failures": failures,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_once(receipt_root / "registration.json", receipt)
    write_json_once(
        receipt_root / "tree.json",
        {
            "repo_id": plan.source.repo_id,
            "revision": plan.source.revision,
            "files": {
                path: {
                    "size": item.size,
                    "sha256": item.sha256,
                    "git_sha1": item.git_sha1,
                }
                for path, item in sorted(tree.items())
            },
        },
    )
    if failures:
        failure(f"Release registration failed for {failures} model(s).")
        return 1, receipt
    success(f"Release {plan.release_id} is registered; receipt: {receipt_root}")
    return 0, receipt


def _register_model(
    plan: ModelRelease,
    model: ReleaseModel,
    index: int,
    tree: dict[str, HubTreeEntry],
    *,
    local_root: Path | None,
) -> dict[str, Any]:
    expected = _expected_entries(tree, model)
    artifact_root = _artifact_root(plan, model)
    content_dir, artifact_state = _content_directory(
        plan, model, artifact_root, local_root
    )

    staging: Path | None = None
    try:
        if artifact_root.is_dir():
            files = _existing_files(artifact_root, expected)
        else:
            staging, files = _stage_artifact(
                artifact_root, content_dir, model.directory, expected
            )

        manifest_value = {
            "schema_version": 1,
            "model_id": model.model_id,
            "source": {
                "provider": plan.source.provider,
                "repo_id": plan.source.repo_id,
                "revision": plan.source.revision,
            },
            "files": files,
        }
        _write_create_only(plan.manifest_path(model), _json_text(manifest_value))
        _write_create_only(
            plan.contract_path(model), _contract_text(plan, model, content_dir)
        )
        _write_create_only(plan.descriptor_path(model), _descriptor_text(plan, model))
        _write_create_only(
            plan.deployment_path(model), _deployment_text(plan, model, index)
        )
        deployment = load_lingbot_config(
            plan.deployment_path(model), repo_root=plan.repo_root
        )
        _check_deployment(plan, model, index, deployment)

        if staging is not None:
            staging.rename(artifact_root)

        config = load_model_config(
            plan.descriptor_path(model), repo_root=plan.repo_root
        )
        validation = validate_artifact(config, verify_hashes=False)
        validate_checkpoint_metadata(deployment, config.artifact_root)
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "key": model.key,
        "model_id": model.model_id,
        "kind": model.kind.name,
        "directory": model.directory.as_posix(),
        "checkpoint_step": model.checkpoint_step,
        "catalog": model.catalog.as_posix(),
        "catalog_id": model.catalog_record.catalog_id,
        "descriptor": plan.descriptor_path(model)
        .relative_to(plan.repo_root)
        .as_posix(),
        "manifest": plan.manifest_path(model).relative_to(plan.repo_root).as_posix(),
        "contract": plan.contract_path(model).relative_to(plan.repo_root).as_posix(),
        "deployment": plan.deployment_path(model)
        .relative_to(plan.repo_root)
        .as_posix(),
        "deployment_id": plan.deployment_id(model),
        "ports": {"server": plan.server_port(index), "master": plan.master_port(index)},
        "artifact_root": str(config.artifact_root),
        "artifact_state": artifact_state,
        "files": validation.files,
        "bytes": validation.bytes,
        "manifest_sha256": validation.manifest_sha256,
        "verified_against": f"{plan.source.repo_id}@{plan.source.revision}",
    }


def _expected_entries(
    tree: dict[str, HubTreeEntry], model: ReleaseModel
) -> dict[str, HubTreeEntry]:
    prefix = model.directory.as_posix() + "/"
    expected: dict[str, HubTreeEntry] = {}
    for name in RELEASE_MODEL_FILES:
        path = prefix + name
        entry = tree.get(path)
        if entry is None:
            raise FileNotFoundError(f"release revision is missing {path}")
        expected[path] = entry
    extra = sorted(
        path for path in tree if path.startswith(prefix) and path not in expected
    )
    if extra:
        raise ValueError("release folder carries unreviewed files: " + ", ".join(extra))
    return expected


def _artifact_root(plan: ModelRelease, model: ReleaseModel) -> Path:
    model_parts = Path(*model.model_id.split("/"))
    return plan.repo_root / "models/artifacts" / model_parts / plan.source.revision


def _content_directory(
    plan: ModelRelease,
    model: ReleaseModel,
    artifact_root: Path,
    local_root: Path | None,
) -> tuple[Path, str]:
    if artifact_root.is_dir():
        return artifact_root / Path(*model.directory.parts), "existing"
    if local_root is not None:
        candidate = local_root / Path(*model.directory.parts)
        if not candidate.is_dir():
            raise FileNotFoundError(f"local release folder is missing: {candidate}")
        return candidate, "imported"
    seed = plan.repo_root / SEED_ROOT / f".{plan.release_id}-{model.key}.seed"
    _download_release_model(plan, model, seed)
    return seed / Path(*model.directory.parts), "downloaded"


def _download_release_model(
    plan: ModelRelease, model: ReleaseModel, seed_root: Path
) -> None:
    from huggingface_hub import snapshot_download

    prefix = model.directory.as_posix()
    info(f"{model.key}: downloading {prefix} from the pinned revision")
    snapshot_download(
        repo_id=plan.source.repo_id,
        revision=plan.source.revision,
        local_dir=seed_root,
        allow_patterns=[prefix + "/" + name for name in RELEASE_MODEL_FILES],
        max_workers=4,
    )


def _existing_files(
    artifact_root: Path, expected: dict[str, HubTreeEntry]
) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for relative, entry in expected.items():
        path = artifact_root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"artifact content is missing: {path}")
        if path.stat().st_size != entry.size:
            raise ValueError(f"artifact size mismatch for {relative}")
        digest = entry.sha256 or file_sha256(path)
        files[relative] = {"size": entry.size, "sha256": digest}
    return files


def _stage_artifact(
    artifact_root: Path,
    content_dir: Path,
    directory: PurePosixPath,
    expected: dict[str, HubTreeEntry],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Hard-link verified content into hidden staging, hash it once, keep it."""

    staging = artifact_root.with_name(f".{artifact_root.name}.staging")
    if staging.exists():
        raise FileExistsError(
            f"artifact staging already exists; inspect it before retrying: {staging}"
        )
    files: dict[str, dict[str, Any]] = {}
    for relative, entry in expected.items():
        name = PurePosixPath(relative).relative_to(directory).as_posix()
        source = content_dir / Path(name)
        if not source.is_file():
            raise FileNotFoundError(f"release content is missing: {source}")
        if source.stat().st_size != entry.size:
            raise ValueError(
                f"release size mismatch for {relative}: "
                f"expected {entry.size}, got {source.stat().st_size}"
            )
        destination = staging / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    for relative, entry in expected.items():
        staged = staging / Path(relative)
        digest = file_sha256(staged)
        if entry.sha256 is not None and digest != entry.sha256:
            raise ValueError(f"staged SHA256 mismatch for {relative}")
        if entry.git_sha1 is not None and _git_blob_sha1(staged) != entry.git_sha1:
            raise ValueError(f"staged git blob mismatch for {relative}")
        files[relative] = {"size": entry.size, "sha256": digest}
    return staging, files


def _descriptor_text(plan: ModelRelease, model: ReleaseModel) -> str:
    manifest = plan.manifest_path(model).relative_to(plan.repo_root).as_posix()
    contract = plan.contract_path(model).relative_to(plan.repo_root).as_posix()
    return (
        "[model]\n"
        "schema_version = 1\n"
        f"id = {_toml_string(model.model_id)}\n"
        f"backend = {_toml_string(model.kind.backend_id)}\n"
        'artifact_format = "diffusers"\n'
        f"checkpoint_step = {model.checkpoint_step}\n"
        f"manifest = {_toml_string(manifest)}\n"
        f"contract = {_toml_string(contract)}\n"
        "\n"
        "[source]\n"
        'provider = "huggingface"\n'
        f"repo_id = {_toml_string(plan.source.repo_id)}\n"
        f"revision = {_toml_string(plan.source.revision)}\n"
        f"revision_label = {_toml_string(plan.source.revision_label)}\n"
    )


def _contract_text(plan: ModelRelease, model: ReleaseModel, content_dir: Path) -> str:
    template = (plan.repo_root / model.kind.contract_template).read_text()
    stats = json.loads((content_dir / "lingbot_norm_stat.json").read_text())
    q01 = stats.get("q01")
    q99 = stats.get("q99")
    if (
        not isinstance(q01, list)
        or not isinstance(q99, list)
        or not q01
        or len(q01) != len(q99)
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in (*q01, *q99)
        )
    ):
        raise ValueError("checkpoint lingbot_norm_stat.json lacks valid q01/q99 lists")
    text = _replace_scalar(
        template,
        ("components",),
        "model_subdirectory",
        _toml_string(model.directory.as_posix()),
    )
    text = _replace_scalar(text, ("normalization",), "q01_source", _toml_array(q01))
    text = _replace_scalar(text, ("normalization",), "q99_source", _toml_array(q99))
    return text


def _deployment_text(plan: ModelRelease, model: ReleaseModel, index: int) -> str:
    template = (plan.repo_root / model.kind.deployment_template).read_text()
    deployment_id = plan.deployment_id(model)
    descriptor = plan.descriptor_path(model).relative_to(plan.repo_root).as_posix()
    text = _replace_scalar(template, ("model",), "config", _toml_string(descriptor))
    text = _replace_scalar(
        text, ("tasks",), "config", _toml_string(model.catalog.as_posix())
    )
    text = _replace_scalar(text, ("deployment",), "id", _toml_string(deployment_id))
    text = _replace_scalar(
        text, ("session",), "master_port", str(plan.master_port(index))
    )
    text = _replace_scalar(text, ("server",), "port", str(plan.server_port(index)))
    text = _replace_scalar(
        text,
        ("recording",),
        "output_root",
        _toml_string(f"outputs/inference/{deployment_id}/recordings"),
    )
    return text


def _check_deployment(
    plan: ModelRelease, model: ReleaseModel, index: int, deployment: Any
) -> None:
    policy = deployment.policy_server
    if policy.backend.backend_id != model.kind.backend_id:
        raise ValueError(
            "generated deployment backend mismatch: "
            f"{policy.backend.backend_id!r} != {model.kind.backend_id!r}"
        )
    if policy.model.model_id != model.model_id:
        raise ValueError(
            f"generated deployment model mismatch: {policy.model.model_id!r}"
        )
    expected_catalog = plan.repo_root / Path(*model.catalog.parts)
    if deployment.task_catalog.path.resolve() != expected_catalog.resolve():
        raise ValueError(
            f"generated deployment catalog mismatch: {deployment.task_catalog.path}"
        )
    if deployment.server.port != plan.server_port(index):
        raise ValueError("generated deployment server port mismatch")
    if policy.master_port != plan.master_port(index):
        raise ValueError("generated deployment master port mismatch")


def _hub_tree(plan: ModelRelease) -> dict[str, HubTreeEntry]:
    from huggingface_hub import HfApi, RepoFile

    api = HfApi()
    entries: dict[str, HubTreeEntry] = {}
    for item in api.list_repo_tree(
        plan.source.repo_id,
        revision=plan.source.revision,
        recursive=True,
        expand=False,
    ):
        if not isinstance(item, RepoFile):
            continue
        lfs = getattr(item, "lfs", None)
        sha256 = None
        if lfs is not None:
            sha256 = (
                lfs["sha256"] if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            )
        entries[item.path] = HubTreeEntry(
            path=item.path,
            size=int(item.size),
            sha256=sha256,
            git_sha1=None if sha256 else item.blob_id,
        )
    if not entries:
        raise ValueError(f"release revision is empty: {plan.source.revision}")
    return entries


def _replace_scalar(text: str, table: tuple[str, ...], key: str, literal: str) -> str:
    current: tuple[str, ...] = ()
    pattern = re.compile(rf"{re.escape(key)}\s*=")
    replaced = False
    result: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped.strip("[]").strip()
            current = tuple(part.strip() for part in name.split("."))
        if current == table and pattern.match(stripped):
            indent = line[: len(line) - len(line.lstrip())]
            result.append(f"{indent}{key} = {literal}")
            replaced = True
            continue
        result.append(line)
    if not replaced:
        raise ValueError(f"template is missing [{'.'.join(table)}].{key}")
    return "\n".join(result) + "\n"


def _write_create_only(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_text() == text:
            return
        raise FileExistsError(f"refusing to replace an edited configuration: {path}")
    write_text_once(path, text)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_array(values: list[float]) -> str:
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--from-local",
        type=Path,
        help="locally transferred release root; content is identity-checked",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="register only this model key; repeat for more",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = load_model_release(args.plan, repo_root=args.repo_root)
        code, receipt = register_release(
            plan, local_root=args.from_local, only=tuple(args.only)
        )
    except (OSError, RuntimeError, ValueError) as exc:
        failure(str(exc))
        return 2
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
