"""Atomic, directly recorded LeRobotDataset v3 episode transactions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.filesystem import OutputDirectoryTransaction, atomic_write_text
from galaxea_a1_runtime.lerobot.dataset import (
    DatasetConfig,
    LEROBOT_GENERATED_FEATURES,
    build_dataset_create_kwargs,
    create_lerobot_dataset,
    resume_lerobot_dataset,
    validate_dataset_repo_id,
)
from galaxea_a1_runtime.lerobot.dataset_package import (
    copy_dataset_tree,
    non_negative_json_int,
    read_json,
)
from galaxea_a1_runtime.lerobot.integrity import validate_lerobot_v3_payloads
from galaxea_a1_runtime.schema import DIRECT_DATASET_SCHEMA_VERSION, DatasetContract

PROVENANCE_PATH = Path("meta/galaxea_a1.json")


@dataclass(frozen=True)
class DirectDatasetState:
    total_episodes: int
    total_frames: int
    task: str | None


@dataclass(frozen=True)
class DirectDatasetIdentity:
    """Stable identity and feature contract for one directly recorded dataset."""

    target_root: Path
    repo_id: str
    fps: int
    contract: DatasetContract
    experiment: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_root", self.target_root.expanduser().resolve())
        DatasetConfig(
            repo_id=self.repo_id,
            root=self.target_root,
            fps=self.fps,
        ).validate()
        if not self.experiment or self.experiment.strip() != self.experiment:
            raise ValueError("direct dataset experiment must be a non-empty identity")


@dataclass(frozen=True)
class InspectedDirectDataset:
    """A discovered direct dataset whose complete local graph was validated."""

    identity: DirectDatasetIdentity
    state: DirectDatasetState


def dataset_repo_id(prefix: str, experiment: str) -> str:
    """Build the configured Hugging Face dataset ID for one experiment."""

    prefix = prefix.strip().rstrip("-")
    if prefix.count("/") != 1 or any(character.isspace() for character in prefix):
        raise ValueError(
            "collection.repo_id_prefix must be a whitespace-free 'owner/name' prefix"
        )
    repo_id = f"{prefix}-{experiment}"
    validate_dataset_repo_id(repo_id, label="collection dataset repo_id")
    return repo_id


def normalize_dataset_task(value: str) -> str:
    """Return one canonical single-line task or reject ambiguous text."""

    task = value.strip()
    if not task:
        raise ValueError("dataset task must not be empty")
    if "\n" in task or "\r" in task:
        raise ValueError("dataset task must be a single line")
    return task


def validate_no_staging_outputs(target_root: Path) -> None:
    """Block collection beside an interrupted direct-dataset transaction."""

    leftovers = sorted(
        path.name for path in target_root.parent.glob(f".{target_root.name}.staging-*")
    )
    if leftovers:
        raise ValueError(
            "direct dataset has uncommitted staging output; inspect and remove or "
            f"quarantine it before collecting: {leftovers}"
        )


def inspect_direct_dataset(
    identity: DirectDatasetIdentity,
    *,
    expected_task: str | None = None,
) -> DirectDatasetState:
    """Validate the metadata and every referenced payload without importing LeRobot."""

    target_root = identity.target_root
    validate_no_staging_outputs(target_root)
    if not target_root.exists():
        return DirectDatasetState(0, 0, None)
    if not target_root.is_dir():
        raise ValueError(f"direct dataset target is not a directory: {target_root}")

    info = read_json(target_root / "meta/info.json", label="LeRobot info")
    provenance = read_json(
        target_root / PROVENANCE_PATH, label="Galaxea collection provenance"
    )
    if info.get("codebase_version") != LEROBOT_DATASET_FORMAT:
        raise ValueError(
            f"direct dataset must use LeRobotDataset {LEROBOT_DATASET_FORMAT}"
        )
    if info.get("robot_type") != "galaxea_a1":
        raise ValueError("direct dataset robot_type must be 'galaxea_a1'")
    if info.get("fps") != identity.fps:
        raise ValueError(
            "direct dataset FPS changed: "
            f"existing={info.get('fps')}, configured={identity.fps}"
        )
    _validate_features(
        info.get("features"),
        contract=identity.contract,
    )
    if provenance.get("schema_version") != DIRECT_DATASET_SCHEMA_VERSION:
        raise ValueError("direct dataset has an unsupported Galaxea schema")
    if provenance.get("repo_id") != identity.repo_id:
        raise ValueError(
            "direct dataset repo_id does not match the tracked collection config"
        )
    if provenance.get("experiment") != identity.experiment:
        raise ValueError(
            "direct dataset experiment identity does not match its directory"
        )
    task = provenance.get("task")
    if not isinstance(task, str):
        raise ValueError("direct dataset provenance has no valid task")
    try:
        normalized_task = normalize_dataset_task(task)
    except ValueError as exc:
        raise ValueError("direct dataset provenance has no valid task") from exc
    if normalized_task != task:
        raise ValueError("direct dataset provenance task is not normalized")
    if expected_task is not None and task != expected_task:
        raise ValueError(
            f"collection task mismatch for {identity.experiment}: "
            f"existing={task!r}, requested={expected_task!r}"
        )
    total_episodes = non_negative_json_int(info, "total_episodes")
    total_frames = non_negative_json_int(info, "total_frames")
    if total_episodes == 0 or total_frames == 0:
        raise ValueError("a committed direct dataset must contain at least one episode")
    if provenance.get("total_episodes") != total_episodes:
        raise ValueError("Galaxea provenance episode count does not match LeRobot info")
    if provenance.get("total_frames") != total_frames:
        raise ValueError("Galaxea provenance frame count does not match LeRobot info")
    validate_lerobot_v3_payloads(
        target_root,
        info=info,
        total_episodes=total_episodes,
        total_frames=total_frames,
        expected_task=task,
    )
    return DirectDatasetState(total_episodes, total_frames, task)


def discover_direct_dataset(
    target_root: Path,
    *,
    contract: DatasetContract,
) -> InspectedDirectDataset:
    """Discover identity from canonical provenance, then validate the complete dataset."""

    target_root = target_root.expanduser().resolve()
    if not target_root.is_dir():
        raise ValueError(f"direct dataset source does not exist: {target_root}")
    info = read_json(target_root / "meta/info.json", label="LeRobot info")
    provenance = read_json(
        target_root / PROVENANCE_PATH, label="Galaxea collection provenance"
    )
    repo_id = _non_empty_string(provenance, "repo_id")
    experiment = _non_empty_string(provenance, "experiment")
    fps = _positive_int(info.get("fps"), label="fps")
    identity = DirectDatasetIdentity(
        target_root=target_root,
        repo_id=repo_id,
        fps=fps,
        contract=contract,
        experiment=experiment,
    )
    return InspectedDirectDataset(identity, inspect_direct_dataset(identity))


class DirectLeRobotEpisode:
    """Record one episode into a hidden dataset snapshot and commit it atomically."""

    def __init__(
        self,
        *,
        identity: DirectDatasetIdentity,
        task: str,
        provenance: dict[str, Any],
    ) -> None:
        self.identity = identity
        self.task = normalize_dataset_task(task)
        self.provenance = dict(provenance)
        self._transaction: OutputDirectoryTransaction | None = None
        self._dataset: Any | None = None
        self._finalized = False
        self._committed = False

    def __enter__(self) -> DirectLeRobotEpisode:
        state = inspect_direct_dataset(
            self.identity,
            expected_task=self.task,
        )
        exists = state.total_episodes > 0
        if exists:
            existing_provenance = read_json(
                self.identity.target_root / PROVENANCE_PATH,
                label="Galaxea collection provenance",
            )
            differences = sorted(
                key
                for key, value in self.provenance.items()
                if existing_provenance.get(key) != value
            )
            if differences:
                raise ValueError(
                    "direct dataset collection provenance changed; use a new "
                    f"experiment identity (fields={differences})"
                )
        transaction = OutputDirectoryTransaction(
            self.identity.target_root,
            overwrite=exists,
            precreate_staging=False,
        )
        transaction.__enter__()
        self._transaction = transaction
        assert transaction.path is not None
        try:
            if exists:
                # LeRobot resume starts fresh data/video files. Keep older
                # payloads immutable and make that scalable contract explicit.
                copy_dataset_tree(
                    self.identity.target_root,
                    transaction.path,
                    skip_roots=(),
                    hardlink_roots=("data", "videos", "images"),
                    require_hardlinks=True,
                )
                self._dataset = resume_lerobot_dataset(
                    repo_id=self.identity.repo_id, root=transaction.path
                )
            else:
                self._dataset = create_lerobot_dataset(
                    config=DatasetConfig(
                        repo_id=self.identity.repo_id,
                        root=transaction.path,
                        fps=self.identity.fps,
                    ),
                    contract=self.identity.contract,
                )
        except BaseException as error:
            transaction.__exit__(type(error), error, error.__traceback__)
            self._transaction = None
            raise
        return self

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self._dataset is None:
            raise RuntimeError("direct episode transaction has not started")
        self._dataset.add_frame(frame)

    def commit(self) -> Path:
        if self._dataset is None or self._transaction is None:
            raise RuntimeError("direct episode transaction has not started")
        if self._committed:
            raise RuntimeError("direct episode transaction was already committed")
        # The collector already owns camera-reader threads. Avoid LeRobot's
        # fork-based parallel encoder here: forking a multi-threaded live process
        # can deadlock and provides little value for an operator-confirmed save.
        self._dataset.save_episode(parallel_encoding=False)
        self._finalize()
        assert self._transaction.path is not None
        info = read_json(
            self._transaction.path / "meta/info.json", label="staged LeRobot info"
        )
        payload = {
            **self.provenance,
            "schema_version": DIRECT_DATASET_SCHEMA_VERSION,
            "repo_id": self.identity.repo_id,
            "experiment": self.identity.experiment,
            "task": self.task,
            "total_episodes": non_negative_json_int(info, "total_episodes"),
            "total_frames": non_negative_json_int(info, "total_frames"),
        }
        atomic_write_text(
            self._transaction.path / PROVENANCE_PATH,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        inspect_direct_dataset(
            replace(self.identity, target_root=self._transaction.path),
            expected_task=self.task,
        )
        result = self._transaction.commit()
        self._committed = True
        return result

    def discard(self) -> None:
        if self._dataset is None:
            return
        self._dataset.clear_episode_buffer()

    def _finalize(self) -> None:
        if self._dataset is not None and not self._finalized:
            self._dataset.finalize()
            self._finalized = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        cleanup_error: BaseException | None = None
        try:
            if not self._committed:
                self.discard()
                self._finalize()
        except BaseException as error:
            cleanup_error = error
        finally:
            if self._transaction is not None:
                self._transaction.__exit__(exc_type, exc, traceback)
        if exc_type is None and cleanup_error is not None:
            raise cleanup_error


def _validate_features(actual: Any, *, contract: DatasetContract) -> None:
    if not isinstance(actual, dict):
        raise ValueError("direct dataset info.features must be a table")
    configured = build_dataset_create_kwargs(
        config=DatasetConfig(
            repo_id="validation/dataset",
            root=Path("validation-only"),
            fps=1,
        ),
        contract=contract,
    )["features"]
    expected = {**configured, **LEROBOT_GENERATED_FEATURES}
    if set(actual) != set(expected):
        raise ValueError(
            "direct dataset feature keys do not match the canonical contract: "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for key, feature in expected.items():
        value = actual.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"direct dataset is missing feature {key!r}")
        for field in ("dtype", "shape", "names"):
            if value.get(field) != _json_value(feature.get(field)):
                raise ValueError(
                    f"direct dataset feature {key!r}.{field} does not match the tracked contract"
                )


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _non_empty_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value
