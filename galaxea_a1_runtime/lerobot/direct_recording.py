"""Atomic, directly recorded LeRobotDataset v3 episode transactions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id

from galaxea_a1_runtime.constants import LEROBOT_DATASET_FORMAT
from galaxea_a1_runtime.filesystem import OutputDirectoryTransaction, atomic_write_text
from galaxea_a1_runtime.lerobot.dataset import (
    DatasetConfig,
    build_dataset_create_kwargs,
    create_lerobot_dataset,
    resume_lerobot_dataset,
)
from galaxea_a1_runtime.lerobot.dataset_package import copy_dataset_tree
from galaxea_a1_runtime.schema import DatasetContract

DIRECT_DATASET_SCHEMA_VERSION = "galaxea_a1_lerobot_dataset_v3_v1"
PROVENANCE_PATH = Path("meta/galaxea_a1.json")


@dataclass(frozen=True)
class DirectDatasetState:
    total_episodes: int
    total_frames: int
    task: str | None


def dataset_repo_id(prefix: str, experiment: str) -> str:
    """Build the configured Hugging Face dataset ID for one experiment."""

    prefix = prefix.strip().rstrip("-")
    if prefix.count("/") != 1 or any(character.isspace() for character in prefix):
        raise ValueError(
            "collection.repo_id_prefix must be a whitespace-free 'owner/name' prefix"
        )
    repo_id = f"{prefix}-{experiment}"
    try:
        validate_repo_id(repo_id)
    except HFValidationError as exc:
        raise ValueError(
            f"invalid collection dataset repo_id {repo_id!r}: {exc}"
        ) from exc
    return repo_id


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
    target_root: Path,
    *,
    repo_id: str,
    fps: int,
    contract: DatasetContract,
    use_videos: bool,
    experiment: str,
    expected_task: str | None = None,
) -> DirectDatasetState:
    """Validate an existing direct dataset without importing LeRobot."""

    target_root = target_root.expanduser().resolve()
    validate_no_staging_outputs(target_root)
    if not target_root.exists():
        return DirectDatasetState(0, 0, None)
    if not target_root.is_dir():
        raise ValueError(f"direct dataset target is not a directory: {target_root}")

    info = _read_json(target_root / "meta/info.json", label="LeRobot info")
    provenance = _read_json(
        target_root / PROVENANCE_PATH, label="Galaxea collection provenance"
    )
    if info.get("codebase_version") != LEROBOT_DATASET_FORMAT:
        raise ValueError(
            f"direct dataset must use LeRobotDataset {LEROBOT_DATASET_FORMAT}"
        )
    if info.get("robot_type") != "galaxea_a1":
        raise ValueError("direct dataset robot_type must be 'galaxea_a1'")
    if info.get("fps") != fps:
        raise ValueError(
            f"direct dataset FPS changed: existing={info.get('fps')}, configured={fps}"
        )
    _validate_features(info.get("features"), contract=contract, use_videos=use_videos)
    if provenance.get("schema_version") != DIRECT_DATASET_SCHEMA_VERSION:
        raise ValueError("direct dataset has an unsupported Galaxea schema")
    if provenance.get("repo_id") != repo_id:
        raise ValueError(
            "direct dataset repo_id does not match the tracked collection config"
        )
    if provenance.get("experiment") != experiment:
        raise ValueError(
            "direct dataset experiment identity does not match its directory"
        )
    task = provenance.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("direct dataset provenance has no valid task")
    if expected_task is not None and task != expected_task:
        raise ValueError(
            f"collection task mismatch for {experiment}: "
            f"existing={task!r}, requested={expected_task!r}"
        )
    total_episodes = _non_negative_int(info, "total_episodes")
    total_frames = _non_negative_int(info, "total_frames")
    if total_episodes == 0 or total_frames == 0:
        raise ValueError("a committed direct dataset must contain at least one episode")
    if provenance.get("total_episodes") != total_episodes:
        raise ValueError("Galaxea provenance episode count does not match LeRobot info")
    if provenance.get("total_frames") != total_frames:
        raise ValueError("Galaxea provenance frame count does not match LeRobot info")
    return DirectDatasetState(total_episodes, total_frames, task)


class DirectLeRobotEpisode:
    """Record one episode into a hidden dataset snapshot and commit it atomically."""

    def __init__(
        self,
        *,
        target_root: Path,
        repo_id: str,
        fps: int,
        contract: DatasetContract,
        use_videos: bool,
        experiment: str,
        task: str,
        provenance: dict[str, Any],
    ) -> None:
        self.target_root = target_root.expanduser().resolve()
        self.repo_id = repo_id
        self.fps = fps
        self.contract = contract
        self.use_videos = use_videos
        self.experiment = experiment
        self.task = task
        self.provenance = dict(provenance)
        self._transaction: OutputDirectoryTransaction | None = None
        self._dataset: Any | None = None
        self._finalized = False
        self._committed = False

    def __enter__(self) -> "DirectLeRobotEpisode":
        state = inspect_direct_dataset(
            self.target_root,
            repo_id=self.repo_id,
            fps=self.fps,
            contract=self.contract,
            use_videos=self.use_videos,
            experiment=self.experiment,
            expected_task=self.task,
        )
        exists = state.total_episodes > 0
        if exists:
            existing_provenance = _read_json(
                self.target_root / PROVENANCE_PATH,
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
            self.target_root,
            overwrite=exists,
            precreate_staging=False,
        )
        transaction.__enter__()
        self._transaction = transaction
        assert transaction.path is not None
        try:
            if exists:
                copy_dataset_tree(
                    self.target_root,
                    transaction.path,
                    skip_roots=(),
                    hardlink_roots=("data", "videos", "images"),
                )
                self._dataset = resume_lerobot_dataset(
                    repo_id=self.repo_id, root=transaction.path
                )
            else:
                self._dataset = create_lerobot_dataset(
                    config=DatasetConfig(
                        repo_id=self.repo_id,
                        root=transaction.path,
                        fps=self.fps,
                        use_videos=self.use_videos,
                    ),
                    contract=self.contract,
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
        info = _read_json(
            self._transaction.path / "meta/info.json", label="staged LeRobot info"
        )
        payload = {
            **self.provenance,
            "schema_version": DIRECT_DATASET_SCHEMA_VERSION,
            "repo_id": self.repo_id,
            "experiment": self.experiment,
            "task": self.task,
            "total_episodes": _non_negative_int(info, "total_episodes"),
            "total_frames": _non_negative_int(info, "total_frames"),
        }
        atomic_write_text(
            self._transaction.path / PROVENANCE_PATH,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        inspect_direct_dataset(
            self._transaction.path,
            repo_id=self.repo_id,
            fps=self.fps,
            contract=self.contract,
            use_videos=self.use_videos,
            experiment=self.experiment,
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


def _validate_features(
    actual: Any, *, contract: DatasetContract, use_videos: bool
) -> None:
    if not isinstance(actual, dict):
        raise ValueError("direct dataset info.features must be a table")
    expected = build_dataset_create_kwargs(
        config=DatasetConfig(
            repo_id="validation/dataset",
            root=Path("validation-only"),
            fps=1,
            use_videos=use_videos,
        ),
        contract=contract,
    )["features"]
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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _non_negative_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value
