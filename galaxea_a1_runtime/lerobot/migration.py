"""Migration planning helpers for old A1 data layouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LegacyDatasetKind(StrEnum):
    RAW_EPISODES = "raw-episodes"
    LEROBOT_V21 = "lerobot-v2.1"


@dataclass(frozen=True)
class MigrationPlan:
    kind: LegacyDatasetKind
    source: Path | str
    target_repo_id: str
    target_root: Path | None
    command: tuple[str, ...]
    notes: tuple[str, ...]

    def shell_command(self) -> str:
        return " ".join(str(part) for part in self.command)


def plan_v21_to_v30(
    *,
    repo_id: str,
    lerobot_python: str = "python",
    target_root: Path | None = None,
) -> MigrationPlan:
    command = (
        lerobot_python,
        "-m",
        "lerobot.scripts.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
    )
    notes = (
        "Run this only after the source v2.1 dataset is backed up.",
        "The official converter aggregates parquet/video files and rewrites v3 metadata.",
    )
    return MigrationPlan(
        kind=LegacyDatasetKind.LEROBOT_V21,
        source=repo_id,
        target_repo_id=repo_id,
        target_root=target_root,
        command=command,
        notes=notes,
    )


def plan_raw_episodes_to_v30(
    *,
    source_root: Path,
    target_repo_id: str,
    target_root: Path,
) -> MigrationPlan:
    command = (
        "python",
        "-m",
        "galaxea_a1_runtime.lerobot.convert_raw",
        f"--source-root={source_root}",
        f"--repo-id={target_repo_id}",
        f"--target-root={target_root}",
    )
    notes = (
        "This is the new one-way A1 raw episode migration path.",
        "The implementation must map old camera/state/action names into the runtime contract.",
    )
    return MigrationPlan(
        kind=LegacyDatasetKind.RAW_EPISODES,
        source=source_root,
        target_repo_id=target_repo_id,
        target_root=target_root,
        command=command,
        notes=notes,
    )
