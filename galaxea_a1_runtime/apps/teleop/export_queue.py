"""Background dataset export queue for finalized collection bags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

from galaxea_a1_runtime.apps.teleop.bag_export import ExportResult, export_bag


@dataclass(frozen=True)
class QueuedExport:
    episode_index: int
    bag_root: Path
    elapsed_s: float


@dataclass(frozen=True)
class ExportOutcome:
    episode_index: int
    sampled_frames: int
    stored_frames: int
    trimmed_frames: int
    elapsed_s: float
    dataset_root: str | None


@dataclass(frozen=True)
class ExportStats:
    exported: int
    rejected: int
    stored_frames: int
    pending: int


class ExportQueue:
    """Serialize exports in recording order; never write to stdout here."""

    def __init__(self, config) -> None:
        self._config = config
        self._condition = threading.Condition()
        self._pending: list[QueuedExport] = []
        self._active: QueuedExport | None = None
        self._outcomes: list[ExportOutcome] = []
        self._failure: BaseException | None = None
        self._stopping = False
        self._exported = 0
        self._rejected = 0
        self._stored_frames = 0
        thread = threading.Thread(target=self._run, name="a1-bag-export", daemon=True)
        thread.start()

    def submit(self, item: QueuedExport) -> None:
        with self._condition:
            if self._failure is not None:
                raise RuntimeError("dataset export already failed for this session")
            self._pending.append(item)
            self._condition.notify_all()

    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    def stats(self) -> ExportStats:
        with self._condition:
            return ExportStats(
                exported=self._exported,
                rejected=self._rejected,
                stored_frames=self._stored_frames,
                pending=len(self._pending) + (1 if self._active is not None else 0),
            )

    def protected_paths(self) -> tuple[Path, ...]:
        with self._condition:
            paths = [item.bag_root for item in self._pending]
            if self._active is not None:
                paths.append(self._active.bag_root)
            return tuple(paths)

    def take_outcomes(self) -> tuple[ExportOutcome, ...]:
        with self._condition:
            outcomes = tuple(self._outcomes)
            self._outcomes.clear()
            return outcomes

    def drain(self) -> int:
        with self._condition:
            while (self._pending or self._active is not None) and self._failure is None:
                self._condition.wait(0.5)
            return len(self._pending) + (1 if self._active is not None else 0)

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if not self._pending and self._stopping:
                    return
                item = self._pending.pop(0)
                self._active = item
                self._condition.notify_all()
            try:
                result: ExportResult = export_bag(item.bag_root, config=self._config)
            except BaseException as exc:
                with self._condition:
                    self._failure = exc
                    self._active = None
                    self._condition.notify_all()
                return
            with self._condition:
                self._active = None
                if result.stored_frames:
                    self._exported += 1
                    self._stored_frames += result.stored_frames
                else:
                    self._rejected += 1
                self._outcomes.append(
                    ExportOutcome(
                        episode_index=item.episode_index,
                        sampled_frames=result.sampled_frames,
                        stored_frames=result.stored_frames,
                        trimmed_frames=result.trimmed_frames,
                        elapsed_s=item.elapsed_s,
                        dataset_root=str(result.root) if result.stored_frames else None,
                    )
                )
                self._condition.notify_all()
