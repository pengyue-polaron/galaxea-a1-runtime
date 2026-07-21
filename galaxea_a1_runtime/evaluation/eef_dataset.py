"""Strict reader for the exact LeRobot v2.1 EEF training package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from embodied_ops.artifacts import read_json_object, read_jsonl_objects

from galaxea_a1_runtime.evaluation.metrics import summary, vector_stats
from galaxea_a1_runtime.evaluation.offline_config import OfflineEvalConfig
from galaxea_a1_runtime.evaluation.types import EpisodeRecord
from galaxea_a1_runtime.schema import A1_STATE_NAMES, EEF_ACTION_NAMES


class EefDataset:
    def __init__(self, config: OfflineEvalConfig) -> None:
        self.config = config
        self.root = config.dataset_root
        self.info = read_json_object(self.root / "meta/info.json")
        self.eef = read_json_object(self.root / "meta/eef.json")
        self.tasks = {
            int(item["task_index"]): str(item["task"])
            for item in read_jsonl_objects(self.root / "meta/tasks.jsonl")
        }
        self.episode_meta = read_jsonl_objects(self.root / "meta/episodes.jsonl")
        self._episodes: dict[int, EpisodeRecord] = {}
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        expected_info = {
            "codebase_version": "v2.1",
            "fps": 30,
            "total_episodes": len(self.episode_meta),
            "total_tasks": len(self.tasks),
        }
        mismatched = {
            key: (self.info.get(key), value)
            for key, value in expected_info.items()
            if self.info.get(key) != value
        }
        if mismatched:
            raise ValueError(f"offline dataset metadata mismatch: {mismatched}")
        if self.eef.get("repo_id") != self.config.dataset_repo_id:
            raise ValueError(
                f"offline dataset repo mismatch: {self.eef.get('repo_id')!r}"
            )
        features = self.info.get("features", {})
        if features.get("observation.state", {}).get("names") != list(A1_STATE_NAMES):
            raise ValueError("offline dataset state names do not match the A1 contract")
        if features.get("action", {}).get("names") != list(EEF_ACTION_NAMES):
            raise ValueError(
                "offline dataset action names do not match the EEF contract"
            )
        if self.eef.get("action", {}).get("semantics") != (
            "EEF target relative to episode initial feedback pose"
        ):
            raise ValueError("offline dataset is not episode-relative EEF")

    def episode(self, episode_index: int) -> EpisodeRecord:
        cached = self._episodes.get(episode_index)
        if cached is not None:
            return cached
        if not 0 <= episode_index < len(self.episode_meta):
            raise IndexError(f"invalid episode index: {episode_index}")
        path = self._data_path(episode_index)
        frame = pd.read_parquet(path)
        required = {
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "task_index",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        states = np.stack(frame["observation.state"]).astype(np.float32)
        actions = np.stack(frame["action"]).astype(np.float32)
        timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
        task_ids = frame["task_index"].to_numpy(dtype=np.int64)
        task_id = int(task_ids[0])
        expected_length = int(self.episode_meta[episode_index]["length"])
        if (
            states.shape != (expected_length, len(A1_STATE_NAMES))
            or actions.shape != (expected_length, len(EEF_ACTION_NAMES))
            or not np.isfinite(states).all()
            or not np.isfinite(actions).all()
            or not np.array_equal(
                frame["frame_index"].to_numpy(dtype=np.int64),
                np.arange(expected_length),
            )
            or not np.all(
                frame["episode_index"].to_numpy(dtype=np.int64) == episode_index
            )
            or not np.all(task_ids == task_id)
        ):
            raise ValueError(f"invalid offline episode arrays: {path}")
        task = self.tasks.get(task_id)
        if task is None or self.episode_meta[episode_index].get("tasks") != [task]:
            raise ValueError(f"episode {episode_index} task metadata is inconsistent")
        record = EpisodeRecord(
            episode_index=episode_index,
            task_index=task_id,
            task=task,
            states=states,
            actions=actions,
            timestamps=timestamps,
        )
        self._episodes[episode_index] = record
        return record

    def preflight(self) -> dict[str, Any]:
        episodes = [self.episode(index) for index in range(len(self.episode_meta))]
        actions = np.concatenate([episode.actions for episode in episodes])
        states = np.concatenate([episode.states for episode in episodes])
        if actions.shape[0] != int(self.info["total_frames"]):
            raise ValueError("offline dataset frame total does not match info.json")
        expected_step = 1.0 / float(self.info["fps"])
        max_timestamp_error = max(
            float(
                np.max(
                    np.abs(
                        episode.timestamps
                        - np.arange(episode.length, dtype=np.float64) * expected_step
                    )
                )
            )
            for episode in episodes
        )
        state_quat_norm = np.linalg.norm(states[:, 3:7], axis=1)
        action_quat_norm = np.linalg.norm(actions[:, 3:7], axis=1)
        if (
            np.min(state_quat_norm) < 0.25
            or np.min(action_quat_norm) < 0.25
            or np.min(states[:, -1]) < 0.0
            or np.max(states[:, -1]) > 1.0
            or np.min(actions[:, -1]) < 0.0
            or np.max(actions[:, -1]) > 1.0
        ):
            raise ValueError("offline dataset violates quaternion/gripper invariants")
        return {
            "repo_id": self.eef["repo_id"],
            "source_dataset": self.eef["source_dataset"],
            "episodes": len(episodes),
            "tasks": len(self.tasks),
            "frames": int(actions.shape[0]),
            "fps": int(self.info["fps"]),
            "max_timestamp_grid_error_s": max_timestamp_error,
            "state_quaternion_norm": summary(state_quat_norm),
            "action_quaternion_norm": summary(action_quat_norm),
            "state_gripper_range": [
                float(states[:, -1].min()),
                float(states[:, -1].max()),
            ],
            "action_gripper_range": [
                float(actions[:, -1].min()),
                float(actions[:, -1].max()),
            ],
            "state_stats": vector_stats(states),
            "action_stats": vector_stats(actions),
        }

    def episodes_by_task(self) -> dict[int, list[int]]:
        task_ids_by_text = {value: index for index, value in self.tasks.items()}
        result = {task_index: [] for task_index in self.tasks}
        for episode_index, item in enumerate(self.episode_meta):
            task = str(item["tasks"][0])
            result[task_ids_by_text[task]].append(episode_index)
        return result

    def frames(
        self, episode_index: int, indices: list[int]
    ) -> dict[int, dict[str, np.ndarray]]:
        unique = sorted(set(indices))
        episode = self.episode(episode_index)
        if not unique or unique[0] < 0 or unique[-1] >= episode.length:
            raise IndexError(f"episode {episode_index} frame selection is out of range")
        camera_keys = tuple(self.eef["cameras"]["ordered_keys"])
        decoded = {
            key: _decode_video_frames(self._video_path(episode_index, key), unique)
            for key in camera_keys
        }
        return {
            index: {key: decoded[key][index] for key in camera_keys} for index in unique
        }

    def _data_path(self, episode_index: int) -> Path:
        return self.root / str(self.info["data_path"]).format(
            episode_chunk=episode_index // int(self.info["chunks_size"]),
            episode_index=episode_index,
        )

    def _video_path(self, episode_index: int, key: str) -> Path:
        return self.root / str(self.info["video_path"]).format(
            episode_chunk=episode_index // int(self.info["chunks_size"]),
            episode_index=episode_index,
            video_key=key,
        )


def _decode_video_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    import av

    wanted = set(indices)
    result: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                result[index] = frame.to_ndarray(format="rgb24")
                if len(result) == len(wanted):
                    break
    missing = sorted(wanted - set(result))
    if missing:
        raise ValueError(f"video {path} is missing decoded frames: {missing}")
    return result
