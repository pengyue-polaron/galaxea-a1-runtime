from pathlib import Path
import numpy as np
import jax
import sys
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class LocalDataset:
    """Local dataset wrapper for LeRobotDataset to mimic DroidRldsDataset behavior."""

    def __init__(self, local_data_dir: str, batch_size: int = 32, action_horizon: int = 1, shuffle: bool = True):
        self.data_dir = Path(local_data_dir)
        self.batch_size = batch_size
        self.action_horizon = action_horizon
        self.shuffle = shuffle

        # Load the LeRobotDataset
        print(self.data_dir)
        self.dataset = LeRobotDataset(revision="v3.0",repo_id="local_data", root="/home/jolia/DataCoach/data/formatted_data/test")
        breakpoint()
        # Build indices for frames / action chunks
        self.frame_indices = self._build_indices()
        if self.shuffle:
            np.random.shuffle(self.frame_indices)

    def _build_indices(self):
        """Build list of (start_idx) for each action chunk."""
        total_frames = len(self.dataset)
        indices = []
        for i in range(total_frames - self.action_horizon + 1):
            indices.append(i)
        return indices

    def __iter__(self):
        """Yield batches of data."""
        batch = []
        for idx in self.frame_indices:
            # Extract action chunk
            obs = self.dataset[idx]["state"]
            img = self.dataset[idx]["image"]
            actions = np.stack([self.dataset[idx + i]["action"] for i in range(self.action_horizon)], axis=0)

            sample = {
                "observation": {"image": img, "state": obs},
                "actions": actions,
            }
            batch.append(sample)

            if len(batch) == self.batch_size:
                # Convert to dict of arrays for JAX/torch compatibility
                batch_dict = self._collate_batch(batch)
                yield batch_dict
                batch = []

        # Yield last batch if not empty
        if batch:
            yield self._collate_batch(batch)

    def _collate_batch(self, batch):
        """Convert list of samples to batch dict."""
        obs_images = np.stack([s["observation"]["image"] for s in batch], axis=0)
        obs_states = np.stack([s["observation"]["state"] for s in batch], axis=0)
        actions = np.stack([s["actions"] for s in batch], axis=0)
        return {
            "observation": {"image": obs_images, "state": obs_states},
            "actions": actions,
        }

    def __len__(self):
        return len(self.frame_indices)
