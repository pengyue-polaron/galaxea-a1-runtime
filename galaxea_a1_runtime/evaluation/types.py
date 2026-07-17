"""Typed records shared by offline evaluation components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task_index: int
    task: str
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])
