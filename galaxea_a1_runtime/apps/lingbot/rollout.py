"""Validated indexing for one LingBot action tensor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LingBotActionChunk:
    values: np.ndarray
    start_frame: int
    end_frame: int
    cache_state: np.ndarray
    actions_per_observation: int
    first: bool

    @classmethod
    def from_response(
        cls,
        action: np.ndarray,
        *,
        first: bool,
        execute_frames: int,
        observations_per_frame: int = 4,
    ) -> "LingBotActionChunk":
        values = np.asarray(action, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 8:
            raise RuntimeError(
                f"Expected LingBot action shape (8, F, H), got {values.shape}. "
                "Restart the server with the corrected Galaxea A1 config."
            )
        horizon = values.shape[2]
        if horizon % observations_per_frame != 0:
            raise RuntimeError(
                f"Action horizon must be divisible by {observations_per_frame}, "
                f"got {horizon}"
            )
        start_frame = 1 if first else 0
        end_frame = min(values.shape[1], start_frame + execute_frames)
        if end_frame <= start_frame:
            raise RuntimeError(
                f"No executable LingBot frames: first={first}, action_shape={values.shape}"
            )
        cache_state = (
            values[:, :end_frame].copy()
            if first
            else values[:, start_frame:end_frame].copy()
        )
        return cls(
            values=values,
            start_frame=start_frame,
            end_frame=end_frame,
            cache_state=cache_state,
            actions_per_observation=horizon // observations_per_frame,
            first=first,
        )

    @property
    def total_steps(self) -> int:
        return (self.end_frame - self.start_frame) * self.values.shape[2]

    def steps(self):
        for frame_index in range(self.start_frame, self.end_frame):
            cache_frame_index = (
                frame_index if self.first else frame_index - self.start_frame
            )
            for step_index in range(self.values.shape[2]):
                yield (
                    frame_index,
                    step_index,
                    cache_frame_index,
                    self.values[:, frame_index, step_index],
                )

    def needs_observation_after(self, step_index: int) -> bool:
        return (step_index + 1) % self.actions_per_observation == 0
