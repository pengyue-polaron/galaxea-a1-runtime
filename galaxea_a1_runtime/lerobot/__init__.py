"""Offline LeRobot conversion and deterministic packaging helpers."""

from __future__ import annotations

__all__ = ["build_dataset_create_kwargs", "create_lerobot_dataset"]


def __getattr__(name: str):
    if name in __all__:
        from .dataset import build_dataset_create_kwargs, create_lerobot_dataset

        return {
            "build_dataset_create_kwargs": build_dataset_create_kwargs,
            "create_lerobot_dataset": create_lerobot_dataset,
        }[name]
    raise AttributeError(name)
