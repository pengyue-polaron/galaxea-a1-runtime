"""Offline LeRobot conversion and deterministic packaging helpers."""

from __future__ import annotations

__all__ = [
    "build_dataset_create_kwargs",
    "create_lerobot_dataset",
    "make_a1_teleop_processors",
]


def __getattr__(name: str):
    if name == "make_a1_teleop_processors":
        from .hardware import make_a1_teleop_processors

        return make_a1_teleop_processors
    if name in {"build_dataset_create_kwargs", "create_lerobot_dataset"}:
        from .dataset import build_dataset_create_kwargs, create_lerobot_dataset

        return {
            "build_dataset_create_kwargs": build_dataset_create_kwargs,
            "create_lerobot_dataset": create_lerobot_dataset,
        }[name]
    raise AttributeError(name)
