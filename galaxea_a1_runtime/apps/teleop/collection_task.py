"""Pure normalization of one collection episode's task identity."""

from __future__ import annotations

from galaxea_a1_runtime.lerobot.direct_recording import normalize_dataset_task


def normalize_collection_task(value: str) -> str:
    """Normalize through the authoritative direct-dataset task contract."""

    return normalize_dataset_task(value)
