"""Atomic per-episode source clocks alongside LeRobot's nominal training clock."""

from __future__ import annotations

import json
from pathlib import Path

from galaxea_a1_runtime.schema import BAG_CAPTURE_CONTRACT as CAPTURE_CONTRACT


def timing_paths(root: Path, episode: int):
    stem = root / "meta/timing" / f"episode-{episode:06d}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".json")


def write_timing(root, *, episode, rows, source):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table, manifest = timing_paths(root, episode)
    table.parent.mkdir(parents=True, exist_ok=True)
    if table.exists() or manifest.exists():
        raise ValueError("episode timing already exists")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {**row, "episode_index": episode, "frame_index": i}
                for i, row in enumerate(rows)
            ]
        ),
        table,
    )
    manifest.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")


def validate_timing(root: Path, *, info, provenance):
    if "capture_contract" not in provenance:
        return
    if provenance["capture_contract"] != CAPTURE_CONTRACT:
        raise ValueError("unsupported collection capture contract")
    import pyarrow.parquet as pq

    counts = {}
    for file in sorted((root / "data").rglob("*.parquet")):
        for row in pq.read_table(file, columns=["episode_index"]).to_pylist():
            episode = int(row["episode_index"])
            counts[episode] = counts.get(episode, 0) + 1
    ids = set()
    image_roles = ["front", "wrist"]
    if "observation.images.front_depth" in info["features"]:
        image_roles.append("depth")
    state_roles = ("eef", "joints", "gripper", "action", "gripper_action")
    prefixes = image_roles + [
        f"{role}_{end}" for role in state_roles for end in ("left", "right")
    ]
    required = {
        "sample_time_ns",
        "grid_index",
        "episode_index",
        "frame_index",
        "camera_pair_skew_ns",
    }
    required.update(
        f"{prefix}_{field}"
        for prefix in prefixes
        for field in ("source_ns", "receive_ns", "receive_fallback", "message_index")
    )
    required.update(
        f"{role}_{field}" for role in image_roles for field in ("reused", "offset_ns")
    )
    limits = provenance["quality_checks"]
    for episode in range(info["total_episodes"]):
        table, manifest = timing_paths(root, episode)
        rows = pq.read_table(table).to_pylist()
        source = json.loads(manifest.read_text())
        if not rows or len(rows) != counts.get(episode):
            raise ValueError("timing rows do not match LeRobot episode frames")
        bag_id = source.get("bag_id")
        if not isinstance(bag_id, str) or not bag_id or bag_id in ids:
            raise ValueError("invalid or duplicate source bag identity")
        ids.add(bag_id)
        if not source.get("files_sha256"):
            raise ValueError("episode lacks raw bag content hashes")
        if (
            source.get("capture_contract") != CAPTURE_CONTRACT
            or source.get("status") != "finalized"
            or source.get("disposition") != "save"
            or source.get("clock") != "ros_system_time"
        ):
            raise ValueError("invalid source capture manifest")
        if source.get("config_sha256") != provenance["config_sha256"]:
            raise ValueError("source and dataset configurations differ")
        if any(set(row) != required for row in rows):
            raise ValueError("timing record columns differ from capture contract")
        first_grid = rows[0]["grid_index"]
        if first_grid < 0:
            raise ValueError("negative sample grid index")
        origin = source["start_ns"]
        for i, row in enumerate(rows):
            if (
                row["episode_index"] != episode
                or row["frame_index"] != i
                or row["grid_index"] != first_grid + i
            ):
                raise ValueError("non-contiguous episode timing indices")
            expected = origin + row["grid_index"] * 1_000_000_000 // info["fps"]
            if (
                row["sample_time_ns"] != expected
                or not origin <= expected < source["end_ns"]
            ):
                raise ValueError("training clock differs from physical-time grid")
            for key, value in row.items():
                if key.endswith("_source_ns"):
                    prefix = key.removesuffix("_source_ns")
                    if not isinstance(value, int) or value < 0:
                        raise ValueError("invalid source timestamp")
                    if (
                        row[f"{prefix}_receive_fallback"] != (value == 0)
                        or row[f"{prefix}_receive_ns"] <= 0
                    ):
                        raise ValueError("invalid source/receive timestamp provenance")
            for role in image_roles:
                if (
                    row[f"{role}_source_ns"] <= 0
                    or row[f"{role}_offset_ns"] != row[f"{role}_source_ns"] - expected
                ):
                    raise ValueError("invalid camera source timing")
                if abs(row[f"{role}_offset_ns"]) > round(
                    limits["max_camera_age_s"] * 1e9
                ):
                    raise ValueError("camera selection exceeds freshness")
                if i and row[f"{role}_reused"] != (
                    row[f"{role}_message_index"] == rows[i - 1][f"{role}_message_index"]
                ):
                    raise ValueError("incorrect camera reuse flag")
            actual_skew = abs(row["front_source_ns"] - row["wrist_source_ns"])
            if row["camera_pair_skew_ns"] != actual_skew or actual_skew > round(
                limits["max_camera_pair_skew_s"] * 1e9
            ):
                raise ValueError("invalid camera pair skew")
            for role in state_roles:
                left = row[f"{role}_left_source_ns"] or row[f"{role}_left_receive_ns"]
                right = (
                    row[f"{role}_right_source_ns"] or row[f"{role}_right_receive_ns"]
                )
                max_age = round(
                    limits[
                        "max_eef_feedback_age_s"
                        if role == "eef"
                        else "max_joint_feedback_age_s"
                    ]
                    * 1e9
                )
                if (
                    left > expected
                    or expected - left > max_age
                    or right - left > max_age
                ):
                    raise ValueError("invalid state/action temporal bounds")
                if role in ("action", "gripper_action"):
                    if (
                        right != left
                        or row[f"{role}_left_message_index"]
                        != row[f"{role}_right_message_index"]
                    ):
                        raise ValueError("command was interpolated")
                elif right < expected or right - expected > max_age:
                    raise ValueError("state interpolation does not bracket sample")
            for role in ("action", "gripper_action"):
                effective = (
                    row[f"{role}_left_source_ns"] or row[f"{role}_left_receive_ns"]
                )
                if effective > expected:
                    raise ValueError("action uses a future command")
    expected_paths = {
        p
        for episode in range(info["total_episodes"])
        for p in timing_paths(root, episode)
    }
    if set((root / "meta/timing").glob("*")) != expected_paths:
        raise ValueError("unexpected episode timing files")


def reject_duplicate_bag(root: Path, bag_id: str):
    for file in (root / "meta/timing").glob("*.json"):
        if json.loads(file.read_text()).get("bag_id") == bag_id:
            raise ValueError(f"bag {bag_id} was already exported into this dataset")
