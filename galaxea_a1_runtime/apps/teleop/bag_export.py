"""Offline, reproducible MCAP-to-LeRobot conversion with source timing records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import numpy as np
from embodied_ops.collection import LeadingStillnessTrimmer

from galaxea_a1_runtime.apps.teleop.bag_contract import (
    CAPTURE_CONTRACT,
    MANIFEST_NAME,
    config_fingerprint,
    file_digest,
    source_topics,
)
from galaxea_a1_runtime.apps.teleop.bag_reader import read_bag
from galaxea_a1_runtime.apps.teleop.dataset_contract import direct_dataset_identity
from galaxea_a1_runtime.apps.teleop.metadata import (
    DatasetProvenanceRequest,
    build_dataset_provenance,
)
from galaxea_a1_runtime.collection import (
    validate_experiment_name,
    find_joint_action_step_violation,
)
from galaxea_a1_runtime.collection.bag_alignment import align_records
from galaxea_a1_runtime.collection.lerobot_frame import build_lerobot_frame
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.hardware.image_geometry import crop_image
from galaxea_a1_runtime.lerobot.direct_recording import (
    DirectLeRobotEpisode,
    validate_direct_dataset_provenance,
)
from galaxea_a1_runtime.lerobot.timing import reject_duplicate_bag
from galaxea_a1_runtime.schema import JOINT_ACTION_NAMES_RAD
from galaxea_a1_runtime.teleop.config import (
    default_config_path,
    load_teleop_config,
    validate_collection_config,
)


@dataclass(frozen=True)
class ExportResult:
    sampled_frames: int
    stored_frames: int
    trimmed_frames: int
    root: Path


def export_provenance(config, experiment):
    repo = Path(__file__).resolve().parents[3]
    result = build_dataset_provenance(
        DatasetProvenanceRequest(
            experiment,
            required_front_roi(config.system.cameras),
            f"ros2:realsense:{config.system.cameras.wrist.serial}",
            config.path.relative_to(repo).as_posix(),
            config,
        )
    )
    result.update(
        capture_contract=CAPTURE_CONTRACT, config_sha256=config_fingerprint(config)
    )
    result["timing"] = {
        "clock": "ros_system_time",
        "storage": "meta/timing/episode-NNNNNN.parquet",
        "training_grid": "integer nanoseconds at collection FPS; stationary prefix only trimmed",
        "camera": "nearest synchronized source pair; bounded reuse explicitly recorded",
        "state": "bounded linear interpolation; quaternion SLERP",
        "action": "most recent command at or before sample time",
        "unstamped_robot": "explicit bag receive-time fallback; source timestamp remains zero",
    }
    return result


def image_array(item):
    encoding = item["encoding"].lower()
    if encoding in ("rgb8", "bgr8"):
        channels, dtype = 3, np.dtype("u1")
    elif encoding in ("16uc1", "mono16"):
        channels, dtype = 1, np.dtype(">u2" if item["bigendian"] else "<u2")
    else:
        raise ValueError(f"unsupported recorded image encoding: {encoding}")
    height, width, stride = item["height"], item["width"], item["step"]
    if (
        min(height, width) <= 0
        or stride < width * channels * dtype.itemsize
        or len(item["data"]) != height * stride
    ):
        raise ValueError("invalid recorded image dimensions/stride/payload")
    image = np.ndarray(
        (height, width, channels),
        dtype=dtype,
        buffer=item["data"],
        strides=(stride, channels * dtype.itemsize, dtype.itemsize),
    ).copy()
    if channels == 1:
        return image[..., 0].astype(np.uint16)
    return image[..., ::-1].copy() if encoding == "rgb8" else image


def export_bag(episode: Path, *, config, experiment: str | None = None) -> ExportResult:
    """Serialize appends across collection and offline conversion processes."""
    import fcntl

    validate_collection_config(config)
    manifest = json.loads((episode / MANIFEST_NAME).read_text())
    experiment = validate_experiment_name(experiment or manifest["experiment"])
    lock_root = config.collection.dataset_root.parent / ".bag-export-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / f"{experiment}.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another exporter owns dataset {experiment}") from exc
        return _export_bag(episode, config=config, experiment=experiment)


def _export_bag(
    episode: Path, *, config, experiment: str | None = None
) -> ExportResult:
    validate_collection_config(config)
    episode = episode.resolve(strict=True)
    manifest = json.loads((episode / MANIFEST_NAME).read_text())
    if (
        manifest.get("capture_contract") != CAPTURE_CONTRACT
        or manifest.get("status") != "finalized"
        or manifest.get("disposition") != "save"
    ):
        raise ValueError(
            "only finalized, operator-saved collection bags can be exported"
        )
    if manifest.get("clock") != "ros_system_time" or manifest.get(
        "topics"
    ) != source_topics(config.system):
        raise ValueError("bag topics/clock differ from the collection contract")
    if manifest.get("config_sha256") != config_fingerprint(config):
        raise ValueError(
            "bag configuration differs; use the recorded configuration before exporting"
        )
    experiment = validate_experiment_name(experiment or manifest["experiment"])
    identity = direct_dataset_identity(config, experiment)
    provenance = export_provenance(config, experiment)
    validate_direct_dataset_provenance(identity, provenance)
    reject_duplicate_bag(identity.target_root, manifest["bag_id"])
    records = defaultdict(list)
    for item in read_bag(episode):
        records[item["role"]].append(item)
    if set(records) != set(manifest["topics"]):
        raise ValueError("bag is missing required collection streams")
    trimmer = LeadingStillnessTrimmer(config.collection.leading_stillness)
    samples = []
    sampled = 0
    for sample in align_records(
        records, config=config, start_ns=manifest["start_ns"], end_ns=manifest["end_ns"]
    ):
        sampled += 1
        samples.extend(trimmer.push(sample, sample.action))
    result = ExportResult(
        sampled, len(samples), trimmer.result.trimmed_frames, identity.target_root
    )
    if not samples:
        return result
    violation = find_joint_action_step_violation(
        (s.action for s in samples),
        action_names=JOINT_ACTION_NAMES_RAD,
        max_step_rad=config.bridge.max_joint_action_step_rad,
    )
    if violation:
        raise ValueError(f"recorded action discontinuity: {violation.describe()}")
    needed = {
        (role, index) for sample in samples for role, index in sample.images.items()
    }
    source = {
        **manifest,
        "raw_episode_path": str(episode),
        "files_sha256": {
            str(p.relative_to(episode)): file_digest(p)
            for p in sorted(episode.rglob("*"))
            if p.is_file() and p.name != "recorder.log"
        },
    }
    crop = required_front_roi(config.system.cameras)
    # Only scratch decoding is ephemeral; raw bags and committed datasets are immutable.
    with tempfile.TemporaryDirectory(prefix="a1-bag-export-") as scratch:
        scratch = Path(scratch)
        available = set()
        for item in read_bag(episode, images=True):
            key = item["role"], item["index"]
            if key not in needed:
                continue
            image = image_array(item)
            if item["role"] in ("front", "depth") and crop is not None:
                image = crop_image(image, crop, label=item["role"])
            np.save(scratch / f"{key[0]}-{key[1]}.npy", image, allow_pickle=False)
            available.add(key)
        if available != needed:
            raise ValueError("bag image decode did not supply all selected frames")
        with DirectLeRobotEpisode(
            identity=identity, task=manifest["task"], provenance=provenance
        ) as writer:
            for sample in samples:
                images = {
                    role: np.load(scratch / f"{role}-{index}.npy", allow_pickle=False)
                    for role, index in sample.images.items()
                }
                writer.add_frame(
                    build_lerobot_frame(
                        state=sample.state,
                        action=sample.action,
                        front_bgr=images["front"],
                        wrist_bgr=images["wrist"],
                        task=manifest["task"],
                        front_depth_mm=images.get("depth"),
                    )
                )
            writer.attach_timing([s.timing for s in samples], source)
            writer.commit()
    return result


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        type=Path,
        help="raw episode directory containing episode.json and bag/",
    )
    parser.add_argument("--config", type=Path, default=default_config_path(repo))
    parser.add_argument(
        "--experiment",
        help="optional new destination experiment for reproducible re-export",
    )
    args = parser.parse_args()
    config = load_teleop_config(args.config, repo_root=repo)
    result = export_bag(args.episode, config=config, experiment=args.experiment)
    print(
        json.dumps(
            {
                "dataset": str(result.root),
                "sampled_frames": result.sampled_frames,
                "stored_frames": result.stored_frames,
                "trimmed_frames": result.trimmed_frames,
            }
        )
    )


if __name__ == "__main__":
    main()
