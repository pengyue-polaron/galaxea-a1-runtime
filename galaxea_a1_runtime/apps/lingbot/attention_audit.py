"""Run a hardware-free LingBot attention audit against a tracked real episode."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from galaxea_a1_runtime.apps.lingbot.attention import (
    decoded_temporal_alignment,
    write_attention_audit,
)
from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata
from galaxea_a1_runtime.apps.lingbot.rollout import validated_action_tensor
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.console import ArgumentParser, step, success
from galaxea_a1_runtime.evaluation.eef_dataset import EefDataset
from galaxea_a1_runtime.evaluation.eef_policy_io import camera_packet
from galaxea_a1_runtime.evaluation.offline_config import (
    OfflineEvalConfig,
    load_offline_eval_config,
)


def run_attention_audit(
    config: LingBotConfig,
    evaluation: OfflineEvalConfig,
) -> Path:
    """Capture WAM paths after populating the real observation history."""

    policy = config.policy_server
    dataset = EefDataset(evaluation)
    episode_index = evaluation.teacher_forcing_episode_index
    episode = dataset.episode(episode_index)
    cache_count = config.execution.kv_observations_per_frame
    frames = dataset.frames(episode_index, list(range(cache_count + 1)))
    prompt = episode.task
    initial_observation = camera_packet(config, frames[0])
    cache_observations = [
        camera_packet(config, frames[index]) for index in range(1, cache_count + 1)
    ]
    raw_actual_observations = [initial_observation, *cache_observations]
    selected_observation = raw_actual_observations[-1]
    initial_packet = {"obs": [initial_observation], "prompt": prompt}
    attention_packet = {"obs": [selected_observation], "prompt": prompt}
    client = LingBotClient(
        config.server.host,
        config.server.port,
        connect_timeout_s=config.server.connect_timeout_s,
        close_timeout_s=config.server.close_timeout_s,
        expected_metadata=server_metadata(config),
    )
    expected_shape = (
        len(policy.action_channel_ids),
        policy.frame_chunk_size,
        policy.action_per_frame,
    )
    try:
        step("Resetting LingBot before the attention audit")
        client.reset(prompt)
        step(f"Running real episode {episode_index} frame 0 inference")
        first = client.infer(initial_packet)
        if set(first) != {"action", "server_timing"}:
            raise RuntimeError(f"Unexpected initial response keys: {sorted(first)}")
        first_action = validated_action_tensor(
            first["action"],
            expected_shape=expected_shape,
        )
        step(f"Populating the temporal cache with real frames 1..{cache_count}")
        cache = client.infer(
            {
                "obs": cache_observations,
                "compute_kv_cache": True,
                "imagine": False,
                "state": first_action[:, : config.execution.execute_frames].copy(),
            }
        )
        if set(cache) != {"server_timing"}:
            raise RuntimeError(f"Unexpected cache response keys: {sorted(cache)}")
        step("Capturing the video and action stages of the 30-layer WAM rollout")
        started = time.monotonic()
        response = client.infer({**attention_packet, "capture_attention": True})
        elapsed = time.monotonic() - started
    finally:
        client.close()

    if set(response) != {
        "action",
        "attention",
        "predicted_observations",
        "server_timing",
    }:
        raise RuntimeError(f"Unexpected attention response keys: {sorted(response)}")
    validated_action_tensor(response["action"], expected_shape=expected_shape)
    raw_attention = response["attention"]
    if (
        not isinstance(raw_attention, dict)
        or raw_attention.get("schema_version") != 3
        or raw_attention.get("kind") != "wam_multistage_cache_aware_attention_rollout"
    ):
        raise RuntimeError("LingBot attention response is invalid")
    attention = {
        **raw_attention,
        "input": {
            "dataset_repo_id": evaluation.dataset_repo_id,
            "episode_index": episode_index,
            "raw_actual_frame_indices": list(range(cache_count + 1)),
            "prediction_after_frame_index": cache_count,
            "task": episode.task,
        },
    }
    expected_grid = (
        policy.height // 16 // 2,
        policy.width // 16 // 2,
    )
    token_layout = attention.get("token_layout", {})
    if token_layout.get("predicted_future_frames") != policy.frame_chunk_size:
        raise RuntimeError("LingBot attention prediction horizon is invalid")
    temporal_scale = token_layout.get("vae_temporal_scale_factor")
    decoded_frames = token_layout.get("decoded_rgb_frames")
    anchor_indices = token_layout.get("decoded_rgb_anchor_indices")
    expected_temporal_alignment = (
        decoded_temporal_alignment(
            latent_frames=policy.frame_chunk_size,
            temporal_scale=temporal_scale,
        )
        if isinstance(temporal_scale, int) and temporal_scale > 0
        else None
    )
    if (
        expected_temporal_alignment is None
        or decoded_frames != expected_temporal_alignment.decoded_frame_count
        or anchor_indices != list(expected_temporal_alignment.anchor_indices)
    ):
        raise RuntimeError("LingBot decoded future temporal alignment is invalid")
    actual_anchor_indices = list(range(0, len(raw_actual_observations), temporal_scale))
    actual_observations = [
        raw_actual_observations[index] for index in actual_anchor_indices
    ]
    if token_layout.get("actual_history_frames") != len(actual_observations):
        raise RuntimeError(
            "LingBot attention history does not match the VAE-aligned real frames"
        )
    token_layout["actual_rgb_anchor_indices"] = actual_anchor_indices
    attention["input"]["actual_attention_anchor_frame_indices"] = actual_anchor_indices
    paths = attention.get("paths")
    if not isinstance(paths, dict) or set(paths) != {
        "action_to_predicted_future",
        "predicted_future_to_actual_history",
        "action_to_actual_history",
        "action_via_predicted_future_to_actual_history",
    }:
        raise RuntimeError("LingBot attention paths are incomplete")
    _validate_path_frames(
        paths["action_to_predicted_future"].get("frames"),
        expected_count=policy.frame_chunk_size,
        expected_grid=expected_grid,
    )
    _validate_path_frames(
        paths["action_to_actual_history"].get("frames"),
        expected_count=len(actual_observations),
        expected_grid=expected_grid,
    )
    _validate_path_frames(
        paths["action_via_predicted_future_to_actual_history"].get("frames"),
        expected_count=len(actual_observations),
        expected_grid=expected_grid,
    )
    source_frames = paths["predicted_future_to_actual_history"].get("source_frames")
    if (
        not isinstance(source_frames, list)
        or len(source_frames) != policy.frame_chunk_size
    ):
        raise RuntimeError("LingBot future-to-history source frames are invalid")
    for source_frame in source_frames:
        _validate_path_frames(
            source_frame.get("history_frames"),
            expected_count=len(actual_observations),
            expected_grid=expected_grid,
        )

    predicted_observations = response["predicted_observations"]
    if (
        not isinstance(predicted_observations, list)
        or len(predicted_observations) != policy.frame_chunk_size
    ):
        raise RuntimeError("LingBot decoded future observation list is invalid")
    for future_index, observation in enumerate(predicted_observations):
        if not isinstance(observation, dict):
            raise RuntimeError(
                f"LingBot decoded future frame {future_index} is not a camera packet"
            )
        for camera_key in (
            config.observations.front_key,
            config.observations.wrist_key,
        ):
            image = np.asarray(observation.get(camera_key))
            if image.shape != (policy.height, policy.width, 3):
                raise RuntimeError(
                    f"LingBot decoded future image is invalid: {image.shape}"
                )
            observation[camera_key] = image.astype(np.uint8, copy=False)
    output = write_attention_audit(
        policy.save_root / "attention-audits",
        attention=attention,
        actual_observations=actual_observations,
        predicted_observations=predicted_observations,
        front_key=config.observations.front_key,
        wrist_key=config.observations.wrist_key,
        prompt=prompt,
    )
    success(
        "LingBot attention audit saved: "
        f"{output} (layers={policy.attention_capture_layers}, "
        f"actual_anchors={actual_anchor_indices}, "
        f"future_frames={len(predicted_observations)}, "
        f"grid={expected_grid[0]}x{expected_grid[1]}, elapsed={elapsed:.3f}s)"
    )
    return output


def _validate_path_frames(
    frames: object,
    *,
    expected_count: int,
    expected_grid: tuple[int, int],
) -> None:
    if not isinstance(frames, list) or len(frames) != expected_count:
        raise RuntimeError(
            "LingBot attention path has the wrong frame count: "
            f"expected {expected_count}"
        )
    for frame in frames:
        if not isinstance(frame, dict):
            raise RuntimeError("LingBot attention frame is invalid")
        maps = frame.get("maps")
        if not isinstance(maps, dict) or set(maps) != {"front", "wrist"}:
            raise RuntimeError("LingBot attention frame camera maps are invalid")
        for camera, value in maps.items():
            values = np.asarray(value)
            if (
                values.shape != expected_grid
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise RuntimeError(
                    f"LingBot {camera} attention map is invalid: {values.shape}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=Path("configs/evaluation/fruit_placement_offline.toml"),
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config or default_config_path(repo_root)
    config = load_lingbot_config(
        config_path,
        repo_root=repo_root,
        model_selector=args.model,
    )
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("LingBot config does not belong to --repo-root")
    evaluation = load_offline_eval_config(
        args.evaluation_config,
        repo_root=repo_root,
    )
    run_attention_audit(config, evaluation)
    return 0
