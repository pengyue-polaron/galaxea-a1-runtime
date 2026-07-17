"""Hardware-free end-to-end smoke test for the LingBot policy service."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from galaxea_a1_runtime.apps.lingbot.client import LingBotClient
from galaxea_a1_runtime.apps.lingbot.config import (
    default_config_path,
    load_lingbot_config,
)
from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.console import ArgumentParser, step, success


def synthetic_observation(config: LingBotConfig) -> dict[str, np.ndarray]:
    """Build deterministic camera-shaped input without opening either camera."""
    front = required_front_roi(config.system.cameras)
    wrist = config.system.cameras.wrist
    return {
        config.observations.front_key: _gradient_rgb(front.height, front.width),
        config.observations.wrist_key: _gradient_rgb(wrist.height, wrist.width),
    }


def _gradient_rgb(height: int, width: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = x
    image[..., 1] = y
    image[..., 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    return image


def run_smoke(config: LingBotConfig) -> np.ndarray:
    """Exercise inference, temporal cache, and reinference without hardware I/O."""
    server = config.server
    prompt = config.task_catalog.default.prompt
    client = LingBotClient(
        server.host,
        server.port,
        connect_timeout_s=server.connect_timeout_s,
        close_timeout_s=server.close_timeout_s,
        expected_metadata=server_metadata(config),
    )
    try:
        step("Resetting the LingBot episode cache")
        client.reset(prompt)
        packet = {
            "obs": [synthetic_observation(config)],
            "prompt": prompt,
        }
        step("Running the first synthetic, hardware-free LingBot inference")
        started = time.monotonic()
        first_action = _validated_action(client.infer(packet), config)
        step("Synchronizing the temporal cache with the predicted action")
        cache_response = client.infer(
            {
                "obs": [
                    synthetic_observation(config)
                    for _ in range(config.execution.kv_observations_per_frame)
                ],
                "compute_kv_cache": True,
                "imagine": False,
                "state": first_action[:, :2].copy(),
            }
        )
        if set(cache_response) != {"server_timing"}:
            raise RuntimeError(
                f"Unexpected LingBot cache response keys: {sorted(cache_response)}"
            )
        step("Running the second inference from synchronized temporal context")
        action = _validated_action(client.infer(packet), config)
        elapsed = time.monotonic() - started
    finally:
        client.close()

    success(
        "LingBot offline closed-loop inference passed: "
        f"shape={action.shape} elapsed={elapsed:.3f}s "
        f"range=[{float(action.min()):.5f}, {float(action.max()):.5f}]"
    )
    return action


def _validated_action(response: dict, config: LingBotConfig) -> np.ndarray:
    if set(response) != {"action", "server_timing"}:
        raise RuntimeError(f"Unexpected LingBot response keys: {sorted(response)}")
    action = np.asarray(response["action"], dtype=np.float32)
    policy = config.policy_server
    expected_shape = (
        8,
        policy.frame_chunk_size,
        policy.action_per_frame,
    )
    if action.shape != expected_shape:
        raise RuntimeError(
            f"Expected LingBot action shape {expected_shape}, got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise RuntimeError("LingBot returned non-finite actions")
    return action


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config or default_config_path(repo_root)
    config = load_lingbot_config(config_path, repo_root=repo_root)
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("LingBot config does not belong to --repo-root")
    run_smoke(config)
    return 0
