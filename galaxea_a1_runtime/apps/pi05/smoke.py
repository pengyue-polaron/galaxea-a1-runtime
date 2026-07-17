"""Hardware-free end-to-end smoke test for the OpenPI pi0.5 service."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from galaxea_a1_runtime.apps.pi05.client import Pi05Client
from galaxea_a1_runtime.apps.pi05.config import default_config_path, load_pi05_config
from galaxea_a1_runtime.apps.pi05.config_schema import Pi05Config
from galaxea_a1_runtime.apps.pi05.protocol import server_metadata
from galaxea_a1_runtime.configuration.base import discover_repo_root
from galaxea_a1_runtime.configuration.cameras import required_front_roi
from galaxea_a1_runtime.console import ArgumentParser, step, success


def synthetic_observation(config: Pi05Config) -> dict[str, object]:
    front = required_front_roi(config.system.cameras)
    wrist = config.system.cameras.wrist
    norm_stats = json.loads(config.model_contract.norm_stats_path.read_text())
    state = np.asarray(norm_stats["norm_stats"]["state"]["mean"], dtype=np.float32)
    if (
        state.shape != (config.model_contract.state_dim,)
        or not np.isfinite(state).all()
    ):
        raise ValueError("pi0.5 checkpoint state normalization mean is invalid")
    return {
        config.observations.front_key: _gradient_rgb(front.height, front.width),
        config.observations.wrist_key: _gradient_rgb(wrist.height, wrist.width),
        "observation/state": state,
        "prompt": config.task_catalog.default.prompt,
    }


def _gradient_rgb(height: int, width: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = x
    image[..., 1] = y
    image[..., 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    return image


def run_smoke(config: Pi05Config) -> np.ndarray:
    client = Pi05Client(
        config.server.host,
        config.server.port,
        connect_timeout_s=config.server.connect_timeout_s,
        close_timeout_s=config.server.close_timeout_s,
        expected_metadata=server_metadata(config),
    )
    try:
        step("Running synthetic, hardware-free OpenPI pi0.5 inference")
        started = time.monotonic()
        response = client.infer(synthetic_observation(config))
        elapsed = time.monotonic() - started
    finally:
        client.close()
    if "actions" not in response:
        raise RuntimeError(f"pi0.5 response is missing actions: {sorted(response)}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    expected_shape = (
        config.model_contract.action_horizon,
        config.model_contract.source_action_dim,
    )
    if actions.shape != expected_shape:
        raise RuntimeError(
            f"expected pi0.5 actions {expected_shape}, got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise RuntimeError("pi0.5 returned non-finite actions")
    success(
        "OpenPI pi0.5 offline inference passed: "
        f"shape={actions.shape} elapsed={elapsed:.3f}s "
        f"range=[{float(actions.min()):.5f}, {float(actions.max()):.5f}]"
    )
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config = load_pi05_config(
        args.config or default_config_path(repo_root), repo_root=repo_root
    )
    if discover_repo_root(config.path) != repo_root:
        raise ValueError("pi0.5 config does not belong to --repo-root")
    run_smoke(config)
    return 0
