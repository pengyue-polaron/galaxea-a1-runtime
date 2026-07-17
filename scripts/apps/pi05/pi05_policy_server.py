#!/usr/bin/env python3
# ruff: noqa: E402
"""Launch the pinned OpenPI pi0.5 policy with the tracked A1 contract."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.apps.pi05.config import load_pi05_config
from galaxea_a1_runtime.apps.pi05.protocol import server_metadata
from galaxea_a1_runtime.console import ArgumentParser, info


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    config = load_pi05_config(args.config, repo_root=args.repo_root)
    if not config.deployment_ready:
        raise RuntimeError("pi0.5 policy server refuses deployment.ready=false")

    os.environ["JAX_PLATFORMS"] = config.engine.jax_platform
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
        config.engine.xla_memory_fraction
    )
    checkout = config.backend.source.checkout
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout / "src"))
        sys.path.insert(0, str(checkout / "packages/openpi-client/src"))
    os.chdir(checkout)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    contract = config.model_contract
    train_config = training_config.get_config(contract.train_config)
    if (
        not train_config.model.pi05
        or train_config.model.action_horizon != contract.action_horizon
        or train_config.model.action_dim != contract.model_action_dim
    ):
        raise ValueError("pinned OpenPI train config does not match the model contract")
    policy = policy_config.create_trained_policy(
        train_config,
        config.model.artifact_root,
        default_prompt=config.task_catalog.default.prompt,
        sample_kwargs={"num_steps": config.engine.sampling_steps},
    )
    metadata = server_metadata(config)
    info(
        "OpenPI pi0.5 server: "
        f"model={config.model.model_id} step={config.model.checkpoint_step} "
        f"checkpoint={config.model.artifact_root} "
        f"action_shape=({contract.action_horizon},{contract.source_action_dim}) "
        f"contract={metadata['contract_sha256']}"
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=config.server.host,
        port=config.server.port,
        metadata=metadata,
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
