"""WebSocket policy server for A1 single-arm inference.

Uses the pi05_a1_v21_lora_5k config defined in openpi/training/config.py.

Usage (via Justfile):
    just policy
"""

import dataclasses
import logging
import socket

import tyro

import numpy as np

import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
try:
    from a1.training.config import get_config as _get_config
except ImportError:
    import openpi.training.config as _config
    _get_config = _config.get_config
import openpi.transforms as _transforms
from openpi.serving import websocket_policy_server


# Remap client keys (observation/image, observation/wrist_image, observation/state)
# to the keys expected by A1JointInputs: data["image/cam_0_rgb"], data["image/cam_1_rgb"], data["state"].
_CLIENT_REPACK = _transforms.Group(
    inputs=[
        _transforms.RepackTransform({
            "image/cam_0_rgb": "observation/image",
            "image/cam_1_rgb": "observation/wrist_image",
            "state":           "observation/state",
            "prompt":          "prompt",
        }),
    ]
)


# ---- CLI args (mirrors serve_policy.py) ----

@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _get_config(args.policy.config),
                args.policy.dir,
                repack_transforms=_CLIENT_REPACK,
                default_prompt=args.default_prompt,
            )
        case Default():
            raise ValueError("Must specify policy:checkpoint --policy.config <name> --policy.dir <path>")


def main(args: Args) -> None:
    policy = create_policy(args)
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    logging.info("Serving on host=%s port=%d", hostname, args.port)

    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
