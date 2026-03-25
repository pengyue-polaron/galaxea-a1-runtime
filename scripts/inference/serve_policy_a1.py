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
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.serving import websocket_policy_server


@dataclasses.dataclass(frozen=True)
class _HWCToCHW(_transforms.DataTransformFn):
    """Transpose images in data["images"] from (H, W, C) to (C, H, W) for AlohaInputs."""
    def __call__(self, data: dict) -> dict:
        data["images"] = {
            k: np.asarray(v).transpose(2, 0, 1) for k, v in data["images"].items()
        }
        return data


# Remap client keys (observation/image, observation/wrist_image, observation/state)
# to the keys expected by AlohaInputs: data["state"] and data["images"]["cam_*"] in (C,H,W).
_CLIENT_REPACK = _transforms.Group(
    inputs=[
        _transforms.RepackTransform({
            "images": {
                "cam_high":        "observation/image",
                "cam_left_wrist":  "observation/wrist_image",
                "cam_right_wrist": "observation/wrist_image",
            },
            "state":  "observation/state",
            "prompt": "prompt",
        }),
        _HWCToCHW(),
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
                _config.get_config(args.policy.config),
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
