"""Wrapper around eric's WebSocket policy server that registers pi05_a1_single_arm.

The checkpoint at /home/eric/4999 was trained with Pi05 (pi05=True), which adds
state_proj + action_time_mlp_in/out vs vanilla Pi0. Eric's config.py only has
pi0_a1_single_arm (Pi0), so we inject the Pi05 variant before starting the server.

Usage (via Justfile):
    just policy
"""

import dataclasses
import logging
import socket

import tyro

import openpi.models.pi0_config as pi0_config
import openpi.policies.libero_policy as libero_policy
import openpi.policies.policy as _policy
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.serving import websocket_policy_server

# ---- inject pi05_a1_single_arm into eric's config registry ----
_PI05_A1 = _config.TrainConfig(
    name="pi05_a1_single_arm",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
    data=_config.SimpleDataConfig(
        repo_id="a1_v21_old",
        base_config=_config.DataConfig(
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "observation.images.cam0",
                            "observation/wrist_image": "observation.images.cam1",
                            "observation/state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            action_sequence_keys=("action",),
            prompt_from_task=True,
        ),
        data_transforms=lambda model: _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=32,
    num_train_steps=30_000,
)
_config._CONFIGS_DICT["pi05_a1_single_arm"] = _PI05_A1  # noqa: SLF001


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
