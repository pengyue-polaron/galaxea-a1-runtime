import dataclasses
import enum
import importlib.util
import logging
import socket
from pathlib import Path
import sys

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config

# Ensure local DataCoach package is preferred over environment-installed copy.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datacoach.inference import ZMQ_policy_server

from datacoach.training import config as _config
from datacoach.constants import ZMQ_CAM_PORT, ZMQ_POLICY_ACTION_PORT, ZMQ_STATE_PORT

_EXTERNAL_TRAINING_CONFIG = None


def _load_external_training_config():
    """Load fallback training config module from jolia repo if available."""
    global _EXTERNAL_TRAINING_CONFIG
    if _EXTERNAL_TRAINING_CONFIG is not None:
        return _EXTERNAL_TRAINING_CONFIG

    fallback_root = Path("/home/jolia/DataCoach")
    config_file = fallback_root / "datacoach" / "training" / "config.py"
    if not config_file.exists():
        _EXTERNAL_TRAINING_CONFIG = False
        return None

    spec = importlib.util.spec_from_file_location("jolia_training_config", str(config_file))
    if spec is None or spec.loader is None:
        _EXTERNAL_TRAINING_CONFIG = False
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EXTERNAL_TRAINING_CONFIG = module
    return module


def _resolve_train_config(config_name: str):
    """Resolve training config, preferring local repo then jolia fallback."""
    try:
        return _config.get_config(config_name)
    except Exception as local_exc:
        fallback_module = _load_external_training_config()
        if fallback_module is not None:
            try:
                logging.warning(
                    "Local config '%s' unavailable, using fallback config from /home/jolia/DataCoach",
                    config_name,
                )
                return fallback_module.get_config(config_name)
            except Exception as fallback_exc:
                raise ValueError(
                    f"Config '{config_name}' is unavailable in both local and fallback configs. "
                    f"local_error={local_exc}; fallback_error={fallback_exc}"
                ) from fallback_exc
        raise

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _resolve_train_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            # _config.get_config(checkpoint.config) return a training config
            train_config = _resolve_train_config(args.policy.config)
            data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
            return _policy_config.create_trained_policy(
                train_config=train_config, repack_transforms=data_config.repack_transforms, checkpoint_dir=args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _extract_norm_stats(policy):
    """Extract norm_stats and use_quantile_norm from the Normalize transform inside the policy."""
    from openpi import transforms as _transforms
    composite = getattr(policy, "_input_transform", None)
    transforms_list = getattr(composite, "transforms", [])
    for t in transforms_list:
        if isinstance(t, _transforms.Normalize):
            return t.norm_stats, t.use_quantiles
    return None, False


def main(args: Args) -> None:

    # 1. Initialize policy
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Extract norm_stats so the ZMQ server can normalize proprio_seq history
    # to match the normalized state sequences used during training.
    norm_stats, use_quantile_norm = _extract_norm_stats(policy)
    if norm_stats is not None:
        logging.info("Extracted norm_stats from policy for proprio_seq normalization.")
    else:
        logging.warning("Could not extract norm_stats from policy — proprio_seq will be unnormalized.")

    # 2. create policy server (recieve ZMQ obs data and publish ZMQ action data)

    server = ZMQ_policy_server.ZMQPolicyServer(
        policy=policy,
        host="127.0.0.1",
        state_port=ZMQ_STATE_PORT,
        action_port=ZMQ_POLICY_ACTION_PORT,
        camera_port=ZMQ_CAM_PORT,
        metadata=policy_metadata,
        norm_stats=norm_stats,
        use_quantile_norm=use_quantile_norm,
        deterministic_inference=True,
        prompt="pick_twice",
    )
    server.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
