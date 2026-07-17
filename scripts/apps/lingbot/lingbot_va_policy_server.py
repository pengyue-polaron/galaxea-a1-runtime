#!/usr/bin/env python3
# ruff: noqa: E402
"""Launch the external LingBot-VA server with the tracked A1 deployment contract."""

from __future__ import annotations

import copy
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config
from galaxea_a1_runtime.console import ArgumentParser, info
from galaxea_a1_runtime.apps.lingbot.protocol import server_metadata


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    config = load_lingbot_config(args.config, repo_root=args.repo_root)
    policy = config.policy_server
    checkout = policy.backend.source.checkout
    model_root = policy.model.artifact_root
    if not policy.deployment_ready:
        raise RuntimeError("LingBot policy server refuses deployment_ready=false")
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    os.chdir(checkout)

    import numpy as np
    import torch
    import wan_va.wan_va_server as server_module

    if policy.vendor_config not in server_module.VA_CONFIGS:
        raise RuntimeError(
            "External LingBot checkout does not provide the configured base: "
            f"{policy.vendor_config}"
        )

    job = copy.deepcopy(server_module.VA_CONFIGS[policy.vendor_config])
    job.__name__ = "Config: Galaxea A1 deployment policy server"
    job.wan22_pretrained_model_name_or_path = str(model_root)
    job.infer_mode = "server"
    job.host = config.server.host
    job.port = config.server.port
    job.height = policy.height
    job.width = policy.width
    job.frame_chunk_size = policy.frame_chunk_size
    job.action_per_frame = policy.action_per_frame
    job.attn_window = policy.attention_window
    job.action_dim = policy.model_action_dim
    job.enable_offload = policy.enable_offload
    job.obs_cam_keys = [config.observations.front_key, config.observations.wrist_key]
    job.guidance_scale = policy.guidance_scale
    job.action_guidance_scale = policy.action_guidance_scale
    job.num_inference_steps = policy.video_inference_steps
    job.action_num_inference_steps = policy.action_inference_steps
    job.snr_shift = policy.snr_shift
    job.action_snr_shift = policy.action_snr_shift
    job.used_action_channel_ids = list(policy.action_channel_ids)

    inverse = [len(job.used_action_channel_ids)] * job.action_dim
    q01 = [0.0] * job.action_dim
    q99 = [0.0] * job.action_dim
    for source_id, model_id in enumerate(job.used_action_channel_ids):
        inverse[model_id] = source_id
        q01[model_id] = policy.q01_source[source_id]
        q99[model_id] = policy.q99_source[source_id]
    job.inverse_used_action_channel_ids = inverse
    job.action_norm_method = "quantiles"
    job.norm_stat = {"q01": q01, "q99": q99}

    original_load_text_encoder = server_module.load_text_encoder
    original_load_transformer = server_module.load_transformer

    def load_text_encoder_on_tracked_device(path, torch_dtype, torch_device):
        del torch_device
        return original_load_text_encoder(
            path,
            torch_dtype=torch_dtype,
            torch_device=policy.text_encoder_device,
        )

    server_module.load_text_encoder = load_text_encoder_on_tracked_device

    def load_transformer_with_tracked_attention(
        path, torch_dtype, torch_device, attn_mode
    ):
        del attn_mode
        return original_load_transformer(
            path,
            torch_dtype=torch_dtype,
            torch_device=torch_device,
            attn_mode=policy.attention_mode,
        )

    server_module.load_transformer = load_transformer_with_tracked_attention

    # The external server initializes a process group even for world size one,
    # then unconditionally applies FSDP. Single-device inference needs neither
    # parameter sharding nor FSDP's uniform-original-dtype restriction. The
    # loader has already placed the complete model on the configured CUDA device.
    def keep_complete_model_on_single_gpu(model):
        return model

    server_module.shard_model = keep_complete_model_on_single_gpu

    def init_single_gpu_process_group(world_size, local_rank, rank):
        if world_size != policy.world_size:
            raise RuntimeError(
                f"Expected world size {policy.world_size}, got {world_size}"
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )

    server_module.init_distributed = init_single_gpu_process_group

    original_preprocess_action = server_module.VA_Server.preprocess_action

    def preprocess_protocol_action(self, action):
        # MessagePack reconstructs arrays over an immutable bytes buffer. Torch
        # warns on from_numpy even though this path does not mutate its input.
        return original_preprocess_action(self, np.array(action, copy=True))

    server_module.VA_Server.preprocess_action = preprocess_protocol_action

    original_policy_server = server_module.run_async_server_mode.__globals__[
        "WebsocketPolicyServer"
    ]
    metadata = server_metadata(config)

    def policy_server_with_metadata(policy_instance, host, port):
        return original_policy_server(
            policy_instance,
            host=host,
            port=port,
            metadata=metadata,
        )

    server_module.run_async_server_mode.__globals__["WebsocketPolicyServer"] = (
        policy_server_with_metadata
    )

    torch.manual_seed(policy.seed)
    torch.cuda.manual_seed_all(policy.seed)
    np.random.seed(policy.seed)
    server_module.VA_CONFIGS["a1_deployment"] = job

    info(f"LingBot checkout: {checkout}")
    info(f"LingBot model root: {model_root}")
    info(
        "LingBot server: "
        f"image={policy.height}x{policy.width} frame_chunk={policy.frame_chunk_size} "
        f"actions_per_frame={policy.action_per_frame} cameras={job.obs_cam_keys} "
        f"text_encoder={policy.text_encoder_device} "
        f"attention={policy.attention_mode} offload={policy.enable_offload} "
        f"world_size={policy.world_size} fsdp=False "
        f"seed={policy.seed} contract={metadata['contract_sha256']}"
    )
    server_module.init_logger()
    server_module.run(
        Namespace(
            config_name="a1_deployment",
            port=config.server.port,
            save_root=str(policy.save_root),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
