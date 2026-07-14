#!/usr/bin/env python3
# ruff: noqa: E402
"""Launch the external LingBot-VA server with the tracked A1 deployment contract."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.apps.lingbot.config import load_lingbot_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    config = load_lingbot_config(args.config, repo_root=args.repo_root)
    policy = config.policy_server
    if not policy.deployment_ready:
        raise RuntimeError("LingBot policy server refuses deployment_ready=false")
    if str(policy.checkout) not in sys.path:
        sys.path.insert(0, str(policy.checkout))
    os.chdir(policy.checkout)

    import numpy as np
    import torch
    import wan_va.wan_va_server as server_module

    if "galaxea_a1" not in server_module.VA_CONFIGS:
        raise RuntimeError("External LingBot checkout does not provide the galaxea_a1 base config")

    job = copy.deepcopy(server_module.VA_CONFIGS["galaxea_a1"])
    job.__name__ = "Config: Galaxea A1 deployment policy server"
    job.wan22_pretrained_model_name_or_path = str(policy.model_root)
    job.infer_mode = "server"
    job.host = config.server.host
    job.port = config.server.port
    job.height = policy.height
    job.width = policy.width
    job.frame_chunk_size = policy.frame_chunk_size
    job.action_per_frame = policy.action_per_frame
    job.attn_window = policy.attention_window
    job.action_dim = 30
    job.obs_cam_keys = [config.observations.front_key, config.observations.wrist_key]
    job.guidance_scale = policy.guidance_scale
    job.action_guidance_scale = policy.action_guidance_scale
    job.num_inference_steps = policy.video_inference_steps
    job.action_num_inference_steps = policy.action_inference_steps
    job.snr_shift = policy.snr_shift
    job.action_snr_shift = policy.action_snr_shift
    job.used_action_channel_ids = list(policy.used_action_channel_ids)

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

    def load_text_encoder_on_tracked_device(path, torch_dtype, torch_device):
        del torch_device
        return original_load_text_encoder(
            path,
            torch_dtype=torch_dtype,
            torch_device=policy.text_encoder_device,
        )

    server_module.load_text_encoder = load_text_encoder_on_tracked_device

    torch.manual_seed(policy.seed)
    torch.cuda.manual_seed_all(policy.seed)
    np.random.seed(policy.seed)
    server_module.VA_CONFIGS["a1_deployment"] = job

    print(f"[LingBot server] checkout={policy.checkout}", flush=True)
    print(f"[LingBot server] model_root={policy.model_root}", flush=True)
    print(
        "[LingBot server] "
        f"image={policy.height}x{policy.width} frame_chunk={policy.frame_chunk_size} "
        f"actions_per_frame={policy.action_per_frame} cameras={job.obs_cam_keys} "
        f"text_encoder={policy.text_encoder_device} seed={policy.seed}",
        flush=True,
    )
    server_module.init_logger()
    server_module.run(
        argparse.Namespace(
            config_name="a1_deployment",
            port=config.server.port,
            save_root=str(policy.save_root),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
