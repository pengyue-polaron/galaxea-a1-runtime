"""Pure checks for the released A1 Distill-WAM student and its components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from galaxea_a1_runtime.apps.lingbot.config_schema import LingBotConfig


def validate_student_config(backend: Any, engine: Any, contract: Any) -> None:
    if backend.backend_id != "diffusion2one":
        raise ValueError("Diffusion2One adapter requires its own backend identity")
    expected = {
        "video_inference_steps": 1,
        "action_inference_steps": 1,
        "guidance_scale": 1.0,
        "action_guidance_scale": 1.0,
        "snr_shift": 5.0,
        "action_snr_shift": 1.0,
        "height": 256,
        "width": 256,
        "attention_mode": "torch",
    }
    for key, value in expected.items():
        if getattr(engine, key) != value:
            raise ValueError(f"Diffusion2One A1 student requires {key}={value!r}")
    if (
        contract.vendor_config != "galaxea_deploy"
        or contract.pose_mode != "episode-relative"
        or contract.frame_chunk_size != 4
        or contract.action_per_frame != 4
        or contract.model_action_dim != 30
        or contract.base_model is None
    ):
        raise ValueError(
            "Diffusion2One A1 student tensor/pose/component contract mismatch"
        )
    base = contract.base_model
    if base.backend != "lingbot_foundation" or base.artifact_format != "diffusers":
        raise ValueError(
            "Diffusion2One requires a registered LingBot foundation artifact"
        )
    components = {item.path.parts[0] for item in base.manifest.files}
    if components != {"vae", "text_encoder", "tokenizer"}:
        raise ValueError(
            "Foundation manifest must contain exactly VAE, text encoder, tokenizer"
        )


def validate_checkpoint_metadata(config: LingBotConfig, artifact_root: Path) -> None:
    policy = config.policy_server
    root = artifact_root / policy.model_subdirectory
    stats = json.loads((root / "lingbot_norm_stat.json").read_text())
    if (
        stats.get("action_dim") != 8
        or stats.get("q01") != list(policy.q01_source)
        or stats.get("q99") != list(policy.q99_source)
    ):
        raise ValueError(
            "Diffusion2One checkpoint quantiles differ from the tracked model contract"
        )
    semantics = json.loads((root / "eef.json").read_text())
    action = semantics["action"]
    from galaxea_a1_runtime.schema import EEF_ACTION_NAMES

    if (
        action["names"] != list(EEF_ACTION_NAMES)
        or action["shape"] != [8]
        or action["rotation"] != "inverse(initial_quaternion) * target_quaternion, xyzw"
        or action["translation"] != "target_xyz_base_link - initial_xyz_base_link"
        or semantics["kinematics"]["base_link"] != "base_link"
    ):
        raise ValueError("Diffusion2One checkpoint EEF semantics mismatch")
    kinematics = semantics["kinematics"]
    if (
        kinematics["joint_names"] != list(config.system.joint_safety.names)
        or kinematics["tip_link"] != "arm_seg6"
        or kinematics["urdf_sha256"]
        != hashlib.sha256(config.system.eef_ik.urdf.read_bytes()).hexdigest()
    ):
        raise ValueError("Diffusion2One checkpoint kinematics mismatch")
    if (action["gripper_stroke_min_mm"], action["gripper_stroke_max_mm"]) != (
        config.system.gripper.stroke_min_mm,
        config.system.gripper.stroke_max_mm,
    ):
        raise ValueError("Diffusion2One checkpoint gripper physical mapping mismatch")
    from galaxea_a1_runtime.configuration.cameras import required_front_roi

    roi = required_front_roi(config.system.cameras)
    if (roi.x, roi.y, roi.width, roi.height) != (103, 0, 480, 480):
        raise ValueError(
            "Diffusion2One A1 requires the training front crop [103,0,480,480]"
        )
    wrist = config.system.cameras.wrist
    if (wrist.height, wrist.width) != (480, 640) or wrist.crop is not None:
        raise ValueError("Diffusion2One A1 requires uncropped 480x640 wrist RGB")
    if semantics["cameras"]["ordered_keys"] != [
        config.observations.front_key,
        config.observations.wrist_key,
    ]:
        raise ValueError("Diffusion2One checkpoint camera keys mismatch")
    if semantics["fps"] != config.execution.exec_rate:
        raise ValueError(
            "Diffusion2One execution cadence must match the checkpoint's 30 Hz"
        )
    transformer = json.loads((root / "transformer/config.json").read_text())
    if (
        transformer["action_dim"] != policy.model_action_dim
        or transformer["num_layers"] != 30
    ):
        raise ValueError("Diffusion2One transformer dimensions mismatch")
