"""ACT checkpoint loading and pure observation-to-action inference."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors


def log(message: str) -> None:
    print(message, flush=True)


class ActPolicyRunner:
    def __init__(self, args: argparse.Namespace):
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"ACT checkpoint not found: {checkpoint}")
        log(f"[ACT] Loading checkpoint: {checkpoint}")
        cfg = PreTrainedConfig.from_pretrained(
            checkpoint,
            cli_overrides=[f"--device={args.device}"],
        )
        if args.disable_backbone_download and hasattr(
            cfg, "pretrained_backbone_weights"
        ):
            cfg.pretrained_backbone_weights = None
        self.front_width, self.front_height = policy_image_hw(
            cfg, "observation.images.front"
        )
        self.wrist_width, self.wrist_height = policy_image_hw(
            cfg, "observation.images.wrist"
        )
        configured_front = (args.cam0_crop_width, args.cam0_crop_height)
        configured_wrist = (args.cam_width, args.cam_height)
        if (self.front_width, self.front_height) != configured_front:
            raise RuntimeError(
                "ACT checkpoint front image contract is "
                f"{self.front_width}x{self.front_height}, but configured AgentView crop is "
                f"{configured_front[0]}x{configured_front[1]}; retrain/register a matching checkpoint"
            )
        if (self.wrist_width, self.wrist_height) != configured_wrist:
            raise RuntimeError(
                "ACT checkpoint wrist image contract is "
                f"{self.wrist_width}x{self.wrist_height}, but configured Wrist source is "
                f"{configured_wrist[0]}x{configured_wrist[1]}; retrain/register a matching checkpoint"
            )
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(
            checkpoint, config=cfg, local_files_only=True
        )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.device = str(cfg.device)
        self.use_amp = bool(getattr(cfg, "use_amp", False))
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        log(
            "[ACT] Ready: "
            f"device={self.device} chunk={getattr(cfg, 'chunk_size', '?')} "
            f"n_action_steps={getattr(cfg, 'n_action_steps', '?')}"
        )

    def predict_chunk(
        self,
        *,
        front_bgr: np.ndarray,
        wrist_bgr: np.ndarray,
        state7: Sequence[float],
    ) -> np.ndarray:
        obs = {
            "observation.images.front": bgr_to_chw_tensor(
                front_bgr, width=self.front_width, height=self.front_height
            ),
            "observation.images.wrist": bgr_to_chw_tensor(
                wrist_bgr, width=self.wrist_width, height=self.wrist_height
            ),
            "observation.state": torch.tensor(
                tuple(float(v) for v in state7), dtype=torch.float32
            ),
        }
        batch = self.preprocessor(obs)
        amp_context = (
            torch.autocast(device_type="cuda")
            if self.use_amp and self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[-1] != 7:
            raise RuntimeError(
                f"ACT returned unexpected action shape: {tuple(chunk.shape)}"
            )
        return chunk[0].detach().cpu().numpy().astype(np.float64, copy=False)


def bgr_to_chw_tensor(image: np.ndarray, *, width: int, height: int) -> torch.Tensor:
    if image.shape[:2] != (height, width):
        raise RuntimeError(
            f"camera image shape {image.shape[:2]} does not match expected {(height, width)}"
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)


def policy_image_hw(config: Any, key: str) -> tuple[int, int]:
    features = getattr(config, "input_features", None)
    feature = features.get(key) if isinstance(features, dict) else None
    shape = tuple(int(value) for value in getattr(feature, "shape", ()))
    if len(shape) != 3 or shape[0] != 3 or shape[1] <= 0 or shape[2] <= 0:
        raise RuntimeError(
            f"ACT checkpoint is missing a valid CHW input feature for {key!r}: {shape!r}"
        )
    return shape[2], shape[1]
