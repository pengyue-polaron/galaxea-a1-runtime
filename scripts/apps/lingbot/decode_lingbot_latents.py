#!/usr/bin/env python3
"""Decode LingBot-VA saved latent tensors into a review video.

The online websocket server saves `latents_*.pt` for each inference, but its
server-mode response only returns actions. This helper decodes those saved
latents with the same Wan VAE so we can inspect the imagined future without
touching the robot execution path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw


LINGBOT_ROOT = Path("/home/pengyue/lingbot-va")
WAN_VA_ROOT = LINGBOT_ROOT / "wan_va"
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from configs import VA_CONFIGS  # noqa: E402
from modules.utils import load_vae  # noqa: E402


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _save_contact_sheet(frames: list[np.ndarray], path: Path, cols: int = 4) -> None:
    thumbs = [_to_uint8(frame) for frame in frames]
    if not thumbs:
        return
    h, w = thumbs[0].shape[:2]
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(thumbs):
        img = Image.fromarray(frame).convert("RGB")
        x = (i % cols) * w
        y = (i // cols) * h
        sheet.paste(img, (x, y))
        draw.rectangle((x, y, x + 48, y + 18), fill=(0, 0, 0))
        draw.text((x + 5, y + 3), f"f{i}", fill=(255, 255, 255))
    sheet.save(path)


def decode_latents(
    latents_path: Path,
    output_path: Path,
    *,
    config_name: str,
    device: str,
    dtype_name: str,
    fps: int,
    contact_sheet_path: Path | None,
) -> None:
    config = VA_CONFIGS[config_name]
    torch_dtype = getattr(torch, dtype_name)
    vae_path = Path(config.wan22_pretrained_model_name_or_path) / "vae"

    print(f"[decode] loading latents: {latents_path}")
    latents = torch.load(latents_path, map_location="cpu")
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Expected tensor in {latents_path}, got {type(latents)!r}")

    print(f"[decode] loading VAE: {vae_path} on {device} dtype={dtype_name}")
    vae = load_vae(str(vae_path), torch_dtype=torch_dtype, torch_device=device)
    vae.eval()

    with torch.no_grad():
        latents = latents.to(device=device, dtype=vae.dtype)
        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(
            1, vae.config.z_dim, 1, 1, 1
        )
        latents_std = (
            1.0
            / torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(
                1, vae.config.z_dim, 1, 1, 1
            )
        )
        latents = latents / latents_std + latents_mean
        decoded = vae.decode(latents, return_dict=False)[0]
        video = VideoProcessor(vae_scale_factor=1).postprocess_video(decoded, output_type="np")[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video, str(output_path), fps=fps)
    print(f"[decode] wrote video: {output_path}")

    if contact_sheet_path is not None:
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        _save_contact_sheet(list(video), contact_sheet_path)
        print(f"[decode] wrote contact sheet: {contact_sheet_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("latents_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--config-name", default="galaxea_a1")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decode_latents(
        args.latents_path,
        args.output,
        config_name=args.config_name,
        device=args.device,
        dtype_name=args.dtype,
        fps=args.fps,
        contact_sheet_path=args.contact_sheet,
    )


if __name__ == "__main__":
    main()
