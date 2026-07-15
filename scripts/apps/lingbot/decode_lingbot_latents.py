#!/usr/bin/env python3
"""Decode LingBot-VA saved latent tensors into a review video.

The online websocket server saves `latents_*.pt` for each inference, but its
server-mode response only returns actions. This helper decodes those saved
latents with the same Wan VAE so we can inspect the imagined future without
touching the robot execution path.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from galaxea_a1_runtime.console import ArgumentParser, step, success


REPO_ROOT = Path(__file__).resolve().parents[3]


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
    lingbot_root: Path,
    config_name: str,
    device: str,
    dtype_name: str,
    fps: int,
    contact_sheet_path: Path | None,
) -> None:
    try:
        from diffusers.video_processor import VideoProcessor
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise RuntimeError(
            "Latent decoding requires the LingBot environment with diffusers. "
            "Run this script with external/lingbot-va/.env312/bin/python."
        ) from exc
    va_configs, load_vae = _load_lingbot_va(lingbot_root)
    config = va_configs[config_name]
    torch_dtype = getattr(torch, dtype_name)
    vae_path = Path(config.wan22_pretrained_model_name_or_path) / "vae"

    step(f"Loading LingBot latents: {latents_path}")
    latents = torch.load(latents_path, map_location="cpu")
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Expected tensor in {latents_path}, got {type(latents)!r}")

    step(f"Loading LingBot VAE: {vae_path} on {device} dtype={dtype_name}")
    vae = load_vae(str(vae_path), torch_dtype=torch_dtype, torch_device=device)
    vae.eval()

    with torch.no_grad():
        latents = latents.to(device=device, dtype=vae.dtype)
        latents_mean = torch.tensor(
            vae.config.latents_mean, device=device, dtype=vae.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            vae.config.latents_std, device=device, dtype=vae.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents = latents / latents_std + latents_mean
        decoded = vae.decode(latents, return_dict=False)[0]
        video = VideoProcessor(vae_scale_factor=1).postprocess_video(
            decoded, output_type="np"
        )[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video, str(output_path), fps=fps)
    success(f"Wrote decoded video: {output_path}")

    if contact_sheet_path is not None:
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        _save_contact_sheet(list(video), contact_sheet_path)
        success(f"Wrote contact sheet: {contact_sheet_path}")


def _load_lingbot_va(lingbot_root: Path) -> tuple[dict[str, Any], Any]:
    wan_va_root = lingbot_root.expanduser().resolve() / "wan_va"
    if not wan_va_root.is_dir():
        raise FileNotFoundError(
            f"LingBot wan_va package not found at {wan_va_root}. "
            "Pass --lingbot-root or set LINGBOT_VA_ROOT."
        )
    if str(wan_va_root) not in sys.path:
        sys.path.insert(0, str(wan_va_root))
    from configs import VA_CONFIGS
    from modules.utils import load_vae

    return VA_CONFIGS, load_vae


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("latents_path", type=Path)
    parser.add_argument(
        "--lingbot-root",
        type=Path,
        default=Path(
            os.environ.get("LINGBOT_VA_ROOT", REPO_ROOT / "external/lingbot-va")
        ),
        help="Path to the external LingBot checkout containing wan_va/.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--config-name", default="galaxea_a1")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decode_latents(
        args.latents_path,
        args.output,
        lingbot_root=args.lingbot_root,
        config_name=args.config_name,
        device=args.device,
        dtype_name=args.dtype,
        fps=args.fps,
        contact_sheet_path=args.contact_sheet,
    )


if __name__ == "__main__":
    main()
