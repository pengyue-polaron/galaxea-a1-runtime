import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# After repack transform (matching dataset key to pipeline key)
# then is data transform (below)

def make_a1_example() -> dict:
    """Creates a dummy input example for the A1 policy."""
    # already after repack transform
    return {
        "image/cam_0_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "actions": np.zeros((8,), dtype=np.float32),
        "prompt": "do something",
        }
    


@dataclasses.dataclass(frozen=True)
class A1Inputs(transforms.DataTransformFn):
    """Inputs for the A1 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [8]
    - actions: [action_horizon, 8]
    """
    model_type: _model.ModelType
    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_0_rgb",)

    def __call__(self, data: dict) -> dict:
        # --- state ---
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"A1 state must be shape (8,), got {state.shape}")

        # --- images ---
        def convert_image(img):
            img = np.asarray(img)

            # float → uint8
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)

            # CHW → HWC
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        # Initialize dictionaries for images and masks
        images = {}
        image_masks = {}

        # reference to openpi/policies/droid_policy.py
        base_image = convert_image(data["image/cam_0_rgb"])
        
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Training-only
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class A1Outputs(transforms.DataTransformFn):
    """
    Outputs for the A1 policy.

    Policy outputs:
    - actions: [T, 8] or [8]
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)

        # Support both single-step and horizon output
        if actions.ndim == 2:
            actions = actions[:, :8]
        else:
            actions = actions[:8]

        return {"actions": actions}
