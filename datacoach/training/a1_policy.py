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
        "image/cam_1_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
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
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_0_rgb", "cam_1_rgb")

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

        if "cam_0" in data:
            base_image = convert_image(data["cam_0"])
            has_wrist_image = "cam_1" in data
            wrist_image = convert_image(data["cam_1"]) if has_wrist_image else np.zeros_like(base_image)
            
        else:
            base_image = convert_image(data["image/cam_0_rgb"])
            has_wrist_image = "image/cam_1_rgb" in data
            wrist_image = convert_image(data["image/cam_1_rgb"]) if has_wrist_image else np.zeros_like(base_image)
            
        match self.model_type:
            case (
                _model.ModelType.PI0
                | _model.ModelType.PI05
                | _model.ModelType.PI0_LTC
                | _model.ModelType.PI05_LTC
            ):
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_ if has_wrist_image else np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Pass through optional LTC/stateful inference fields if provided.
        for key in ("proprio_seq", "time_deltas", "ltc_dt", "ltc_h_prev"):
            if key in data:
                inputs[key] = np.asarray(data[key], dtype=np.float32)

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


@dataclasses.dataclass(frozen=True)
class A1JointInputs(transforms.DataTransformFn):
    """Inputs for the A1 joint-space policy."""

    model_type: _model.ModelType
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_0_rgb", "cam_1_rgb")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"A1 joint state must be shape (7,), got {state.shape}")

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        base_image = convert_image(data["image/cam_0_rgb"])
        wrist_image = convert_image(data["image/cam_1_rgb"])

        match self.model_type:
            case (
                _model.ModelType.PI0
                | _model.ModelType.PI05
                | _model.ModelType.PI0_LTC
                | _model.ModelType.PI05_LTC
            ):
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        for key in ("proprio_seq", "time_deltas", "ltc_dt", "ltc_h_prev"):
            if key in data:
                inputs[key] = np.asarray(data[key], dtype=np.float32)

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class A1JointOutputs(transforms.DataTransformFn):
    """Outputs for the A1 joint-space policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 2:
            actions = actions[:, :7]
        else:
            actions = actions[:7]
        return {"actions": actions}
