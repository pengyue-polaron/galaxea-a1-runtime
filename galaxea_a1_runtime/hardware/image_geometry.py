"""Pure image-region helpers shared by camera capture and previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageRoi:
    """Pixel-aligned rectangular region in an image, expressed as ``x, y, w, h``."""

    x: int
    y: int
    width: int
    height: int

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def validate(self, *, image_width: int, image_height: int, label: str = "image ROI") -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"{label} source dimensions must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"{label} x/y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{label} width/height must be positive")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(
                f"{label} {self.xywh} exceeds source image "
                f"{image_width}x{image_height}"
            )


def parse_optional_image_roi(
    data: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    prefix: str = "crop",
    label: str = "image ROI",
    require_square: bool = False,
) -> ImageRoi | None:
    """Parse an optional flat TOML region such as ``crop_x`` and ``crop_width``."""

    if not bool(data.get(f"{prefix}_enabled", False)):
        return None
    roi = ImageRoi(
        x=int(data.get(f"{prefix}_x", 0)),
        y=int(data.get(f"{prefix}_y", 0)),
        width=int(data.get(f"{prefix}_width", image_width)),
        height=int(data.get(f"{prefix}_height", image_height)),
    )
    roi.validate(image_width=image_width, image_height=image_height, label=label)
    if require_square and roi.width != roi.height:
        raise ValueError(f"{label} must be square, got {roi.width}x{roi.height}")
    return roi


def crop_image(image: np.ndarray, roi: ImageRoi, *, label: str = "image") -> np.ndarray:
    """Return a contiguous copy of ``roi`` and fail if the live frame shape changed."""

    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        raise ValueError(f"{label} must be a 2D or 3D numpy image")
    image_height, image_width = image.shape[:2]
    roi.validate(image_width=image_width, image_height=image_height, label=f"{label} ROI")
    return np.ascontiguousarray(image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width])


def draw_image_roi(
    image: np.ndarray,
    roi: ImageRoi,
    *,
    label: str = "COLLECTION ROI",
    color_bgr: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draw a non-destructive ROI overlay for operator previews."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ROI overlay requires a BGR image with three channels")
    image_height, image_width = image.shape[:2]
    roi.validate(image_width=image_width, image_height=image_height, label="preview ROI")
    out = image.copy()
    thickness = max(2, round(min(image_width, image_height) / 160))
    p0 = (roi.x, roi.y)
    p1 = (roi.x + roi.width - 1, roi.y + roi.height - 1)
    cv2.rectangle(out, p0, p1, color_bgr, thickness=thickness, lineType=cv2.LINE_AA)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(image_width, image_height) / 900.0)
        text_thickness = max(1, thickness // 2)
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, text_thickness)
        text_x = min(max(roi.x + thickness + 2, 0), max(0, image_width - text_width - 4))
        if roi.y >= text_height + baseline + 8:
            text_y = roi.y - baseline - 4
        else:
            text_y = min(image_height - baseline - 4, roi.y + text_height + thickness + 4)
        cv2.rectangle(
            out,
            (text_x - 3, text_y - text_height - 3),
            (text_x + text_width + 3, text_y + baseline + 3),
            (0, 0, 0),
            thickness=-1,
        )
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            font,
            scale,
            color_bgr,
            text_thickness,
            lineType=cv2.LINE_AA,
        )
    return out
