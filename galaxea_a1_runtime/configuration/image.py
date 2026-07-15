"""Pure image ROI schema and configuration parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from galaxea_a1_runtime.configuration.base import boolean, integer


@dataclass(frozen=True)
class ImageRoi:
    x: int
    y: int
    width: int
    height: int

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def validate(
        self, *, image_width: int, image_height: int, label: str = "image ROI"
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"{label} source dimensions must be positive")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"{label} x/y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{label} width/height must be positive")
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(
                f"{label} {self.xywh} exceeds source image {image_width}x{image_height}"
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
    if not boolean(data, f"{prefix}_enabled"):
        return None
    roi = ImageRoi(
        x=integer(data, f"{prefix}_x"),
        y=integer(data, f"{prefix}_y"),
        width=integer(data, f"{prefix}_width"),
        height=integer(data, f"{prefix}_height"),
    )
    roi.validate(image_width=image_width, image_height=image_height, label=label)
    if require_square and roi.width != roi.height:
        raise ValueError(f"{label} must be square, got {roi.width}x{roi.height}")
    return roi
