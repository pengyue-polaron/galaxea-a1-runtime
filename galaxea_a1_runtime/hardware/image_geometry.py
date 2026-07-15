"""Pure image-region helpers shared by camera capture and previews."""

from __future__ import annotations

import cv2
import numpy as np

from galaxea_a1_runtime.configuration.image import ImageRoi


def crop_image(image: np.ndarray, roi: ImageRoi, *, label: str = "image") -> np.ndarray:
    """Return a contiguous copy of ``roi`` and fail if the live frame shape changed."""

    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        raise ValueError(f"{label} must be a 2D or 3D numpy image")
    image_height, image_width = image.shape[:2]
    roi.validate(
        image_width=image_width, image_height=image_height, label=f"{label} ROI"
    )
    return np.ascontiguousarray(
        image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    )


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
    roi.validate(
        image_width=image_width, image_height=image_height, label="preview ROI"
    )
    out = image.copy()
    thickness = max(2, round(min(image_width, image_height) / 160))
    p0 = (roi.x, roi.y)
    p1 = (roi.x + roi.width - 1, roi.y + roi.height - 1)
    cv2.rectangle(out, p0, p1, color_bgr, thickness=thickness, lineType=cv2.LINE_AA)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(image_width, image_height) / 900.0)
        text_thickness = max(1, thickness // 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, text_thickness
        )
        text_x = min(
            max(roi.x + thickness + 2, 0), max(0, image_width - text_width - 4)
        )
        if roi.y >= text_height + baseline + 8:
            text_y = roi.y - baseline - 4
        else:
            text_y = min(
                image_height - baseline - 4, roi.y + text_height + thickness + 4
            )
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
