import numpy as np

from galaxea_a1_runtime.apps.cameras.diagnostics import (
    contact_sheet,
    depth_preview,
)


def test_depth_preview_preserves_shape_and_marks_missing_depth_black():
    depth = np.array([[0, 100], [200, 300]], dtype=np.uint16)

    preview = depth_preview(depth)

    assert preview.shape == (2, 2, 3)
    assert preview.dtype == np.uint8
    assert preview[0, 0].tolist() == [0, 0, 0]


def test_contact_sheet_accepts_different_camera_heights():
    short = np.zeros((20, 30, 3), dtype=np.uint8)
    tall = np.zeros((40, 30, 3), dtype=np.uint8)

    sheet = contact_sheet((("front", short), ("wrist", tall)))

    assert sheet.shape[0] == 74
    assert sheet.shape[2] == 3
    assert sheet.shape[1] > tall.shape[1]
