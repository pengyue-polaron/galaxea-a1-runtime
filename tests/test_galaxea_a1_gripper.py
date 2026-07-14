import pytest

from galaxea_a1_runtime.gripper import denormalize_stroke, normalize_stroke


def test_continuous_gripper_mapping_round_trips():
    normalized = normalize_stroke(50.0, stroke_min_mm=0.0, stroke_max_mm=200.0)

    assert normalized == pytest.approx(0.25)
    assert denormalize_stroke(normalized, stroke_min_mm=0.0, stroke_max_mm=200.0) == pytest.approx(50.0)


def test_continuous_gripper_output_clips_but_feedback_rejects_bad_units():
    assert denormalize_stroke(1.5, stroke_min_mm=0.0, stroke_max_mm=200.0) == 200.0
    with pytest.raises(ValueError, match="outside configured range"):
        normalize_stroke(-10.0, stroke_min_mm=0.0, stroke_max_mm=200.0)
    with pytest.raises(ValueError, match="finite"):
        denormalize_stroke(float("nan"), stroke_min_mm=0.0, stroke_max_mm=200.0)
