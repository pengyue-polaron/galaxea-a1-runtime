import pytest

from galaxea_a1_runtime.gripper import (
    denormalize_stroke,
    normalize_source_position,
    normalize_stroke,
)


def test_continuous_gripper_mapping_round_trips():
    normalized = normalize_stroke(50.0, stroke_min_mm=0.0, stroke_max_mm=100.0)

    assert normalized == pytest.approx(0.5)
    assert denormalize_stroke(
        normalized, stroke_min_mm=0.0, stroke_max_mm=100.0
    ) == pytest.approx(50.0)


def test_continuous_gripper_mapping_rejects_values_outside_its_units():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        denormalize_stroke(1.5, stroke_min_mm=0.0, stroke_max_mm=100.0)
    with pytest.raises(ValueError, match="outside configured range"):
        normalize_stroke(-10.0, stroke_min_mm=0.0, stroke_max_mm=100.0)
    with pytest.raises(ValueError, match="finite"):
        denormalize_stroke(float("nan"), stroke_min_mm=0.0, stroke_max_mm=100.0)


def test_leader_gripper_range_is_explicit_and_not_silently_clipped():
    assert normalize_source_position(
        25.0, source_min=0.0, source_max=100.0, invert=False
    ) == pytest.approx(0.25)
    assert normalize_source_position(
        25.0, source_min=0.0, source_max=100.0, invert=True
    ) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="outside configured range"):
        normalize_source_position(101.0, source_min=0.0, source_max=100.0, invert=False)
    assert normalize_source_position(
        101.0,
        source_min=0.0,
        source_max=100.0,
        invert=False,
        saturate_out_of_range=True,
    ) == pytest.approx(1.0)
