import pytest

from galaxea_a1_runtime.policies.actions import normalize_action
from galaxea_a1_runtime.schema import ActionMode


def test_normalize_eef_delta_from_mapping_clamps_values_when_limits_are_explicit():
    action = normalize_action(
        {
            "delta_x": 0.2,
            "delta_y": -0.2,
            "delta_z": 0.01,
            "delta_roll": 0.5,
            "delta_pitch": -0.5,
            "delta_yaw": 0.01,
            "gripper": 2.0,
        },
        mode=ActionMode.EEF_DELTA,
        source="groot",
        max_translation=0.03,
        max_rotation=0.1,
    )

    assert action.source == "groot"
    assert action.values == pytest.approx((0.03, -0.03, 0.01, 0.1, -0.1, 0.01, 1.0))


def test_normalize_translation_action_from_sequence():
    action = normalize_action([0.01, 0.02, 0.03, -1.0], mode=ActionMode.EEF_TRANSLATION)

    assert action.as_dict() == {
        "delta_x": 0.01,
        "delta_y": 0.02,
        "delta_z": 0.03,
        "gripper": -1.0,
    }


def test_normalize_action_requires_mapping_keys():
    with pytest.raises(ValueError, match="delta_y"):
        normalize_action({"delta_x": 0.0}, mode=ActionMode.EEF_TRANSLATION)
