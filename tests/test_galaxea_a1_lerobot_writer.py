import pytest

from galaxea_a1_runtime.hardware.io import A1Observation
from galaxea_a1_runtime.policies.actions import normalize_action
from galaxea_a1_runtime.schema import ActionMode, CameraSpec, default_dataset_contract
from galaxea_a1_runtime.lerobot.writer import build_lerobot_frame


def test_build_lerobot_frame_from_runtime_observation_and_action():
    contract = default_dataset_contract(
        action_mode=ActionMode.EEF_TRANSLATION,
        cameras=(CameraSpec("front", height=480, width=640),),
    )
    observation = A1Observation(
        state=(0.0,) * 14,
        images={"front": "image"},
        timestamp=12.5,
    )
    action = normalize_action([0.01, 0.02, 0.03, 0.5], mode=ActionMode.EEF_TRANSLATION)

    frame = build_lerobot_frame(
        observation=observation,
        action=action,
        task="pick cube",
        contract=contract,
    )

    assert frame["observation.state"] == pytest.approx((0.0,) * 14)
    assert frame["observation.images.front"] == "image"
    assert frame["action"] == pytest.approx((0.01, 0.02, 0.03, 0.5))
    assert frame["task"] == "pick cube"
    assert frame["timestamp"] == 12.5


def test_build_lerobot_frame_requires_camera_image():
    contract = default_dataset_contract(
        cameras=(CameraSpec("front", height=480, width=640),)
    )
    action = normalize_action([0.0] * 7, mode=ActionMode.EEF_DELTA)

    with pytest.raises(ValueError, match="front"):
        build_lerobot_frame(
            observation=A1Observation(state=(0.0,) * 14, images={}),
            action=action,
            task="pick cube",
            contract=contract,
        )
