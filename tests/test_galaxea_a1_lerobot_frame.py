import numpy as np
import pytest

from galaxea_a1_runtime.collection.lerobot_frame import build_lerobot_frame


def test_frame_builder_preserves_canonical_vectors_and_converts_bgr_to_rgb():
    front = np.array([[[1, 2, 3]]], dtype=np.uint8)
    wrist = np.array([[[4, 5, 6]]], dtype=np.uint8)
    depth = np.array([[123]], dtype=np.uint16)

    frame = build_lerobot_frame(
        state=(*([0.1] * 13), 0.25),
        action=(*([0.2] * 6), 0.75),
        front_bgr=front,
        wrist_bgr=wrist,
        front_depth_mm=depth,
        task="pick cube",
    )

    assert frame["observation.state"].shape == (14,)
    assert frame["action"].shape == (7,)
    assert frame["observation.images.front"].tolist() == [[[3, 2, 1]]]
    assert frame["observation.images.wrist"].tolist() == [[[6, 5, 4]]]
    assert frame["observation.images.front_depth"].shape == (1, 1, 1)
    assert frame["task"] == "pick cube"


@pytest.mark.parametrize(
    ("state", "action", "match"),
    [
        ((*([0.0] * 13), float("nan")), (*([0.0] * 7),), "non-finite"),
        ((*([0.0] * 14),), (*([0.0] * 6), 1.1), "normalized"),
    ],
)
def test_frame_builder_rejects_invalid_canonical_vectors(state, action, match):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=match):
        build_lerobot_frame(
            state=state,
            action=action,
            front_bgr=image,
            wrist_bgr=image,
            task="pick cube",
        )
