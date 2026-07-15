from types import SimpleNamespace

import numpy as np
import pytest

from galaxea_a1_runtime.apps.lingbot.actions import LingBotActionConfig
from galaxea_a1_runtime.apps.lingbot.episode_state import LingBotEpisodeState


def _config() -> LingBotActionConfig:
    return LingBotActionConfig(
        xyz_min=(0.0, -1.0, 0.0),
        xyz_max=(1.0, 1.0, 1.0),
        min_quat_norm=0.25,
        orientation_mode="hold-current",
        gripper_stroke_min=0.0,
        gripper_stroke_max=100.0,
    )


def _pose(x: float, y: float, z: float):
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )


def _state(*, pose_mode: str = "episode-relative") -> LingBotEpisodeState:
    return LingBotEpisodeState(
        action_config=_config(),
        pose_mode=pose_mode,
        max_feedback_age_s=1.0,
        initial_action8=None,
        frame_chunk_size=4,
        action_per_frame=20,
    )


def test_episode_state_uses_fresh_pose_and_continuous_gripper_feedback():
    state = _state()
    state.pose_callback(_pose(0.2, -0.1, 0.3))
    state.gripper_callback(SimpleNamespace(position=[25.0]))

    action = state.current_absolute_action()

    assert action is not None
    assert action == pytest.approx((0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0, 0.25))


def test_episode_state_owns_relative_world_transform_and_condition_shape():
    state = _state()
    state.pose_callback(_pose(0.2, -0.1, 0.3))
    state.gripper_callback(SimpleNamespace(position=[50.0]))
    origin = state.ensure_episode_origin()
    assert origin is not None

    model = np.array([0.1, 0.02, -0.03, 0.0, 0.0, 0.0, 1.0, 0.25])
    absolute = state.model_to_absolute(model)

    assert absolute[:3] == pytest.approx((0.3, -0.08, 0.27))
    assert state.absolute_to_model(absolute) == pytest.approx(model)
    assert state.model_condition().shape == (8, 4, 20)


def test_episode_state_falls_back_to_explicit_initial_action():
    state = LingBotEpisodeState(
        action_config=_config(),
        pose_mode="absolute",
        max_feedback_age_s=1.0,
        initial_action8=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.75),
        frame_chunk_size=4,
        action_per_frame=20,
    )

    assert state.current_absolute_action() == pytest.approx(
        (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.75)
    )
