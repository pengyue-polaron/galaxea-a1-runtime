from types import SimpleNamespace

import numpy as np
import pytest

from galaxea_a1_runtime.apps.eef_policy_actions import EefActionTransformConfig
from galaxea_a1_runtime.apps.eef_policy_state import EefPolicyState


def _config() -> EefActionTransformConfig:
    return EefActionTransformConfig(
        xyz_min=(0.0, -1.0, 0.0),
        xyz_max=(1.0, 1.0, 1.0),
        min_quat_norm=0.25,
        gripper_stroke_min=0.0,
        gripper_stroke_max=100.0,
        gripper_normalized_endpoint_tolerance=2e-6,
    )


def _pose(x: float, y: float, z: float):
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=y, z=z),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )


def _state(*, pose_mode: str = "episode-relative") -> EefPolicyState:
    return EefPolicyState(
        action_config=_config(),
        pose_mode=pose_mode,
        max_feedback_age_s=1.0,
    )


def test_episode_state_uses_fresh_pose_and_continuous_gripper_feedback():
    state = _state()
    state.pose_callback(_pose(0.2, -0.1, 0.3))
    state.gripper_callback(SimpleNamespace(position=[25.0]))

    action = state.current_absolute_action()

    assert action is not None
    assert action == pytest.approx((0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0, 0.25))


def test_episode_state_owns_relative_world_transform():
    state = _state()
    state.pose_callback(_pose(0.2, -0.1, 0.3))
    origin = state.ensure_episode_origin()
    assert origin is not None
    assert origin.shape == (7,)

    model = np.array([0.1, 0.02, -0.03, 0.0, 0.0, 0.0, 1.0, 0.25])
    absolute = state.model_to_absolute(model)

    assert absolute[:3] == pytest.approx((0.3, -0.08, 0.27))
    assert state.absolute_to_model(absolute) == pytest.approx(model)


def test_episode_state_requires_fresh_feedback_without_a_fallback():
    state = EefPolicyState(
        action_config=_config(),
        pose_mode="absolute",
        max_feedback_age_s=1.0,
    )

    assert state.current_absolute_action() is None
    assert state.ensure_episode_origin() is None


def test_invalid_gripper_feedback_clears_the_previous_sample():
    state = _state()
    state.gripper_callback(SimpleNamespace(position=[25.0]))
    assert state.gripper_is_fresh()

    state.gripper_callback(SimpleNamespace(position=[float("nan")]))

    assert not state.gripper_is_fresh()


@pytest.mark.parametrize(
    "pose",
    [
        _pose(float("nan"), 0.0, 0.0),
        SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.2, y=0.0, z=0.3),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
            )
        ),
    ],
)
def test_invalid_pose_feedback_clears_the_previous_sample(pose):
    state = _state()
    state.pose_callback(_pose(0.2, 0.0, 0.3))
    assert state.pose_is_fresh()

    state.pose_callback(pose)

    assert not state.pose_is_fresh()
