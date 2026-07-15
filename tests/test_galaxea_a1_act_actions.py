import numpy as np
import pytest

from galaxea_a1_runtime.apps.act.actions import ActActionValidator


def _validator(**overrides) -> ActActionValidator:
    values = {
        "joint_names": tuple(f"joint_{index}" for index in range(6)),
        "lower_limits": np.full(6, -1.0),
        "upper_limits": np.full(6, 1.0),
        "execute_steps": 2,
        "step_guard_enabled": True,
        "max_first_delta_rad": 0.2,
        "max_step_rad": 0.1,
    }
    values.update(overrides)
    return ActActionValidator(**values)


def test_act_action_validator_accepts_finite_bounded_continuous_chunk():
    chunk = np.array(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50],
        ]
    )

    steps = _validator().validate(chunk, (0.0,) * 6)

    np.testing.assert_allclose(steps, chunk)


def test_act_action_validator_rejects_joint_limit_and_step_violations():
    limit_chunk = np.array([[1.1, 0, 0, 0, 0, 0, 0.5]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="joint limits"):
        _validator().validate(limit_chunk, (0.0,) * 6)

    jump_chunk = np.array([[0.3, 0, 0, 0, 0, 0, 0.5]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="exceeds"):
        _validator().validate(jump_chunk, (0.0,) * 6)
