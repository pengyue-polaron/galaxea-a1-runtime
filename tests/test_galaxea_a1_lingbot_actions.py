import numpy as np
import pytest

from galaxea_a1_runtime.apps.eef_policy_actions import (
    EefActionTransformConfig,
    EefPolicyWorkspaceRejected,
    absolute_action_to_relative,
    gripper_norm_from_stroke,
    gripper_stroke_from_norm,
    normalize_condition_action,
    relative_action_to_absolute,
    validate_policy_action,
)


def action_config(**overrides) -> EefActionTransformConfig:
    values = {
        "xyz_min": (0.06, -0.27, 0.06),
        "xyz_max": (0.44, 0.14, 0.50),
        "min_quat_norm": 0.25,
        "gripper_stroke_min": 0.0,
        "gripper_stroke_max": 100.0,
        "gripper_normalized_endpoint_tolerance": 2e-6,
    }
    values.update(overrides)
    return EefActionTransformConfig(**values)


def test_condition_action_preserves_feedback_xyz_outside_workspace():
    cfg = action_config(xyz_min=(0.0, 0.0, 0.0), xyz_max=(1.0, 1.0, 1.0))

    action = normalize_condition_action([2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 2.0, 0.8], cfg)

    assert action[:3] == pytest.approx((2.0, -1.0, 0.5))
    assert action[3:7] == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert action[7] == pytest.approx(0.8)


def test_policy_action_rejects_workspace_and_gripper_violations():
    cfg = action_config(
        xyz_min=(0.0, 0.0, 0.0),
        xyz_max=(1.0, 1.0, 1.0),
    )

    with pytest.raises(EefPolicyWorkspaceRejected, match="outside.*workspace.*y,z"):
        validate_policy_action(
            [0.3, -0.2, 2.0, 0.0, 0.0, 0.0, 1.0, 0.5],
            cfg,
        )

    with pytest.raises(ValueError, match=r"Gripper value must be in \[0, 1\]"):
        validate_policy_action(
            [0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, -0.5],
            cfg,
        )


@pytest.mark.parametrize(
    ("raw_gripper", "expected"),
    [
        (1.0 + 1.5e-6, 1.0),
        (-1.5e-6, 0.0),
    ],
)
def test_policy_action_accepts_only_configured_gripper_endpoint_roundoff(
    raw_gripper, expected
):
    action = validate_policy_action(
        [0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, raw_gripper],
        action_config(
            xyz_min=(0.0, 0.0, 0.0),
            xyz_max=(1.0, 1.0, 1.0),
        ),
    )

    assert action[7] == expected


def test_policy_action_rejects_material_gripper_overshoot_with_full_precision():
    with pytest.raises(ValueError, match=r"got 1\.0001"):
        validate_policy_action(
            [0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0001],
            action_config(
                xyz_min=(0.0, 0.0, 0.0),
                xyz_max=(1.0, 1.0, 1.0),
            ),
        )


def test_policy_action_preserves_model_quaternion():
    cfg = action_config()

    action = validate_policy_action(
        [0.1, 0.1, 0.1, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5), 0.5],
        cfg,
    )

    assert action[3:7] == pytest.approx((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))


def test_gripper_mapping_is_continuous_across_shared_stroke():
    cfg = action_config()

    assert gripper_stroke_from_norm(0.25, cfg) == pytest.approx(25.0)
    assert gripper_stroke_from_norm(0.50, cfg) == pytest.approx(50.0)
    assert gripper_norm_from_stroke(50.0, cfg) == pytest.approx(0.50)
    assert gripper_norm_from_stroke(100.0, cfg) == pytest.approx(1.0)


def test_gripper_mapping_uses_configured_stroke():
    cfg = action_config(
        gripper_stroke_min=0.0,
        gripper_stroke_max=80.0,
    )

    action = validate_policy_action(
        [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0, 0.25],
        cfg,
    )

    assert action[7] == pytest.approx(0.25)
    assert gripper_stroke_from_norm(action[7], cfg) == pytest.approx(20.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gripper_stroke_from_norm(2.0, cfg)


def test_episode_relative_action_roundtrips_xyz_and_quaternion():
    origin = [0.2, -0.1, 0.3, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    relative = [0.1, 0.02, -0.03, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5), 0.25]

    absolute = relative_action_to_absolute(relative, origin, min_quat_norm=0.25)
    recovered = absolute_action_to_relative(absolute, origin, min_quat_norm=0.25)

    assert absolute[:3] == pytest.approx((0.3, -0.08, 0.27))
    assert abs(float(np.dot(recovered[3:7], relative[3:7]))) == pytest.approx(1.0)
    assert recovered[:3] == pytest.approx(relative[:3])
    assert recovered[7] == pytest.approx(0.25)
