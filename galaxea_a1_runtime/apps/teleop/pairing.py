"""SO-Leader to A1 relative-anchor processor for LeRobot control loops."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.processor.pipeline import ProcessorStepRegistry, RobotActionProcessorStep

try:  # LeRobot 0.6.0 source layout
    from lerobot.types import RobotAction, TransitionKey
except ModuleNotFoundError:  # LeRobot 0.6.1 wheel layout
    from lerobot.lerobot_types import RobotAction, TransitionKey

DEFAULT_LEADER_JOINT_KEYS = tuple(f"joint{index}.pos" for index in range(6))
DEFAULT_A1_JOINT_KEYS = tuple(f"joint_{index}_rad" for index in range(1, 7))
DEFAULT_GRIPPER_INPUT_KEY = "gripper.pos"
DEFAULT_GRIPPER_OUTPUT_KEY = "gripper_normalized"
LEADER_DEGREE_PERIOD = 360.0


@dataclass(frozen=True)
class GalaxeaA1TeleopMapping:
    """Explicit, serializable mapping owned by the application configuration."""

    sign: tuple[float, ...]
    scale: tuple[float, ...]
    bias_rad: tuple[float, ...]
    lower_limits_rad: tuple[float, ...]
    upper_limits_rad: tuple[float, ...]
    max_joint_action_step_rad: float
    leader_joint_keys: tuple[str, ...] = DEFAULT_LEADER_JOINT_KEYS
    a1_joint_keys: tuple[str, ...] = DEFAULT_A1_JOINT_KEYS
    gripper_input_key: str = DEFAULT_GRIPPER_INPUT_KEY
    gripper_output_key: str = DEFAULT_GRIPPER_OUTPUT_KEY
    gripper_source_min: float = 0.0
    gripper_source_max: float = 53.16
    gripper_invert: bool = False
    gripper_saturate: bool = True

    def __post_init__(self) -> None:
        dof = len(self.a1_joint_keys)
        for name, values in (
            ("leader_joint_keys", self.leader_joint_keys),
            ("sign", self.sign),
            ("scale", self.scale),
            ("bias_rad", self.bias_rad),
            ("lower_limits_rad", self.lower_limits_rad),
            ("upper_limits_rad", self.upper_limits_rad),
        ):
            if len(values) != dof:
                raise ValueError(f"{name} expects {dof} values, got {len(values)}")
        numeric_groups = (
            self.sign,
            self.scale,
            self.bias_rad,
            self.lower_limits_rad,
            self.upper_limits_rad,
        )
        if not all(
            math.isfinite(float(value)) for values in numeric_groups for value in values
        ):
            raise ValueError("teleop mapping values must be finite")
        if any(
            lo > hi
            for lo, hi in zip(self.lower_limits_rad, self.upper_limits_rad, strict=True)
        ):
            raise ValueError("teleop mapping contains inverted joint limits")
        if (
            not math.isfinite(self.max_joint_action_step_rad)
            or self.max_joint_action_step_rad <= 0
        ):
            raise ValueError("max_joint_action_step_rad must be finite and positive")
        if not math.isfinite(self.gripper_source_min) or not math.isfinite(
            self.gripper_source_max
        ):
            raise ValueError("gripper source range must be finite")
        if self.gripper_source_max <= self.gripper_source_min:
            raise ValueError("gripper_source_max must exceed gripper_source_min")

    def to_dict(self) -> dict[str, object]:
        return {
            "sign": list(self.sign),
            "scale": list(self.scale),
            "bias_rad": list(self.bias_rad),
            "lower_limits_rad": list(self.lower_limits_rad),
            "upper_limits_rad": list(self.upper_limits_rad),
            "max_joint_action_step_rad": self.max_joint_action_step_rad,
            "leader_joint_keys": list(self.leader_joint_keys),
            "a1_joint_keys": list(self.a1_joint_keys),
            "gripper_input_key": self.gripper_input_key,
            "gripper_output_key": self.gripper_output_key,
            "gripper_source_min": self.gripper_source_min,
            "gripper_source_max": self.gripper_source_max,
            "gripper_invert": self.gripper_invert,
            "gripper_saturate": self.gripper_saturate,
        }


@ProcessorStepRegistry.register("galaxea_a1_relative_anchor")
@dataclass
class GalaxeaA1RelativeAnchorProcessorStep(RobotActionProcessorStep):
    """Map continuous leader motion onto the A1 pose observed on the first call."""

    mapping: GalaxeaA1TeleopMapping
    _leader_previous: tuple[float, ...] | None = field(
        default=None, init=False, repr=False
    )
    _leader_delta_deg: tuple[float, ...] | None = field(
        default=None, init=False, repr=False
    )
    _a1_start: tuple[float, ...] | None = field(default=None, init=False, repr=False)
    _previous_target: tuple[float, ...] | None = field(
        default=None, init=False, repr=False
    )

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise TypeError(
                "A1 relative mapping requires the current robot observation"
            )

        expected = {*self.mapping.leader_joint_keys, self.mapping.gripper_input_key}
        missing = expected - set(action)
        unknown = set(action) - expected
        if missing or unknown:
            raise ValueError(
                f"leader action keys do not match mapping: missing={sorted(missing)} "
                f"unknown={sorted(unknown)}"
            )
        observation_keys = {
            *self.mapping.a1_joint_keys,
            self.mapping.gripper_output_key,
        }
        missing_observation = observation_keys - set(observation)
        if missing_observation:
            raise ValueError(
                "robot observation is missing control features: "
                f"{sorted(missing_observation)}"
            )

        leader_now = tuple(float(action[key]) for key in self.mapping.leader_joint_keys)
        a1_now = tuple(float(observation[key]) for key in self.mapping.a1_joint_keys)
        gripper_now = float(observation[self.mapping.gripper_output_key])
        if not all(
            math.isfinite(value) for value in (*leader_now, *a1_now, gripper_now)
        ):
            raise ValueError("leader action and A1 observation must be finite")
        if self._leader_previous is None:
            self._leader_delta_deg = (0.0,) * len(leader_now)
            self._a1_start = a1_now
            self._leader_previous = leader_now
            self._previous_target = a1_now
            return {
                **dict(zip(self.mapping.a1_joint_keys, a1_now, strict=True)),
                self.mapping.gripper_output_key: gripper_now,
            }
        else:
            assert self._leader_delta_deg is not None
            self._leader_delta_deg = tuple(
                accumulated + math.remainder(current - previous, LEADER_DEGREE_PERIOD)
                for accumulated, current, previous in zip(
                    self._leader_delta_deg,
                    leader_now,
                    self._leader_previous,
                    strict=True,
                )
            )
        self._leader_previous = leader_now
        assert self._leader_delta_deg is not None
        assert self._a1_start is not None

        target = tuple(
            self._a1_start[index]
            + self.mapping.sign[index]
            * self.mapping.scale[index]
            * math.radians(self._leader_delta_deg[index])
            + self.mapping.bias_rad[index]
            for index in range(len(self.mapping.a1_joint_keys))
        )
        target = tuple(
            min(
                self.mapping.upper_limits_rad[index],
                max(self.mapping.lower_limits_rad[index], value),
            )
            for index, value in enumerate(target)
        )
        if self._previous_target is not None:
            for name, previous, current in zip(
                self.mapping.a1_joint_keys,
                self._previous_target,
                target,
                strict=True,
            ):
                step = current - previous
                if abs(step) > self.mapping.max_joint_action_step_rad:
                    raise ValueError(
                        f"live joint action discontinuity for {name}: "
                        f"{previous:.6f}->{current:.6f} rad "
                        f"(step={step:+.6f}, "
                        f"limit={self.mapping.max_joint_action_step_rad:.6f})"
                    )
        self._previous_target = target
        gripper = _normalize_gripper(
            float(action[self.mapping.gripper_input_key]), self.mapping
        )
        return {
            **dict(zip(self.mapping.a1_joint_keys, target, strict=True)),
            self.mapping.gripper_output_key: gripper,
        }

    def reset(self) -> None:
        self._leader_previous = None
        self._leader_delta_deg = None
        self._a1_start = None
        self._previous_target = None

    def get_config(self) -> dict[str, Any]:
        return {"mapping": self.mapping.to_dict()}

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # LeRobot 0.6 record initializes action dataset features from the Robot,
        # which are already the canonical output features of this processor.
        return features


def make_galaxea_a1_processors(mapping: GalaxeaA1TeleopMapping):
    """Return the three pipelines accepted by LeRobot teleoperate/record APIs."""

    teleop_action_processor = RobotProcessorPipeline[
        tuple[RobotAction, dict[str, Any]], RobotAction
    ](
        steps=[GalaxeaA1RelativeAnchorProcessorStep(mapping)],
        name="galaxea_a1_teleop_action",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return (
        teleop_action_processor,
        make_default_robot_action_processor(),
        make_default_robot_observation_processor(),
    )


def _normalize_gripper(value: float, mapping: GalaxeaA1TeleopMapping) -> float:
    if not math.isfinite(value):
        raise ValueError("leader gripper value must be finite")
    normalized = (value - mapping.gripper_source_min) / (
        mapping.gripper_source_max - mapping.gripper_source_min
    )
    if mapping.gripper_invert:
        normalized = 1.0 - normalized
    if mapping.gripper_saturate:
        return min(1.0, max(0.0, normalized))
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("leader gripper value is outside the configured source range")
    return normalized
