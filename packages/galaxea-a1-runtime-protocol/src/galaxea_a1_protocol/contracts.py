"""A1-specific contracts shared by the LeRobot client and Runtime host."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

API_VERSION = 1
A1_CONTROL_FEATURE_NAMES = (
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_normalized",
)


class RuntimeContractError(ValueError):
    """The A1 Runtime protocol received an invalid value."""


class RuntimeLifecycleError(RuntimeError):
    """An A1 Runtime session operation is invalid or failed."""


class RuntimeRpcError(RuntimeError):
    """An A1 Runtime RPC could not be completed."""


class HealthStatus(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class HealthReport:
    status: HealthStatus
    summary: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary:
            raise RuntimeContractError("health summary must not be empty")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One named scalar in the fixed A1 observation or action contract."""

    name: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name.strip() != self.name
            or any(character.isspace() for character in self.name)
        ):
            raise RuntimeContractError(f"invalid A1 feature name: {self.name!r}")
        for label, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is not None and not math.isfinite(float(value)):
                raise RuntimeContractError(f"A1 feature {self.name!r} has non-finite {label}")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise RuntimeContractError(f"A1 feature {self.name!r} minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    """Description of the scalar A1 control surface exposed to LeRobot."""

    identifier: str
    observation_features: tuple[FeatureSpec, ...]
    action_features: tuple[FeatureSpec, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)
    api_version: int = API_VERSION

    def __post_init__(self) -> None:
        if self.identifier != "galaxea-a1":
            raise RuntimeContractError(f"unexpected A1 Runtime identifier: {self.identifier!r}")
        if self.api_version != API_VERSION:
            raise RuntimeContractError(
                f"unsupported A1 Runtime API {self.api_version}; expected {API_VERSION}"
            )
        _index_features(self.observation_features)
        _index_features(self.action_features)
        if not self.observation_features or not self.action_features:
            raise RuntimeContractError("A1 Runtime manifest must expose observations and actions")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _index_features(features: tuple[FeatureSpec, ...]) -> dict[str, FeatureSpec]:
    result: dict[str, FeatureSpec] = {}
    for feature in features:
        if feature.name in result:
            raise RuntimeContractError(f"duplicate A1 feature name: {feature.name!r}")
        result[feature.name] = feature
    return result


def validate_feature_values(
    values: Mapping[str, object], features: tuple[FeatureSpec, ...]
) -> dict[str, float]:
    """Validate exact named scalar values without clamping or rewriting them."""

    specs = _index_features(features)
    missing = set(specs) - set(values)
    unknown = set(values) - set(specs)
    if missing:
        raise RuntimeContractError(f"missing A1 feature values: {sorted(missing)}")
    if unknown:
        raise RuntimeContractError(f"unknown A1 feature values: {sorted(unknown)}")
    result: dict[str, float] = {}
    for name, spec in specs.items():
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise RuntimeContractError(f"A1 feature {name!r} must be a real scalar")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeContractError(f"A1 feature {name!r} must be finite")
        if spec.minimum is not None and numeric < spec.minimum:
            raise RuntimeContractError(f"A1 feature {name!r} is below minimum {spec.minimum}")
        if spec.maximum is not None and numeric > spec.maximum:
            raise RuntimeContractError(f"A1 feature {name!r} is above maximum {spec.maximum}")
        result[name] = numeric
    return result


class RuntimeDevice(Protocol):
    """A1 backend hosted by the Runtime-owned service process."""

    @property
    def manifest(self) -> RuntimeManifest: ...

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def observe(self) -> Mapping[str, object]: ...

    def acquire_command_lease(self) -> None: ...

    def release_command_lease(self) -> None: ...

    def command(self, action: Mapping[str, object]) -> Mapping[str, object]: ...

    def health(self) -> HealthReport: ...

    def disconnect(self) -> None: ...
