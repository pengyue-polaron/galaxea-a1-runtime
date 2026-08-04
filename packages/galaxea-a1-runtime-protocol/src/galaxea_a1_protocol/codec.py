"""Strict protobuf conversion for the A1 Runtime transport."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct

from .contracts import (
    FeatureSpec,
    HealthReport,
    HealthStatus,
    RuntimeContractError,
    RuntimeManifest,
    validate_feature_values,
)
from .v1 import a1_runtime_pb2


def manifest_to_proto(manifest: RuntimeManifest) -> a1_runtime_pb2.RuntimeManifest:
    return a1_runtime_pb2.RuntimeManifest(
        api_version=manifest.api_version,
        identifier=manifest.identifier,
        observation_features=[
            feature_to_proto(feature) for feature in manifest.observation_features
        ],
        action_features=[feature_to_proto(feature) for feature in manifest.action_features],
        metadata=dict(manifest.metadata),
    )


def manifest_from_proto(message: a1_runtime_pb2.RuntimeManifest) -> RuntimeManifest:
    return RuntimeManifest(
        api_version=message.api_version,
        identifier=message.identifier,
        observation_features=tuple(
            feature_from_proto(feature) for feature in message.observation_features
        ),
        action_features=tuple(feature_from_proto(feature) for feature in message.action_features),
        metadata=dict(message.metadata),
    )


def feature_to_proto(feature: FeatureSpec) -> a1_runtime_pb2.FeatureSpec:
    message = a1_runtime_pb2.FeatureSpec(name=feature.name)
    if feature.unit is not None:
        message.unit = feature.unit
    if feature.minimum is not None:
        message.minimum = feature.minimum
    if feature.maximum is not None:
        message.maximum = feature.maximum
    return message


def feature_from_proto(message: a1_runtime_pb2.FeatureSpec) -> FeatureSpec:
    return FeatureSpec(
        name=message.name,
        unit=message.unit if message.HasField("unit") else None,
        minimum=message.minimum if message.HasField("minimum") else None,
        maximum=message.maximum if message.HasField("maximum") else None,
    )


def values_to_proto(
    values: Mapping[str, object], features: tuple[FeatureSpec, ...]
) -> list[a1_runtime_pb2.FeatureValue]:
    validated = validate_feature_values(values, features)
    return [
        a1_runtime_pb2.FeatureValue(name=feature.name, scalar=validated[feature.name])
        for feature in features
    ]


def values_from_proto(
    messages: Sequence[a1_runtime_pb2.FeatureValue],
    features: tuple[FeatureSpec, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for message in messages:
        if message.name in values:
            raise RuntimeContractError(f"duplicate A1 feature value: {message.name!r}")
        values[message.name] = message.scalar
    return validate_feature_values(values, features)


def health_to_proto(report: HealthReport) -> a1_runtime_pb2.HealthResponse:
    details = Struct()
    try:
        ParseDict(dict(report.details), details)
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError("A1 health details must be protobuf-Struct compatible") from exc
    return a1_runtime_pb2.HealthResponse(
        status=report.status.value,
        summary=report.summary,
        details=details,
    )


def health_from_proto(message: a1_runtime_pb2.HealthResponse) -> HealthReport:
    return HealthReport(
        status=HealthStatus(message.status),
        summary=message.summary,
        details=MessageToDict(message.details),
    )
