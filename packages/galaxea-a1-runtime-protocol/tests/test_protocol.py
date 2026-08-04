from __future__ import annotations

import pytest

from galaxea_a1_protocol.codec import manifest_from_proto, manifest_to_proto
from galaxea_a1_protocol.contracts import (
    A1_CONTROL_FEATURE_NAMES,
    FeatureSpec,
    RuntimeContractError,
    RuntimeManifest,
    validate_feature_values,
)
from galaxea_a1_protocol.endpoint import unix_socket_path


def test_manifest_round_trips_through_protobuf() -> None:
    features = tuple(FeatureSpec(name) for name in A1_CONTROL_FEATURE_NAMES)
    manifest = RuntimeManifest(
        identifier="galaxea-a1",
        observation_features=features,
        action_features=features,
        metadata={"robot_type": "galaxea_a1"},
    )

    assert manifest_from_proto(manifest_to_proto(manifest)) == manifest


def test_feature_values_are_exact_and_bounded() -> None:
    features = (FeatureSpec("gripper_normalized", minimum=0.0, maximum=1.0),)

    assert validate_feature_values({"gripper_normalized": 0.5}, features) == {
        "gripper_normalized": 0.5
    }
    with pytest.raises(RuntimeContractError, match="above maximum"):
        validate_feature_values({"gripper_normalized": 1.1}, features)


def test_endpoint_requires_an_absolute_unix_socket() -> None:
    assert str(unix_socket_path("unix:///tmp/a1.sock")) == "/tmp/a1.sock"
    with pytest.raises(RuntimeContractError, match="absolute unix"):
        unix_socket_path("localhost:50051")
