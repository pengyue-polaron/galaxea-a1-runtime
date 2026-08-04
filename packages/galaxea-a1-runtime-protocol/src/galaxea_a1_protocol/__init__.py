"""Lightweight wire contract and client for the Galaxea A1 Runtime."""

from ._version import __version__
from .client import A1RuntimeClient
from .contracts import (
    A1_CONTROL_FEATURE_NAMES,
    API_VERSION,
    FeatureSpec,
    HealthReport,
    HealthStatus,
    RuntimeContractError,
    RuntimeDevice,
    RuntimeLifecycleError,
    RuntimeManifest,
    RuntimeRpcError,
    validate_feature_values,
)
from .endpoint import unix_socket_path
from .types import PROTOCOL_VERSION, SessionMode

__all__ = [
    "A1_CONTROL_FEATURE_NAMES",
    "API_VERSION",
    "PROTOCOL_VERSION",
    "A1RuntimeClient",
    "FeatureSpec",
    "HealthReport",
    "HealthStatus",
    "RuntimeContractError",
    "RuntimeDevice",
    "RuntimeLifecycleError",
    "RuntimeManifest",
    "RuntimeRpcError",
    "SessionMode",
    "__version__",
    "unix_socket_path",
    "validate_feature_values",
]
