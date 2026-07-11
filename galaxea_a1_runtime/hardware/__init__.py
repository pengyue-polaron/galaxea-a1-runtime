"""Hardware boundary interfaces.

Concrete hardware adapters live at this boundary. Core runtime code depends on
the protocol and pure EEF helpers, not on ROS.
"""

from __future__ import annotations

from .eef import EefPose, action_to_eef_target
from .io import A1HardwareIO, A1Observation, NullA1HardwareIO

__all__ = [
    "A1HardwareIO",
    "A1Observation",
    "EefPose",
    "NullA1HardwareIO",
    "action_to_eef_target",
]
