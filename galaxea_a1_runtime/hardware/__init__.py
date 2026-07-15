"""Hardware boundary interfaces.

Concrete hardware adapters live at this boundary. Core runtime code depends on
the hardware protocol, not on ROS.
"""

from __future__ import annotations

from .io import A1HardwareIO, A1Observation, InMemoryA1HardwareIO

__all__ = [
    "A1HardwareIO",
    "A1Observation",
    "InMemoryA1HardwareIO",
]
