"""Galaxea A1 runtime core package.

The package is intentionally split so pure safety and dataset schema logic can
be tested without ROS, Docker, cameras, or powered hardware.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
