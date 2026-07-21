"""Utility functions and constants for COMET library.

This module provides:
- Constants: Physical and astronomical constants (Earth parameters, gravitational constants, etc.)
- Vector operations: Cross products, norms, etc.
- Rotation matrices: Euler angle rotations
- Astrodynamics utilities: Anomaly conversions, etc.

All constants are in SI-derived units: km, km/s, radians.
"""

from .constants import Constants
from . import vector
from . import rotations
from . import astro

__all__ = ["Constants", "vector", "rotations", "astro"]
