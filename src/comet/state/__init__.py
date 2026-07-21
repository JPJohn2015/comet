"""State representation module for COMET library.

This module provides classes for representing spacecraft states in various forms:
- State: Cartesian position and velocity (ECI, ECEF, etc.)
- Elements: Keplerian orbital elements
- LLA: Latitude, longitude, altitude
- TLE: Two-line element sets

All state representations support conversion between forms and frame transformations.
"""

from .state import State
from .elements import Elements
from .lla import LLA
from .tle import TLE

__all__ = ["State", "Elements", "LLA", "TLE"]
