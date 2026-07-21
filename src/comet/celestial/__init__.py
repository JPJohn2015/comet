"""Celestial body position calculations for COMET library.

This module provides position calculations for celestial bodies:
- Sun: Solar position at arbitrary epochs (low and high fidelity)
- Moon: Lunar position at arbitrary epochs (low and high fidelity)
- CelestialFidelity: Enum for selecting calculation fidelity

Algorithms from Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
Low fidelity: Simple truncated series (~0.01° accuracy)
High fidelity: Full precision ephemeris models
"""

from .sun import Sun
from .moon import Moon
from .celestial_fidelity import CelestialFidelity

__all__ = ["Sun", "Moon", "CelestialFidelity"]
