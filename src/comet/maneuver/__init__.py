"""Maneuver planning and execution for COMET library.

This module provides classes for spacecraft maneuvers:
- Maneuver: Base class for all maneuvers
- ImpulsiveManeuver: Instantaneous delta-v maneuvers
- FiniteManeuver: Finite-burn maneuvers with thrust profiles

Maneuvers can be applied to satellite states and integrated into propagation.
"""

from .maneuver import Maneuver, ImpulsiveManeuver, FiniteManeuver

__all__ = ["Maneuver", "ImpulsiveManeuver", "FiniteManeuver"]
