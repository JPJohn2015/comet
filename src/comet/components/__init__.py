"""Component models for spacecraft systems.

This module provides classes for spacecraft components:
- Component: Base class for all spacecraft components
- Thruster: Propulsion system components with performance characteristics

Components can be attached to satellites and used in maneuver planning.
"""

from .component import Component
from .thruster import Thruster

__all__ = ["Component", "Thruster"]
