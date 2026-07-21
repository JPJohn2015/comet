"""Orbit propagation and force modeling for COMET library.

This module provides numerical integration and analytical propagation:
- Propagator classes: Base, Space, Ground, Sun, Moon, TLE
- Force models: Gravity, drag, SRP, third-body perturbations
- Perturbation model enums: AtmosphereModel, NBodyModel, SRPModel, GravityPotentialModel
- Integrator enum: RK45, DOP853, etc.

Propagators use scipy.integrate.ode for numerical integration with poliastro
perturbation kernels for high-fidelity force modeling.
"""

from .propagator import (
    Propagator,
    SpacePropagator,
    GroundPropagator,
    SunPropagator,
    MoonPropagator,
    TLEPropagator,
    Integrator,
    PropagatorCategory,
)
from .force_model import ForceModel
from .perturbation_model import (
    AtmosphereModel,
    NBodyModel,
    SRPModel,
    GravityPotentialModel,
)
from .satellite_properties import SatelliteProperties

__all__ = [
    # Propagator classes
    "Propagator",
    "SpacePropagator",
    "GroundPropagator",
    "SunPropagator",
    "MoonPropagator",
    "TLEPropagator",
    # Enums
    "Integrator",
    "PropagatorCategory",
    "AtmosphereModel",
    "NBodyModel",
    "SRPModel",
    "GravityPotentialModel",
    # Supporting classes
    "ForceModel",
    "SatelliteProperties",
]
