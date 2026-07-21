"""Reference frames and coordinate transformations for COMET library.

This module provides:
- Frame enumerations (StateFrame, ManeuverFrame)
- Coordinate transformations (ECI ↔ ECEF ↔ LLA)
- IAU2000 Earth orientation data (polar motion, UT1-UTC, nutation)
- IERS Earth orientation parameters

Supported transformations:
    ECI ↔ ECEF: Inertial to Earth-fixed frames (includes polar motion, nutation, precession)
    ECEF ↔ LLA: Earth-fixed to geodetic latitude/longitude/altitude
    ECI ↔ LLA: Direct transformations via ECEF

Reference:
    Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
    IERS Conventions (2010)
"""

from .frame import ManeuverFrame, StateFrame
from .transformations import (
    rot_eci_to_ecef,
    rot_ecef_to_eci,
    eci_to_ecef,
    ecef_to_eci,
    ecef_to_lla,
    lla_to_ecef,
    eci_to_lla,
    lla_to_eci,
)
from .iau2000 import EarthOrientationData, IERSData

__all__ = [
    # Frame enumerations
    "ManeuverFrame",
    "StateFrame",
    # Transformation functions
    "rot_eci_to_ecef",
    "rot_ecef_to_eci",
    "eci_to_ecef",
    "ecef_to_eci",
    "ecef_to_lla",
    "lla_to_ecef",
    "eci_to_lla",
    "lla_to_eci",
    # Earth orientation data
    "EarthOrientationData",
    "IERSData",
]
