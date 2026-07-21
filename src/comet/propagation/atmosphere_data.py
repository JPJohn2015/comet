# python imports
import numpy as np

# COMET imports
# None

"""
Atmospheric Data for Drag Modeling

This module contains atmospheric density tables and interpolation functions
for various atmosphere models used in orbital propagation.

Data sources:
- COESA76: U.S. Standard Atmosphere 1976
  Reference: NASA-TM-X-74335, "U.S. Standard Atmosphere, 1976"
  https://ntrs.nasa.gov/citations/19770009539
"""


# U.S. Standard Atmosphere 1976 (COESA76) Density Table
# Geometric altitude [km] vs. Density [kg/m³]
# Valid for altitudes 100-1000 km
# Source: NASA-TM-X-74335, Table III pg. 51-73
COESA76_ALTITUDE = np.array([
    # Low Earth Orbit regime (100-300 km) - more detailed
    100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
    200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
    300, 320, 340, 360, 380, 400,
    # Medium altitude (400-600 km) - less detailed
    420, 440, 460, 480, 500, 520, 540, 560, 580, 600,
    # High altitude (600-1000 km) - sparse
    620, 640, 660, 680, 700, 750, 800, 850, 900, 950, 1000
])  # km

COESA76_DENSITY = np.array([
    # 100-300 km (most critical for drag)
    5.604e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9, 2.076e-9,
    1.233e-9, 7.815e-10, 5.194e-10, 3.581e-10,
    2.541e-10, 1.854e-10, 1.389e-10, 1.064e-10, 8.300e-11, 6.598e-11,
    5.328e-11, 4.368e-11, 3.622e-11, 3.040e-11,
    2.578e-11, 1.916e-11, 1.451e-11, 1.114e-11, 8.684e-12, 6.858e-12,
    # 400-600 km
    5.469e-12, 4.403e-12, 3.577e-12, 2.935e-12, 2.438e-12, 2.046e-12,
    1.732e-12, 1.478e-12, 1.270e-12, 1.097e-12,
    # 600-1000 km
    9.545e-13, 8.339e-13, 7.317e-13, 6.445e-13, 5.704e-13,
    4.008e-13, 2.876e-13, 2.105e-13, 1.569e-13, 1.192e-13, 9.201e-14
])  # kg/m³


def get_density_coesa76(altitude_km: float) -> float:
    """Get atmospheric density using COESA76 tables with linear interpolation.

    Uses the U.S. Standard Atmosphere 1976 density tables for altitudes
    from 100 to 1000 km. Linear interpolation in log-density space is
    used between tabulated values.

    For altitudes below 100 km, returns sea level density (not accurate).
    For altitudes above 1000 km, extrapolates using exponential decay.

    Reference: NASA-TM-X-74335, "U.S. Standard Atmosphere, 1976"

    Args:
        altitude_km (float): Geometric altitude above sea level in km.

    Returns:
        float: Atmospheric density in kg/m³.
    """
    # Handle edge cases
    if altitude_km < 100.0:
        # Below 100 km, use first table entry (not accurate for very low altitudes)
        return COESA76_DENSITY[0]

    if altitude_km > 1000.0:
        # Above 1000 km, use exponential extrapolation
        # Use scale height of ~70 km for upper atmosphere
        h_scale = 70.0  # km
        rho_1000 = COESA76_DENSITY[-1]
        return rho_1000 * np.exp(-(altitude_km - 1000.0) / h_scale)

    # Interpolate in log space for better accuracy over large density ranges
    # Log-linear interpolation: log(ρ) varies linearly with altitude
    log_density = np.log(COESA76_DENSITY)

    # Use numpy interpolation
    log_rho = np.interp(altitude_km, COESA76_ALTITUDE, log_density)

    return np.exp(log_rho)


def get_density_exponential(altitude_km: float,
                            h0: float = 700.0,
                            rho0: float = 5.24e-14,
                            H: float = 88.667) -> float:
    """Get atmospheric density using simple exponential model.

    ρ = ρ₀ * exp(-(h - h₀) / H)

    This is a simplified model that works reasonably well for altitudes
    above ~200 km but is less accurate than tabulated models.

    Args:
        altitude_km (float): Geometric altitude above sea level in km.
        h0 (float): Reference altitude in km (default: 700 km).
        rho0 (float): Reference density at h0 in kg/m³ (default: 5.24e-14).
        H (float): Scale height in km (default: 88.667 km).

    Returns:
        float: Atmospheric density in kg/m³.
    """
    return rho0 * np.exp(-(altitude_km - h0) / H)
