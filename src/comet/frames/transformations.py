# python imports
import numpy as np
import numpy.typing as npt

# comet imports
from comet.utilities.constants import Constants as c
from comet.time.epoch import Epoch
from comet.frames.iau2000 import polar_motion, earth_rotation, precession_nutation
from comet.utilities.vector import unit


# ECI and ECEF Transformations
def rot_eci_to_ecef(epoch: npt.ArrayLike):
    """Calculates the Rotation Matrix from ECI to ECEF for the specified Epochs.

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.

    Returns:
        rot (npt.ArrayLike): Nx3x3 Array of Rotation Matrices.
    """
    # Calculate Rotation Matrices (vectorized)
    rot_pm = polar_motion(epoch)
    rot_era = earth_rotation(epoch)
    rot_pn = precession_nutation(epoch)

    # Handle both scalar and array cases
    if rot_pm.ndim == 2:
        # Single epoch case
        return np.matmul(rot_pm, np.matmul(rot_era, rot_pn))
    else:
        # Multiple epochs case
        return np.matmul(rot_pm, np.matmul(rot_era, rot_pn))


def rot_ecef_to_eci(epoch: Epoch):
    """Calculates the Rotation Matrix from ECEF to ECI for the specified Epochs.

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.

    Returns:
        rot (npt.ArrayLike): Nx3x3 Array of Rotation Matrices.
    """
    # Calculate Rotation Matrices (vectorized)
    rot_pm = polar_motion(epoch)
    rot_era = earth_rotation(epoch)
    rot_pn = precession_nutation(epoch)

    # Handle both scalar and array cases
    if rot_pm.ndim == 2:
        # Single epoch case
        return np.matmul(rot_pn.T, np.matmul(rot_era.T, rot_pm.T))
    else:
        # Multiple epochs case - transpose last two dimensions
        return np.matmul(np.transpose(rot_pn, (0, 2, 1)), np.matmul(np.transpose(rot_era, (0, 2, 1)), np.transpose(rot_pm, (0, 2, 1))))


def eci_to_ecef(epoch: npt.ArrayLike, state: npt.ArrayLike):
    """Calculates the Transformed state vector from ECI to ECEF for the specified Epochs.

    Implements IAU 2000 reduction: GCRS (ECI) to ITRS (ECEF) transformation.
    Transformation chain: R_ECEF = R_pm × R_era × R_pn × R_ECI

    Reference:
        IERS Conventions (2010), Chapter 5
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Section 3.7

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.
        state (npt.ArrayLike): Nx6 Array of ECI states in km and km/s.

    Returns:
        ecef (npt.ArrayLike): Nx6 Array of ECEF states in km and km/s.
    """
    # Check dimensions of state and epoch
    state = np.atleast_2d(state)
    epoch = np.atleast_1d(epoch)
    if len(state) != len(epoch):
        raise ValueError("eci_to_ecef(): length of epoch does not match length of state")

    # Calculate Rotation Matrices (vectorized)
    rot_pm = polar_motion(epoch)
    rot_era = earth_rotation(epoch)
    rot_pn = precession_nutation(epoch)

    # Ensure matrices are 3D (N, 3, 3)
    if rot_pm.ndim == 2:
        rot_pm = rot_pm[np.newaxis, ...]
        rot_era = rot_era[np.newaxis, ...]
        rot_pn = rot_pn[np.newaxis, ...]

    # Transform positions using vectorized matmul
    pos_eci = state[:, 0:3]
    pos_temp = np.einsum("nij,nj->ni", rot_pn, pos_eci)
    pos_temp = np.einsum("nij,nj->ni", rot_era, pos_temp)
    ecef_pos = np.einsum("nij,nj->ni", rot_pm, pos_temp)

    # Transform velocities if present
    if state.shape[1] == 6:
        vel_eci = state[:, 3:6]
        # Account for Earth rotation velocity: V_ECEF = R × (V_ECI - ω × R_ECI)
        # where ω = [0, 0, OMEGA_EARTH] is Earth's angular velocity in ECI
        # Reference: IERS Conventions (2010), Section 5.4.4
        omega_cross_pos = np.cross([0, 0, c.OMEGA_EARTH], pos_eci)
        vel_adjusted = vel_eci - omega_cross_pos
        # Apply full rotation chain
        vel_temp = np.einsum("nij,nj->ni", rot_pn, vel_adjusted)
        vel_temp = np.einsum("nij,nj->ni", rot_era, vel_temp)
        ecef_vel = np.einsum("nij,nj->ni", rot_pm, vel_temp)
        ecef_state = np.hstack([ecef_pos, ecef_vel])
    else:
        ecef_state = ecef_pos

    return np.squeeze(ecef_state)


def ecef_to_eci(epoch: npt.ArrayLike, state: npt.ArrayLike):
    """Calculates the Transformed state vector from ECEF to ECI for the specified Epochs.

    Implements IAU 2000 reduction: ITRS (ECEF) to GCRS (ECI) transformation.
    Transformation chain: R_ECI = R_pn^T × R_era^T × R_pm^T × R_ECEF

    Reference:
        IERS Conventions (2010), Chapter 5
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Section 3.7

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.
        state (npt.ArrayLike): Nx6 Array of ECEF states in km and km/s.

    Returns:
        eci (npt.ArrayLike): Nx6 Array of ECI states in km and km/s.
    """
    # Check dimensions of state and epoch
    state = np.atleast_2d(state)
    epoch = np.atleast_1d(epoch)
    if len(state) != len(epoch):
        raise ValueError("ecef_to_eci(): length of epoch does not match length of state")

    # Calculate Rotation Matrices (vectorized)
    rot_pm = polar_motion(epoch)
    rot_era = earth_rotation(epoch)
    rot_pn = precession_nutation(epoch)

    # Ensure matrices are 3D (N, 3, 3)
    if rot_pm.ndim == 2:
        rot_pm = rot_pm[np.newaxis, ...]
        rot_era = rot_era[np.newaxis, ...]
        rot_pn = rot_pn[np.newaxis, ...]

    # Transform positions using vectorized matmul (transpose for inverse)
    pos_ecef = state[:, 0:3]
    pos_temp = np.einsum("nji,nj->ni", rot_pm, pos_ecef)
    pos_temp = np.einsum("nji,nj->ni", rot_era, pos_temp)
    eci_pos = np.einsum("nji,nj->ni", rot_pn, pos_temp)

    # Transform velocities if present
    if state.shape[1] == 6:
        vel_ecef = state[:, 3:6]
        # Account for Earth rotation velocity: V_ECI = R^T × V_ECEF + ω × R_ECI
        # where ω = [0, 0, OMEGA_EARTH] is Earth's angular velocity in ECI
        # Reference: IERS Conventions (2010), Section 5.4.4
        # Rotate velocity through full chain
        vel_temp = np.einsum("nji,nj->ni", rot_pm, vel_ecef)
        vel_temp = np.einsum("nji,nj->ni", rot_era, vel_temp)
        vel_temp = np.einsum("nji,nj->ni", rot_pn, vel_temp)
        # Add Earth rotation effect
        omega_cross_pos = np.cross([0, 0, c.OMEGA_EARTH], eci_pos)
        eci_vel = vel_temp + omega_cross_pos
        eci_state = np.hstack([eci_pos, eci_vel])
    else:
        eci_state = eci_pos

    return np.squeeze(eci_state)


# ECEF and LLA Transformations
def ecef_to_lla(ecef_state: npt.ArrayLike):
    """Calculates the Latitude, Longitude and Altitude (LLA) from the ECEF state.

    Converts Earth-Centered Earth-Fixed Cartesian coordinates to geodetic coordinates
    using WGS-84 ellipsoid parameters.

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Algorithm 12, pg. 172-173

    Args:
        ecef_state (npt.ArrayLike): Nx6 Array of ECEF State in km and km/s.

    Returns:
        lla_state (npt.ArrayLike): Nx3 Array of [Latitude [rad], Longitude [rad], Altitude [km]].
    """
    # Parse ECEF and convert to m
    ecef_state = np.atleast_2d(ecef_state)
    x = ecef_state[:, 0] * 1000
    y = ecef_state[:, 1] * 1000
    z = ecef_state[:, 2] * 1000

    # Calculate longitude
    lon = np.arctan2(y, x)

    # Calculate latitude
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z, (p * (1 - c.E_SQ_EARTH)))
    lat = np.arctan2(
        (z + c.EP_SQ_EARTH * c.B_EARTH * np.sin(theta) ** 3),
        (p - c.E_SQ_EARTH * c.A_EARTH * np.cos(theta) ** 3),
    )

    # Calculate altitude
    radius = c.A_EARTH / np.sqrt(1 - c.E_SQ_EARTH * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - radius

    # Package LLA coordinates
    lla_state = np.array([lat, lon, alt / 1000]).T

    return np.squeeze(lla_state)


def lla_to_ecef(lla_state: npt.ArrayLike, velocity: npt.ArrayLike = None):
    """Calculates the ECEF state from Latitude, Longitude and Altitude (LLA).

    Converts geodetic coordinates to Earth-Centered Earth-Fixed Cartesian coordinates
    using WGS-84 ellipsoid parameters.

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Algorithm 12, pg. 172-173

    Args:
        lla_state (npt.ArrayLike): Nx3 Array of [Latitude [rad], Longitude [rad], Altitude [km]].
        velocity (npt.ArrayLike, optional): Nx3 Array of ECEF velocities in km/s. Defaults to [0,0,0].

    Returns:
        ecef_state (npt.ArrayLike): ECEF state in km and km/s.
    """
    # Parse LLA and convert altitude to m
    lla_state = np.atleast_2d(lla_state)
    lat = lla_state[:, 0]
    lon = lla_state[:, 1]
    alt = lla_state[:, 2] * 1000

    # Calculate radius of curvature in the prime vertical
    radius = c.A_EARTH / np.sqrt(1 - c.E_SQ_EARTH * np.sin(lat) ** 2)

    # Calculate ECEF coordinates
    x = (radius + alt) * np.cos(lat) * np.cos(lon)
    y = (radius + alt) * np.cos(lat) * np.sin(lon)
    z = (radius * (1 - c.E_SQ_EARTH) + alt) * np.sin(lat)

    # Package ECEF coordinates
    if velocity is None:
        ecef_state = np.array(
            [x / 1000, y / 1000, z / 1000, np.zeros(x.shape), np.zeros(y.shape), np.zeros(z.shape)]
        ).T
    else:
        # Handle both 1D and 2D velocity arrays
        velocity = np.atleast_2d(velocity)
        ecef_state = np.array(
            [x / 1000, y / 1000, z / 1000, velocity[:, 0], velocity[:, 1], velocity[:, 2]]
        ).T

    return np.squeeze(ecef_state)


# ECI and LLA Transformations
def eci_to_lla(epoch: npt.ArrayLike, eci_state: npt.ArrayLike):
    """Calculates the Latitude, Longitude and Altitude (LLA) from the ECI state.

    Args:
        epoch (npt.ArrayLike): Nx1 Array of Epochs.
        eci_state (npt.ArrayLike): Nx6 Array of ECI State in km and km/s.

    Returns:
        lla_state (npt.ArrayLike): Nx3 Array of [Latitude [rad], Longitue [rad], Altitude [km]].
    """
    # Calculate ECEF, then LLA
    ecef_state = eci_to_ecef(epoch, eci_state)
    lla_state = ecef_to_lla(ecef_state)

    return lla_state


def lla_to_eci(epoch: npt.ArrayLike, lla_state: npt.ArrayLike, velocity: npt.ArrayLike = None):
    """Calculates the Latitude, Longitude and Altitude (LLA) from the ECEF state.

    Args:
        epoch (npt.ArrayLike): Nx1 Array of Epochs.
        lla_state (npt.ArrayLike): Nx3 Array of [Latitude [rad], Longitue [rad], Altitude [km]].
        velocity (npt.ArrayLike, optional): Nx3 Array of ECEF velocities. Defaults to [0,0,0].

    Returns:
        ecef_state (npt.ArrayLike): ECEF state in km and km/s.
    """
    # Calculate ECEF, then ECI
    ecef_state = lla_to_ecef(lla_state)
    eci_state = ecef_to_eci(epoch, ecef_state)

    return eci_state


# Testing
if __name__ == "__main__":
    eci = np.array(
        [
            [-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446],
            [-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446],
            [-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446],
            [-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446],
        ]
    )
    epoch = np.array(
        [
            Epoch(2004, 4, 6, 7, 51, 28.386009),
            Epoch(2004, 4, 6, 7, 51, 28.386009),
            Epoch(2004, 4, 6, 7, 51, 28.386009),
            Epoch(2004, 4, 6, 7, 51, 28.386009),
        ]
    )

    ecef = eci_to_ecef(epoch, eci)
    print(ecef)
    eci_back = ecef_to_eci(epoch, ecef)
    print(eci_back)

    print("------")
    print(ecef)
    lla = ecef_to_lla(ecef)
    print(lla)
    ecef2 = lla_to_ecef(lla)
    print(ecef2)

    print("------")
    print(eci)
    lla = eci_to_lla(epoch, eci)
    print(lla)
    eci2 = lla_to_eci(epoch, lla)
    print(eci2)
