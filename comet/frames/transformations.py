# python imports
import numpy as np
import numpy.typing as npt

# comet imports
from comet.utilities.constants import Constants as c
from comet.state.state import State
from comet.time.epoch import Epoch
from comet.frames.iau2000 import polar_motion, earth_rotation, precession_nutation
from comet.utilities.vector import unit


# ECI and ECEF Transformations
def rot_eci_to_ecef(epoch: npt.ArrayLike):
    """Calculates the Rotation Matrix from ECI to ECEF for the specified Epochs.

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.

    Returns:
        rot( npt.ArrayLike): Nx3x3 Array of Rotation Matricies.
    """
    # Check dimensions of Epoch
    epoch = np.atleast_1d(epoch)

    rot = np.zeros((len(epoch), 3, 3))
    for i in range(0, len(epoch)):
        # Calculate Rotation Matricies
        rot_pm = polar_motion(epoch)
        rot_era = earth_rotation(epoch)
        rot_pn = precession_nutation(epoch)

        rot[i, ...] = np.matmul(rot_pn, np.matmul(rot_era, rot_pm))

    return rot


def rot_ecef_to_eci(epoch: Epoch):
    """Calculates the Rotation Matrix from ECEF to ECI for the specified Epochs.

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.

    Returns:
        rot( npt.ArrayLike): Nx3x3 Array of Rotation Matricies.
    """
    # Check dimensions of Epoch
    epoch = np.atleast_1d(epoch)

    rot = np.zeros((len(epoch), 3, 3))
    for i in range(0, len(epoch)):
        # Calculate Rotation Matricies
        rot_pm = polar_motion(epoch)
        rot_era = earth_rotation(epoch)
        rot_pn = precession_nutation(epoch)

        rot[i, ...] = np.matmul(rot_pm.T, np.matmul(rot_era.T, rot_pn.T))

    return rot


def eci_to_ecef(epoch: npt.ArrayLike, state: npt.ArrayLike):
    """Calculates the Transformed state vector from ECI to ECEF for the specified Epochs.

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.
        state (npt.ArrayLike): Nx6 Array of ECI states.

    Returns:
        ecef (npt.ArrayLike): Nx6 Array of ECEF states.
    """
    # Check dimensions of state and epoch
    state = np.atleast_2d(state)
    epoch = np.atleast_1d(epoch)
    if len(state) != len(epoch):
        raise ValueError("eci_to_ecef(): length of epoch does not match length of state")

    ecef_state = np.zeros(state.shape)
    for i in range(0, len(epoch)):
        # Calculate Rotation Matricies
        rot_pm = polar_motion(epoch[i])
        rot_era = earth_rotation(epoch[i])
        rot_pn = precession_nutation(epoch[i])

        # Calculate ECEF positions
        ecef_pos = np.matmul(rot_pn, np.matmul(rot_era, np.matmul(rot_pm, state[i, 0:3])))
        if len(state[i, :]) == 6:
            ecef_vel = np.matmul(
                rot_pn,
                np.matmul(
                    rot_era,
                    np.matmul(
                        rot_pm,
                        state[i, 3:6]
                        + np.cross([0, 0, c.OMEGA_EARTH], np.matmul(rot_pm, state[i, 0:3])),
                    ),
                ),
            )
            ecef_state[i, :] = np.append(ecef_pos, ecef_vel)
        else:
            ecef_state[i, :] = ecef_pos

    return np.squeeze(ecef_state)


def ecef_to_eci(epoch: npt.ArrayLike, state: npt.ArrayLike):
    """Calculates the Transformed state vector from ECEF to ECI for the specified Epochs.

    Args:
        epoch (npt.ArrayLike): Epoch or Array of Epochs.
        state (npt.ArrayLike): Nx6 Array of ECEF states.

    Returns:
        eci (npt.ArrayLike): Nx6 Array of ECI states.
    """
    # Check dimensions of state and epoch
    state = np.atleast_2d(state)
    epoch = np.atleast_1d(epoch)
    if len(state) != len(epoch):
        raise ValueError("ecef_to_eci(): length of epoch does not match length of state")

    eci_state = np.zeros(state.shape)
    for i in range(0, len(epoch)):
        # Calculate Rotation Matricies
        rot_pm = polar_motion(epoch[i])
        rot_era = earth_rotation(epoch[i])
        rot_pn = precession_nutation(epoch[i])

        # Calculate ECEF positions
        eci_pos = np.matmul(rot_pm.T, np.matmul(rot_era.T, np.matmul(rot_pn.T, state[i, 0:3])))
        if len(state[i, :]) == 6:
            eci_vel = np.matmul(
                rot_pm.T,
                np.matmul(rot_era.T, np.matmul(rot_pn.T, state[i, 3:6]))
                - np.cross(
                    [0, 0, c.OMEGA_EARTH], np.matmul(rot_era.T, np.matmul(rot_pn.T, state[i, 0:3]))
                ),
            )
            eci_state[i, :] = np.append(eci_pos, eci_vel)
        else:
            eci_state[i, :] = eci_pos

    return np.squeeze(eci_state)


# ECEF and LLA Transformations
def ecef_to_lla(ecef_state: npt.ArrayLike):
    """Calculates the Latitude, Longitude and Altitude (LLA) from the ECEF state.

    Args:
        ecef_state (npt.ArrayLike): Nx6 Array of ECEF State in km and km/s.

    Returns:
        lla_state (npt.ArrayLike): Nx3 Array of [Latitude [rad], Longitue [rad], Altitude [km]].
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
    """Calculates the Latitude, Longitude and Altitude (LLA) from the ECEF state.

    Args:
        lla_state (npt.ArrayLike): Nx3 Array of [Latitude [rad], Longitue [rad], Altitude [km]].
        velocity (npt.ArrayLike, optional): Nx3 Array of ECEF velocities. Defaults to [0,0,0].

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
