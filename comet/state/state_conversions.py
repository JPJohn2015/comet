# python imports
import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
import warnings

# local imports
from comet.utilities.constants import Constants as c


def cartesian_to_elements(state: npt.ArrayLike, mu: float = c.MU_EARTH):
    """Method that converts from a cartesian state representation to a
    classical orbital elements representation.

    Args:
        state (npt.ArrayLike): N x 6 Cartesian State [km , km/s]
        mu (float, optional): Central Body Gravitational Parameter [km^3/s^2]

    Returns:
        elements (npt.ArrayLike): N x 6 Orbital Element State [km, -, rad, rad, rad, rad]
    """
    # Validate inputs
    if state.shape == (6,):
        # Expand dimensions if single state
        state = np.expand_dims(state, axis=0)
    if len(state[0]) != 6:
        raise ValueError("cartesian_to_elements(): State Array must be of length 6")

    # Parse position and velocity
    pos = state[..., :3]
    vel = state[..., 3:]

    # Calculate angular momentum (h), node vector (n) and eccentricity vector (e)
    h = np.cross(pos, vel, axis=-1)
    n = np.cross([0, 0, 1], h, axis=-1)
    e = (1 / mu) * (
        ((norm(vel, axis=-1) ** 2 - mu / norm(pos, axis=-1)) * pos.T).T
        - (np.sum(pos * vel, axis=-1) * vel.T).T
    )

    # Calculate specific energy (xi), semi-major axis (a)
    xi = 0.5 * norm(vel, axis=-1) ** 2 - mu / norm(pos, axis=-1)
    a = np.where(norm(e, axis=-1) != 1.0, -mu / (2 * xi), float("inf"))

    # Calculate inclination (i)
    i = np.arccos(h[..., 2] / norm(h, axis=-1))

    # Calculate right ascension (O), arguement of perigee (w) and true anomaly (v)
    warnings.filterwarnings("ignore")
    n_hat = norm(n, axis=-1)
    O = np.where(n_hat == 0.0, 0.0, np.arccos(n[..., 0] / n_hat))
    e_hat = norm(e, axis=-1)
    w = np.where(n_hat * e_hat == 0.0, 0.0, np.arccos(np.sum(n * e, axis=-1) / (n_hat * e_hat)))
    v = np.where(
        e_hat == 0.0,
        0.0,
        np.arccos(np.sum(e * pos, axis=-1) / (norm(e, axis=-1) * norm(pos, axis=-1))),
    )
    warnings.filterwarnings("default")

    # Adjust Angles
    O = np.where(n[..., 1] < 0.0, 2 * np.pi - O, O)
    w = np.where(e[..., 2] < 0.0, 2 * np.pi - w, w)
    v = np.where(np.sum(pos * vel, axis=-1) < 0.0, 2 * np.pi - v, v)

    # Append orbital elements vector
    elements = np.array([a, norm(e, axis=-1), i, O, w, v]).T

    # Remove from extra list if there is one entry
    if len(elements) == 1:
        elements = elements[0]

    return elements


def elements_to_cartesian(elements: npt.ArrayLike, mu: float = c.MU_EARTH):
    """Method that converts from a classical orbital elements representation to a
    cartesian state representation.

    Args:
        elements (npt.ArrayLike): N x 6 Orbital Element State [km, -, rad, rad, rad, rad]
        mu (float): Central Body Gravitational Parameter [km^3/s^2]

    Returns:
        state (npt.ArrayLike): N x 6 Cartesian State [km , km/s]
    """
    # Validate inputs
    if elements.shape == (6,):
        # Expand dimensions if single orbital elements
        elements = np.expand_dims(elements, axis=0)
    if len(elements[0]) != 6:
        raise ValueError("elements_to_array(): Orbital Elements Array must be of length 6")

    # Position and Velocity vectors in PQW frame
    p = elements[..., 0] * (1 - elements[..., 1] ** 2)
    r_pqw = np.array(
        [
            (p * np.cos(elements[..., 5])) / (1 + elements[..., 1] * np.cos(elements[..., 5])),
            (p * np.sin(elements[..., 5])) / (1 + elements[..., 1] * np.cos(elements[..., 5])),
            np.zeros((len(p))),
        ]
    ).T
    v_pqw = np.array(
        [
            -np.sqrt(mu / p) * np.sin(elements[..., 5]),
            np.sqrt(mu / p) * (elements[..., 1] + np.cos(elements[..., 5])),
            np.zeros((len(p))),
        ]
    ).T

    # Precalculate Trigonometry functions for speed and readability
    cI = np.cos(elements[..., 2])
    sI = np.sin(elements[..., 2])
    cRAAN = np.cos(elements[..., 3])
    sRAAN = np.sin(elements[..., 3])
    cW = np.cos(elements[..., 4])
    sW = np.sin(elements[..., 4])

    # Rotation between PQW frame and ECI frame
    PQW_ECI = np.array(
        [
            [cRAAN * cW - sRAAN * sW * cI, -cRAAN * sW - sRAAN * cW * cI, sRAAN * sI],
            [sRAAN * cW + cRAAN * sW * cI, -sRAAN * sW + cRAAN * cW * cI, -cRAAN * sI],
            [sW * sI, cW * sI, cI],
        ]
    )

    # Rotate vectors
    pos = np.array([np.matmul(PQW_ECI[..., i], r_pqw[i, ...]) for i in range(0, len(elements))])
    vel = np.array([np.matmul(PQW_ECI[..., i], v_pqw[i, ...]) for i in range(0, len(elements))])

    # Append state vector
    state = np.append(pos, vel, axis=-1)

    # Remove from extra list if there is one entry
    if len(state) == 1:
        state = state[0]

    return state
