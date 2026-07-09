# python imports
import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

# local imports
from comet.utilities.constants import Constants as c


def cartesian_to_elements(state: npt.ArrayLike, mu: float = c.MU_EARTH):
    """Converts Cartesian state to classical orbital elements.

    Implements the standard Cartesian to Keplerian element transformation using
    the angular momentum and eccentricity vectors.

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Algorithm 9, pg. 113-118

    Coordinate Frame:
        Input: ECI Cartesian coordinates (inertial)
        Output: Classical orbital elements with RAAN/inclination relative to equatorial plane

    Singularity Handling:
        - Circular orbits (e ≈ 0): ω (argument of perigee) is undefined, set to 0
        - Equatorial orbits (i ≈ 0): Ω (RAAN) is undefined, set to 0
        - Parabolic orbits (e = 1): a (semi-major axis) is infinite

    Args:
        state (npt.ArrayLike): N x 6 or 6-element Cartesian State [km, km/s]
                              Position and velocity in ECI frame
        mu (float, optional): Central Body Gravitational Parameter [km^3/s^2]
                            Default: Earth (398600.4418 km^3/s^2)

    Returns:
        elements (npt.ArrayLike): N x 6 or 6-element Orbital Element State
                                [a, e, i, Ω, ω, ν] in [km, -, rad, rad, rad, rad]
                                - a: Semi-major axis (negative for hyperbolic, inf for parabolic)
                                - e: Eccentricity (0=circular, <1=elliptical, 1=parabolic, >1=hyperbolic)
                                - i: Inclination (0 to π rad)
                                - Ω: Right ascension of ascending node (0 to 2π rad)
                                - ω: Argument of perigee (0 to 2π rad)
                                - ν: True anomaly (0 to 2π rad)
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
    # Node vector n = k x h, where k = [0, 0, 1]
    # Optimize: n = [0,0,1] x [hx,hy,hz] = [-hy, hx, 0]
    n = np.stack([-h[..., 1], h[..., 0], np.zeros(h.shape[:-1])], axis=-1)
    e = (1 / mu) * (
        ((norm(vel, axis=-1) ** 2 - mu / norm(pos, axis=-1)) * pos.T).T
        - (np.sum(pos * vel, axis=-1) * vel.T).T
    )

    # Calculate specific energy (xi), semi-major axis (a)
    xi = 0.5 * norm(vel, axis=-1) ** 2 - mu / norm(pos, axis=-1)
    a = np.where(norm(e, axis=-1) != 1.0, -mu / (2 * xi), float("inf"))

    # Calculate inclination (i)
    i = np.arccos(h[..., 2] / norm(h, axis=-1))

    # Calculate right ascension (O), argument of perigee (w) and true anomaly (v)
    # Use errstate to suppress division by zero warnings for degenerate cases
    with np.errstate(invalid='ignore', divide='ignore'):
        n_hat = norm(n, axis=-1)
        O = np.where(n_hat == 0.0, 0.0, np.arccos(n[..., 0] / n_hat))
        e_hat = norm(e, axis=-1)
        w = np.where(n_hat * e_hat == 0.0, 0.0, np.arccos(np.sum(n * e, axis=-1) / (n_hat * e_hat)))
        v = np.where(
            e_hat == 0.0,
            0.0,
            np.arccos(np.sum(e * pos, axis=-1) / (norm(e, axis=-1) * norm(pos, axis=-1))),
        )

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
    """Converts classical orbital elements to Cartesian state.

    Implements the standard Keplerian to Cartesian transformation using the
    perifocal (PQW) frame and rotation to ECI.

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Algorithm 10, pg. 118-119

    Coordinate Frame:
        Input: Classical orbital elements relative to equatorial plane
        Output: ECI Cartesian coordinates (inertial)

    Method:
        1. Compute position and velocity in perifocal (PQW) frame
        2. Construct rotation matrix from PQW to ECI using Euler angles (Ω, i, ω)
        3. Rotate position and velocity vectors to ECI frame

    Args:
        elements (npt.ArrayLike): N x 6 or 6-element Orbital Element State
                                [a, e, i, Ω, ω, ν] in [km, -, rad, rad, rad, rad]
                                - a: Semi-major axis
                                - e: Eccentricity
                                - i: Inclination
                                - Ω: Right ascension of ascending node
                                - ω: Argument of perigee
                                - ν: True anomaly
        mu (float): Central Body Gravitational Parameter [km^3/s^2]
                   Default: Earth (398600.4418 km^3/s^2)

    Returns:
        state (npt.ArrayLike): N x 6 or 6-element Cartesian State [km, km/s]
                              Position and velocity in ECI frame
    """
    # Validate inputs
    if elements.shape == (6,):
        # Expand dimensions if single orbital elements
        elements = np.expand_dims(elements, axis=0)
    if len(elements[0]) != 6:
        raise ValueError("elements_to_cartesian(): Orbital Elements Array must be of length 6")

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
    # Construct as (N, 3, 3) rotation matrices for batch processing
    N = len(elements)
    PQW_ECI = np.zeros((N, 3, 3))
    PQW_ECI[:, 0, 0] = cRAAN * cW - sRAAN * sW * cI
    PQW_ECI[:, 0, 1] = -cRAAN * sW - sRAAN * cW * cI
    PQW_ECI[:, 0, 2] = sRAAN * sI
    PQW_ECI[:, 1, 0] = sRAAN * cW + cRAAN * sW * cI
    PQW_ECI[:, 1, 1] = -sRAAN * sW + cRAAN * cW * cI
    PQW_ECI[:, 1, 2] = -cRAAN * sI
    PQW_ECI[:, 2, 0] = sW * sI
    PQW_ECI[:, 2, 1] = cW * sI
    PQW_ECI[:, 2, 2] = cI

    # Rotate vectors using batch matrix-vector multiplication via einsum
    # Einstein notation: 'ijk,ik->ij' means for each i, multiply matrix [j,k] by vector [k]
    pos = np.einsum('ijk,ik->ij', PQW_ECI, r_pqw)
    vel = np.einsum('ijk,ik->ij', PQW_ECI, v_pqw)

    # Append state vector
    state = np.append(pos, vel, axis=-1)

    # Remove from extra list if there is one entry
    if len(state) == 1:
        state = state[0]

    return state


def validate_state(state: npt.ArrayLike, warn: bool = True) -> bool:
    """Validate physical constraints on Cartesian state vector.

    Checks for:
    - NaN or Inf values
    - Position inside Earth radius
    - Unreasonable position magnitudes (> 1e6 km)
    - Unreasonable velocity magnitudes (> escape velocity at 1 Earth radius)

    Args:
        state (npt.ArrayLike): N x 6 or 6-element state vector [km, km/s]
        warn (bool): If True, print warnings for constraint violations

    Returns:
        bool: True if state is valid, False otherwise
    """
    state_arr = np.atleast_2d(state)

    # Check for NaN or Inf
    if np.any(np.isnan(state_arr)) or np.any(np.isinf(state_arr)):
        if warn:
            print("Warning: State contains NaN or Inf values")
        return False

    # Extract position and velocity
    pos = state_arr[:, :3]
    vel = state_arr[:, 3:]

    # Check position magnitude
    r_mag = np.linalg.norm(pos, axis=-1)

    # Check if inside Earth
    if np.any(r_mag < c.RADIUS_EARTH):
        if warn:
            print(f"Warning: State position inside Earth radius ({c.RADIUS_EARTH} km)")
        return False

    # Check unreasonably large distances (> 1e6 km, beyond Moon)
    if np.any(r_mag > 1e6):
        if warn:
            print("Warning: State position > 1e6 km (beyond typical Earth orbit regime)")

    # Check velocity magnitude
    v_mag = np.linalg.norm(vel, axis=-1)

    # Escape velocity at Earth surface is ~11.2 km/s
    # At 1 Earth radius: v_esc = sqrt(2*mu/r)
    v_esc_surface = np.sqrt(2 * c.MU_EARTH / c.RADIUS_EARTH)

    if np.any(v_mag > 2 * v_esc_surface):
        if warn:
            print(f"Warning: Velocity magnitude > {2*v_esc_surface:.1f} km/s (very high)")

    return True


def validate_elements(elements: npt.ArrayLike, warn: bool = True) -> bool:
    """Validate physical constraints on orbital elements.

    Checks for:
    - NaN or Inf values
    - Negative eccentricity
    - Semi-major axis = 0
    - Inclination outside [0, pi]
    - Angles outside [0, 2*pi] (warns only, not failure)
    - Periapsis inside Earth radius

    Args:
        elements (npt.ArrayLike): N x 6 or 6-element array [a, e, i, Ω, ω, ν] in [km, -, rad, rad, rad, rad]
        warn (bool): If True, print warnings for constraint violations

    Returns:
        bool: True if elements are valid, False otherwise
    """
    elements_arr = np.atleast_2d(elements)

    # Check for NaN or Inf
    if np.any(np.isnan(elements_arr)) or np.any(np.isinf(elements_arr[elements_arr[:, 0] < 1e10])):
        # Allow inf for parabolic semi-major axis
        if warn:
            print("Warning: Elements contain NaN or invalid Inf values")
        return False

    a = elements_arr[:, 0]  # Semi-major axis
    e = elements_arr[:, 1]  # Eccentricity
    i = elements_arr[:, 2]  # Inclination
    O = elements_arr[:, 3]  # RAAN
    w = elements_arr[:, 4]  # Argument of perigee
    v = elements_arr[:, 5]  # True anomaly

    # Check eccentricity
    if np.any(e < 0):
        if warn:
            print("Warning: Eccentricity < 0 (non-physical)")
        return False

    # Check semi-major axis
    if np.any((a == 0) & (e != 1.0)):
        if warn:
            print("Warning: Semi-major axis = 0 for non-parabolic orbit")
        return False

    # Check inclination
    if np.any((i < 0) | (i > np.pi)):
        if warn:
            print("Warning: Inclination outside [0, π] range")
        return False

    # Check periapsis for bound orbits
    # r_p = a(1-e) for elliptical orbits
    elliptical_mask = (e < 1) & np.isfinite(a)
    if np.any(elliptical_mask):
        r_p = a[elliptical_mask] * (1 - e[elliptical_mask])
        if np.any(r_p < c.RADIUS_EARTH):
            if warn:
                print(f"Warning: Periapsis inside Earth radius ({c.RADIUS_EARTH} km)")
            return False

    # Warn about angles (but don't fail)
    if warn:
        if np.any((O < 0) | (O > 2*np.pi)):
            print("Warning: RAAN outside [0, 2π] range (not normalized)")
        if np.any((w < 0) | (w > 2*np.pi)):
            print("Warning: Argument of perigee outside [0, 2π] range (not normalized)")
        if np.any((v < 0) | (v > 2*np.pi)):
            print("Warning: True anomaly outside [0, 2π] range (not normalized)")

    return True
