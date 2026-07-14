# python imports
import numpy as np
import numpy.typing as npt

# comet imports
from comet.utilities.constants import Constants as c


def semimajor_axis_to_mean_motion(a: npt.ArrayLike, mu: float = c.MU_EARTH) -> npt.ArrayLike:
    """Calculates the Mean Motion from the Semi-Major Axis.

    Args:
        a (float): Semi-Major Axis in km.
        mu (float, optional): Gravitational Parameter in km^3/s^2. Defaults to Constants.MU_EARTH_KM3_S2.

    Returns:
        n (float): Mean Motion in rad/s
    """
    return np.sqrt(mu / ((a) ** 3))


def mean_motion_to_semimajor_axis(n: npt.ArrayLike, mu: float = c.MU_EARTH) -> npt.ArrayLike:
    """Calculates the Semi-Major Axis from the Mean Motion.

    Args:
        n (float): Mean Motion in rad/s
        mu (float, optional): Gravitational Parameter in km^3/s^2. Defaults to Constants.MU_EARTH_KM3_S2.

    Returns:
        a (float): Semi-Major Axis in km.
    """
    return (mu / (n**2)) ** (1 / 3)


def semimajoraxis_to_period(a: npt.ArrayLike, mu: float = c.MU_EARTH) -> npt.ArrayLike:
    """Calculates the Period from the Semi-Major Axis.

    Args:
        a (float): Semi-Major Axis in km.
        mu (float, optional): Gravitational Parameter in km^3/s^2. Defaults to Constants.MU_EARTH_KM3_S2.

    Returns:
        T (float): Period in sec.
    """
    return 2 * np.pi * np.sqrt((a**3) / mu)


def period_to_semimajor_axis(T: npt.ArrayLike, mu: float = c.MU_EARTH) -> npt.ArrayLike:
    """Calculates the Semi-Major Axis from the Period.

    Args:
        T (float): Period in sec.
        mu (float, optional): Gravitational Parameter in km^3/s^2. Defaults to Constants.MU_EARTH_KM3_S2.

    Returns:
        a (float): Semi-Major Axis in km.
    """
    return (mu * (T / (2 * np.pi)) ** 2) ** (1 / 3)


def eccentric_to_mean_anomaly(eccentric: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the Mean Anomaly from the Eccentric Anomaly (Kepler's equation).

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., pg. 65, Eq. 2-3

    Args:
        eccentric (npt.ArrayLike): Eccentric Anomaly in rad.
        ecc (npt.ArrayLike): Eccentricity.

    Returns:
        npt.ArrayLike: Mean Anomaly in rad.
    """
    return eccentric - ecc * np.sin(eccentric)


def mean_to_eccentric_anomaly(
    mean: npt.ArrayLike, ecc: npt.ArrayLike, max_iter: int = 100, tol: float = 1e-14
) -> npt.ArrayLike:
    """Calculates the Eccentric Anomaly from the Mean Anomaly using Newton-Raphson.

    Solves Kepler's equation: M = E - e*sin(E) for elliptical orbits (e < 1).

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Algorithm 2, pg. 65

    Args:
        mean (npt.ArrayLike): Mean Anomaly in rad.
        ecc (npt.ArrayLike): Eccentricity (must be < 1 for elliptical orbits).
        max_iter (int, optional): Maximum iterations for convergence. Defaults to 100.
        tol (float, optional): Convergence tolerance. Defaults to 1e-14.

    Returns:
        npt.ArrayLike: Eccentric Anomaly in rad.

    Raises:
        ValueError: If eccentricity >= 1 (use hyperbolic formulas for e >= 1).
        RuntimeError: If iteration does not converge within max_iter iterations.
    """
    mean = np.asarray(mean)
    ecc = np.asarray(ecc)

    # Check eccentricity range
    if np.any(ecc >= 1.0):
        raise ValueError(
            "Eccentricity must be < 1 for elliptical orbits. Use hyperbolic formulas for e >= 1."
        )

    # Initial guess (Vallado pg. 65)
    # For e < 0.8, use M as initial guess; otherwise use π
    eccentric = np.where(ecc < 0.8, mean, np.pi)

    # Newton-Raphson iteration
    for i in range(max_iter):
        f = eccentric - ecc * np.sin(eccentric) - mean
        fp = 1 - ecc * np.cos(eccentric)
        eccentric_new = eccentric - f / fp

        # Check convergence
        if np.all(np.abs(eccentric_new - eccentric) < tol):
            return eccentric_new

        eccentric = eccentric_new

    # If we get here, iteration did not converge
    raise RuntimeError(f"mean_to_eccentric_anomaly did not converge after {max_iter} iterations")


def true_to_eccentric_anomaly(true: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the Eccentric Anomaly from the True Anomaly for elliptical orbits.

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., pg. 82, Eq. 2-13

    Args:
        true (npt.ArrayLike): True Anomaly in rad.
        ecc (npt.ArrayLike): Eccentricity (must be < 1).

    Returns:
        npt.ArrayLike: Eccentric Anomaly in rad (0 to 2π).
    """
    # Use arctan2 for proper quadrant handling
    # E = arctan2(sqrt(1-e^2)*sin(ν), e+cos(ν))
    sqrt_term = np.sqrt(1 - ecc**2)
    sin_E = sqrt_term * np.sin(true) / (1 + ecc * np.cos(true))
    cos_E = (ecc + np.cos(true)) / (1 + ecc * np.cos(true))

    E = np.arctan2(sin_E, cos_E)

    # Ensure result is in [0, 2π]
    return np.where(E < 0, E + 2 * np.pi, E)


def eccentric_to_true_anomaly(eccentric: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the True Anomaly from the Eccentric Anomaly for elliptical orbits.

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., pg. 82, Eq. 2-13

    Args:
        eccentric (npt.ArrayLike): Eccentric Anomaly in rad.
        ecc (npt.ArrayLike): Eccentricity (must be < 1).

    Returns:
        npt.ArrayLike: True Anomaly in rad (0 to 2π).
    """
    # Use arctan2 for proper quadrant handling
    # ν = arctan2(sqrt(1-e^2)*sin(E), cos(E)-e)
    sqrt_term = np.sqrt(1 - ecc**2)
    sin_nu = sqrt_term * np.sin(eccentric) / (1 - ecc * np.cos(eccentric))
    cos_nu = (np.cos(eccentric) - ecc) / (1 - ecc * np.cos(eccentric))

    nu = np.arctan2(sin_nu, cos_nu)

    # Ensure result is in [0, 2π]
    return np.where(nu < 0, nu + 2 * np.pi, nu)


def true_to_mean_anomaly(true: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the Mean Anomaly from the True Anomaly.

    Args:
        true (float): True Anomaly in rad.
        ecc (float): Eccentricity.

    Returns:
        mean (float): Mean Anomaly in rad.
    """
    eccentric = true_to_eccentric_anomaly(true, ecc)
    return eccentric_to_mean_anomaly(eccentric, ecc)


def mean_to_true_anomaly(mean: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the True Anomaly from the Mean Anomaly.

    Args:
        mean (float): Mean Anomaly in rad.
        ecc (float): Eccentricity.

    Returns:
        true (float): True Anomaly in rad.
    """
    eccentric = mean_to_eccentric_anomaly(mean, ecc)
    return eccentric_to_true_anomaly(eccentric, ecc)


def time_to_perigee(
    true_anomaly: npt.ArrayLike, sma: npt.ArrayLike, ecc: npt.ArrayLike
) -> npt.ArrayLike:
    """Calculates the time to arrival at Perigee.

    Supports vectorized inputs (broadcasts over arrays).

    Args:
        true_anomaly (npt.ArrayLike): True Anomaly in rad.
        sma (npt.ArrayLike): Semi-Major Axis in km.
        ecc (npt.ArrayLike): Eccentricity.

    Returns:
        dt (npt.ArrayLike): Time To Perigee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    # Delta Mean Anomaly to Perigee (target = 0 or 2π)
    delta_mean = (2 * np.pi - mean_anomaly) % (2 * np.pi)
    # If already at perigee, return full period to next perigee (vectorized)
    delta_mean = np.where(delta_mean == 0.0, 2 * np.pi, delta_mean)

    return delta_mean / mean_motion


def time_since_perigee(
    true_anomaly: npt.ArrayLike, sma: npt.ArrayLike, ecc: npt.ArrayLike
) -> npt.ArrayLike:
    """Calculates the time since passing Perigee.

    Supports vectorized inputs (broadcasts over arrays).

    Args:
        true_anomaly (npt.ArrayLike): True Anomaly in rad.
        sma (npt.ArrayLike): Semi-Major Axis in km.
        ecc (npt.ArrayLike): Eccentricity.

    Returns:
        dt (npt.ArrayLike): Time Since Perigee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    return mean_anomaly / mean_motion


def time_to_apogee(
    true_anomaly: npt.ArrayLike, sma: npt.ArrayLike, ecc: npt.ArrayLike
) -> npt.ArrayLike:
    """Calculates the time to arrival at Apogee.

    Supports vectorized inputs (broadcasts over arrays).

    Args:
        true_anomaly (npt.ArrayLike): True Anomaly in rad.
        sma (npt.ArrayLike): Semi-Major Axis in km.
        ecc (npt.ArrayLike): Eccentricity.

    Returns:
        dt (npt.ArrayLike): Time To Apogee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    # Delta Mean Anomaly to Apogee (target = π)
    delta_mean = (np.pi - mean_anomaly) % (2 * np.pi)
    # If already at apogee, return full period to next apogee (vectorized)
    delta_mean = np.where(delta_mean == 0.0, 2 * np.pi, delta_mean)

    return delta_mean / mean_motion


def time_since_apogee(
    true_anomaly: npt.ArrayLike, sma: npt.ArrayLike, ecc: npt.ArrayLike
) -> npt.ArrayLike:
    """Calculates the time since passing Apogee.

    Supports vectorized inputs (broadcasts over arrays).

    Args:
        true_anomaly (npt.ArrayLike): True Anomaly in rad.
        sma (npt.ArrayLike): Semi-Major Axis in km.
        ecc (npt.ArrayLike): Eccentricity.

    Returns:
        dt (npt.ArrayLike): Time Since Apogee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    # Delta Mean Anomaly since Apogee (vectorized)
    delta_mean = mean_anomaly - np.pi
    delta_mean = np.where(delta_mean < 0.0, delta_mean + 2 * np.pi, delta_mean)

    return delta_mean / mean_motion
