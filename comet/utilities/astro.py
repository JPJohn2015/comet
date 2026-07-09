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
    """Calculates the Mean Anomaly from the Eccentric Anomaly.

    Args:
        ecentric (float): Eccentric Anomaly in rad.
        ecc (float): Eccentricity.

    Returns:
        mean (float): Mean Anomaly in rad.
    """
    return eccentric - ecc * np.sin(eccentric)


def mean_to_eccentric_anomaly(mean: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the Eccentric Anomaly from the Mean Anomaly.

    Args:
        mean (float): Mean Anomaly in rad.
        ecc (float): Eccentricity.

    Returns:
        ecentric (float): Eccentric Anomaly in rad.
    """
    # Iteratively solve for Eccentric Anomaly using Newton's Method
    eccentric = mean
    error = 1
    while error >= 1e-14:
        eccentric_new = eccentric - (eccentric - ecc * np.sin(eccentric) - mean) / (
            1 - ecc * np.cos(eccentric)
        )
        error = np.abs(eccentric_new - eccentric)
        eccentric = eccentric_new

    return eccentric


def true_to_eccentric_anomaly(true: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the Eccentric Anomaly from the True Anomaly.

    Args:
        true (float): True Anomaly in rad.
        ecc (float): Eccentricity.

    Returns:
        eccentric (float): Eccentric Anomaly in rad.
    """
    return 2 * np.arctan(np.sqrt((1 - ecc) / (1 + ecc)) * np.tan(true / 2))


def eccentric_to_true_anomaly(eccentric: npt.ArrayLike, ecc: npt.ArrayLike) -> npt.ArrayLike:
    """Calculates the True Anomaly from the Eccentric Anomaly.

    Args:
        eccentric (float): Eccentric Anomaly in rad.
        ecc (float): Eccentricity.

    Returns:
        true (float): True Anomaly in rad.
    """
    return 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(eccentric / 2))


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


def time_to_perigee(true_anomaly: float, sma: float, ecc: float):
    """Calculates the time to arrival at Perigee.

    Args:
        true_anomaly (float): True Anomaly in rad.
        sma (float): Semi-Major Axis in km.
        ecc (float): Eccentricity.

    Returns:
        dt (float): Time To Perigee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    # Delta Mean Anomaly to Perigee (2pi or 360 or 0 degrees)
    delta_mean = (2 * np.pi - mean_anomaly) % (2 * np.pi)

    return delta_mean / mean_motion


def time_since_perigee(true_anomaly: float, sma: float, ecc: float):
    """Calculates the time since passing Perigee.

    Args:
        true_anomaly (float): True Anomaly in rad.
        sma (float): Semi-Major Axis in km.
        ecc (float): Eccentricity.

    Returns:
        dt (float): Time Since Perigee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    return mean_anomaly / mean_motion


def time_to_apogee(true_anomaly: float, sma: float, ecc: float):
    """Calculates the time to arrival at Apogee.

    Args:
        true_anomaly (float): True Anomaly in rad.
        sma (float): Semi-Major Axis in km.
        ecc (float): Eccentricity.

    Returns:
        dt (float): Time To Apogee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    # Delta Mean Anomaly to Apogee (pi or 180 degrees)
    delta_mean = np.pi - mean_anomaly
    if delta_mean < 0.0:
        delta_mean += 2 * np.pi

    return delta_mean / mean_motion


def time_since_apogee(true_anomaly: float, sma: float, ecc: float):
    """Calculates the time since passing Apogee.

    Args:
        true_anomaly (float): True Anomaly in rad.
        sma (float): Semi-Major Axis in km.
        ecc (float): Eccentricity.

    Returns:
        dt (float): Time Since Apogee in sec.
    """
    # Calculate Mean Anomaly and Mean Motion
    mean_anomaly = true_to_mean_anomaly(true_anomaly, ecc)
    mean_motion = semimajor_axis_to_mean_motion(sma)

    # Delta Mean Anomaly to Apogee (pi or 180 degrees)
    delta_mean = mean_anomaly - np.pi
    if delta_mean < 0.0:
        delta_mean += 2 * np.pi

    return delta_mean / mean_motion
