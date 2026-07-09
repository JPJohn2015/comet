# python imports
import numpy as np
import numpy.typing as npt
from calendar import isleap

# COMET imports
from comet.utilities.constants import Constants as c


def ymdhms_to_jd(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0
):
    """Method that converts a gregorian calendar datetime to a julian date.

    Args:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int, optional): Hour. Defaults to 0.
        minute (int, optional): Minute. Defaults to 0.
        second (float, optional): Second. Defaults to 0.0.

    Returns:
        jd (float): Julian Date

    Source:
        Vallado, "Fundamentals of Astrodynamics and Applications", pg. 183
    """
    # Adjust for January/February (treat as months 13/14 of previous year)
    if month <= 2:
        year = year - 1
        month = month + 12

    # Calculate Julian Date from YMDHMS
    b_coeff = 2 - int(year / 100) + int(int(year / 100) / 4)
    c_coeff = (((second / 60) + minute) / 60 + hour) / 24
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b_coeff - 1524.5 + c_coeff

    return jd


def jd_to_ymdhms(jd: float):
    """Method that converts from julian date to gregorian calendar datetime.

    Args:
        jd (float): Julian Date

    Returns:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hours
        minute (int): Minutes
        second (float): Seconds

    Source:
        Vallado, "Fundamentals of Astrodynamics and Applications", pg. 184, Algorithm 22
    """
    # Separate integer and fractional parts
    T1900 = (jd - 2415019.5) / 365.25
    year = 1900 + int(T1900)
    leapyrs = int((year - 1900 - 1) / 4)
    days = (jd - 2415019.5) - ((year - 1900) * 365.0 + leapyrs)

    if days < 1.0:
        year = year - 1
        leapyrs = int((year - 1900 - 1) / 4)
        days = (jd - 2415019.5) - ((year - 1900) * 365.0 + leapyrs)

    # Determine if leap year
    if ((year % 4) == 0 and ((year % 100) != 0 or (year % 400) == 0)):
        lmonth = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
    else:
        lmonth = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    dayofyr = int(days)

    # Find month
    month = 1
    for i in range(1, 13):
        if dayofyr > lmonth[i]:
            month = i + 1

    day = dayofyr - lmonth[month - 1]

    # Calculate time
    tau = (days - dayofyr) * 24.0
    hour = int(tau)
    minute = int((tau - hour) * 60.0)
    second = (tau - hour - minute / 60.0) * 3600.0

    return int(year), int(month), int(day), int(hour), int(minute), float(second)


def date_round(ymdhms: np.ndarray) -> np.ndarray:
    """Recalculates overflow of time values.

    Args:
        ymdhms (list): list of Year, Month, Day, Hour, Minute, Second.

    Returns:
        ymdhms (list): Properly Rounded list of Year, Month, Day, Hour, Minute, Second.
    """
    # Check the significant figures on seconds, adjust values
    if ymdhms[5] >= 59.999:
        ymdhms[5] = 0.0
        ymdhms[4] += 1
    if ymdhms[4] == 60:
        ymdhms[4] = 0
        ymdhms[3] += 1
    if ymdhms[3] == 24:
        ymdhms[3] = 0
        ymdhms[2] += 1

    # Increment months
    if (ymdhms[1] in [1, 3, 5, 7, 8, 10, 12]) and (ymdhms[2] == 32):
        ymdhms[2] = 1
        ymdhms[1] += 1
    if (ymdhms[1] in [4, 6, 9, 11]) and (ymdhms[2] == 31):
        ymdhms[2] = 1
        ymdhms[1] += 1
    if (ymdhms[1] == 2) and (ymdhms[2] == 29) and (isleap(ymdhms[0]) == False):
        ymdhms[2] = 1
        ymdhms[1] += 1
    if (ymdhms[1] == 2) and (ymdhms[2] == 30) and (isleap(ymdhms[0]) == True):
        ymdhms[2] = 1
        ymdhms[1] += 1

    # Increment Years
    if ymdhms[1] == 13:
        ymdhms[1] = 1
        ymdhms[0] += 1

    return ymdhms


def jd_to_mjd(jd: npt.ArrayLike):
    """Converts from Julian Date to Modified Julian Date.

    Args:
        jd (npt.ArrayLike): Julian Date

    Returns:
        mjd (npt.ArrayLike): Modified Julian Date
    """
    return jd - c.MJD0


def mjd_to_jd(mjd: npt.ArrayLike):
    """Converts from Modified Julian Date to Julian Date.

    Args:
        mjd (npt.ArrayLike): Modified Julian Date

    Returns:
        jd (npt.ArrayLike): Julian Date
    """
    return mjd + c.MJD0


def unix_to_jd(unix: npt.ArrayLike):
    """Converts from Unix Seconds to Julian Date.

    Args:
        unix (npt.ArrayLike): Unix Seconds

    Returns:
        jd (npt.ArrayLike): Julian Date
    """
    return (unix / c.DAY) + c.UNIX0


def jd_to_unix(jd: npt.ArrayLike):
    """Converts from Julian Date to Unix Seconds.

    Args:
        jd (npt.ArrayLike): Julian Date
    Returns:
        unix (npt.ArrayLike): Unix Seconds
    """
    return (jd - c.UNIX0) * c.DAY


def jd_to_gmst(jd_tt: npt.ArrayLike):
    """Converts from Julian Date to Greenwich Mean Sidereal Time in degrees.

    Args:
        jd_tt (npt.ArrayLike): Julian Date in Terrestrial Time.

    Returns:
        gmst (npt.ArrayLike): Greenwich Mean Sidereal Time in degrees.
    """
    # Calculate Julian Centuries
    T = (jd_tt - c.J2000) / c.JULIAN_CENTURY

    # Calculate GMST
    gmst_asec = (
        67310.54841 + (876600 * 3600 + 864 - 184.812866) * T + 0.093104 * (T**2) - 6.2e-6 * (T**3)
    )
    gmst = np.mod(gmst_asec / 240, 360)

    return gmst


def jd_to_lst(jd_tt: npt.ArrayLike, longitude: npt.ArrayLike):
    """Converts from Julian Date to Local Sidereal Time in degrees.

    Args:
        jd_tt (npt.ArrayLike): Julian Date in Terrestrial Time.
        longitude (npt.ArrayLike): Longitude in degrees.

    Returns:
        lst (npt.ArrayLike): Local Sidereal Time in degrees.
    """
    # Calculate GMST
    gmst_deg = jd_to_gmst(jd_tt)

    # Calculate LST
    lst_deg = gmst_deg + longitude
    lst = lst_deg % 360.0

    return lst
