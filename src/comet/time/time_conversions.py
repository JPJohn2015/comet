# python imports
import numpy as np
import numpy.typing as npt
from calendar import isleap
from numba import njit

# COMET imports
from comet.utilities.constants import Constants as c


@njit(cache=True, inline='always')
def _isleap_numba(year: int) -> bool:
    """Numba-compatible leap year check.

    Args:
        year (int): Year to check

    Returns:
        bool: True if leap year, False otherwise
    """
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


@njit(cache=True)
def _ymdhms_to_jd_core(
    year: int, month: int, day: int, hour: int, minute: int, second: float
) -> float:
    """Core Julian Date calculation (numba-optimized).

    Args:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hour
        minute (int): Minute
        second (float): Second

    Returns:
        float: Julian Date
    """
    # Adjust for January/February (treat as months 13/14 of previous year)
    yr = year
    mo = month
    if mo <= 2:
        yr = yr - 1
        mo = mo + 12

    # Calculate Julian Date from YMDHMS
    a = int(yr / 100)
    b = 2 - a + int(a / 4)
    c = (((second / 60.0) + minute) / 60.0 + hour) / 24.0
    jd = int(365.25 * (yr + 4716)) + int(30.6001 * (mo + 1)) + day + b - 1524.5 + c

    return jd


def ymdhms_to_jd(
    year: int | npt.ArrayLike,
    month: int | npt.ArrayLike,
    day: int | npt.ArrayLike,
    hour: int | npt.ArrayLike = 0,
    minute: int | npt.ArrayLike = 0,
    second: float | npt.ArrayLike = 0.0
) -> float | npt.NDArray[np.float64]:
    """Converts gregorian calendar datetime to Julian Date.

    Accepts both scalar and array inputs. For array inputs, all arrays must
    have the same shape and will be broadcast together.

    Args:
        year: Year (scalar or array)
        month: Month (scalar or array)
        day: Day (scalar or array)
        hour: Hour (scalar or array). Defaults to 0.
        minute: Minute (scalar or array). Defaults to 0.
        second: Second (scalar or array). Defaults to 0.0.

    Returns:
        Julian Date (scalar or array)

    Source:
        Vallado, "Fundamentals of Astrodynamics and Applications", pg. 183

    Note:
        Uses numba @njit for ~10-50x speedup if numba is installed.
        Falls back to pure Python if numba unavailable.
    """
    # Handle scalar case
    if np.ndim(year) == 0:
        return _ymdhms_to_jd_core(
            int(year), int(month), int(day), int(hour), int(minute), float(second)
        )

    # Handle array case - vectorize over inputs
    year = np.atleast_1d(year)
    month = np.atleast_1d(month)
    day = np.atleast_1d(day)
    hour = np.atleast_1d(hour)
    minute = np.atleast_1d(minute)
    second = np.atleast_1d(second)

    # Broadcast arrays to same shape
    year, month, day, hour, minute, second = np.broadcast_arrays(
        year, month, day, hour, minute, second
    )

    # Vectorize computation
    result = np.empty(year.shape, dtype=np.float64)
    flat_year = year.ravel()
    flat_month = month.ravel()
    flat_day = day.ravel()
    flat_hour = hour.ravel()
    flat_minute = minute.ravel()
    flat_second = second.ravel()
    flat_result = result.ravel()

    for i in range(flat_year.size):
        flat_result[i] = _ymdhms_to_jd_core(
            int(flat_year[i]), int(flat_month[i]), int(flat_day[i]),
            int(flat_hour[i]), int(flat_minute[i]), float(flat_second[i])
        )

    return result


@njit(cache=True)
def _jd_to_ymdhms_core(jd: float):
    """Core JD to YMDHMS calculation (numba-optimized).

    Args:
        jd (float): Julian Date

    Returns:
        tuple: (year, month, day, hour, minute, second)
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

    # Determine if leap year and set month boundaries
    is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
    if is_leap:
        lmonth = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    else:
        lmonth = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])

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

    return year, month, day, hour, minute, second


def jd_to_ymdhms(jd: float | npt.ArrayLike):
    """Converts Julian Date to gregorian calendar datetime.

    Accepts both scalar and array inputs. For arrays, returns tuple of arrays.

    Args:
        jd: Julian Date (scalar or array)

    Returns:
        If scalar input:
            tuple: (year, month, day, hour, minute, second)
        If array input:
            tuple of arrays: (years, months, days, hours, minutes, seconds)

    Source:
        Vallado, "Fundamentals of Astrodynamics and Applications", pg. 184, Algorithm 22

    Note:
        Uses numba @njit for ~5-20x speedup if numba is installed.
        Falls back to pure Python if numba unavailable.
    """
    # Handle scalar case
    if np.ndim(jd) == 0:
        year, month, day, hour, minute, second = _jd_to_ymdhms_core(float(jd))
        return int(year), int(month), int(day), int(hour), int(minute), float(second)

    # Handle array case
    jd = np.atleast_1d(jd)
    n = jd.size
    years = np.empty(n, dtype=np.int32)
    months = np.empty(n, dtype=np.int32)
    days = np.empty(n, dtype=np.int32)
    hours = np.empty(n, dtype=np.int32)
    minutes = np.empty(n, dtype=np.int32)
    seconds = np.empty(n, dtype=np.float64)

    flat_jd = jd.ravel()
    for i in range(n):
        y, mo, d, h, m, s = _jd_to_ymdhms_core(flat_jd[i])
        years[i] = y
        months[i] = mo
        days[i] = d
        hours[i] = h
        minutes[i] = m
        seconds[i] = s

    # Reshape to match input shape
    if jd.ndim > 1:
        years = years.reshape(jd.shape)
        months = months.reshape(jd.shape)
        days = days.reshape(jd.shape)
        hours = hours.reshape(jd.shape)
        minutes = minutes.reshape(jd.shape)
        seconds = seconds.reshape(jd.shape)

    return years, months, days, hours, minutes, seconds


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


@njit(cache=True)
def _jd_to_gmst_core(jd_tt: float) -> float:
    """Core GMST calculation (numba-optimized).

    Args:
        jd_tt (float): Julian Date in Terrestrial Time

    Returns:
        float: Greenwich Mean Sidereal Time in degrees
    """
    # Constants inlined for numba compatibility
    J2000 = 2451545.0  # c.J2000
    JULIAN_CENTURY = 36525.0  # c.JULIAN_CENTURY

    # Calculate Julian Centuries
    T = (jd_tt - J2000) / JULIAN_CENTURY

    # Calculate GMST in arcseconds
    gmst_asec = (
        67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * (T**2) - 6.2e-6 * (T**3)
    )
    gmst = (gmst_asec / 240.0) % 360.0

    return gmst


def jd_to_gmst(jd_tt: float | npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    """Converts from Julian Date to Greenwich Mean Sidereal Time in degrees.

    Accepts both scalar and array inputs.

    Args:
        jd_tt: Julian Date in Terrestrial Time (scalar or array)

    Returns:
        Greenwich Mean Sidereal Time in degrees (scalar or array)

    Source:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed, pg. 188
    """
    # Handle scalar case
    if np.ndim(jd_tt) == 0:
        return _jd_to_gmst_core(float(jd_tt))

    # Handle array case - vectorize
    jd_tt = np.atleast_1d(jd_tt)
    result = np.empty(jd_tt.shape, dtype=np.float64)
    flat_jd = jd_tt.ravel()
    flat_result = result.ravel()

    for i in range(flat_jd.size):
        flat_result[i] = _jd_to_gmst_core(flat_jd[i])

    return result


def jd_to_lst(
    jd_tt: float | npt.ArrayLike,
    longitude: float | npt.ArrayLike
) -> float | npt.NDArray[np.float64]:
    """Converts from Julian Date to Local Sidereal Time in degrees.

    Accepts both scalar and array inputs. Arrays are broadcast together.

    Args:
        jd_tt: Julian Date in Terrestrial Time (scalar or array)
        longitude: Longitude in degrees (scalar or array)

    Returns:
        Local Sidereal Time in degrees (scalar or array)
    """
    # Calculate GMST
    gmst_deg = jd_to_gmst(jd_tt)

    # Handle scalar case
    if np.ndim(gmst_deg) == 0 and np.ndim(longitude) == 0:
        return (gmst_deg + longitude) % 360.0

    # Handle array case with broadcasting
    gmst_deg = np.atleast_1d(gmst_deg)
    longitude = np.atleast_1d(longitude)
    gmst_deg, longitude = np.broadcast_arrays(gmst_deg, longitude)

    # Calculate LST
    lst = (gmst_deg + longitude) % 360.0

    return lst


def unix_to_mjd(unix: npt.ArrayLike):
    """Converts from Unix Seconds to Modified Julian Date.

    Args:
        unix (npt.ArrayLike): Unix Seconds

    Returns:
        mjd (npt.ArrayLike): Modified Julian Date
    """
    jd = unix_to_jd(unix)
    return jd_to_mjd(jd)


def mjd_to_unix(mjd: npt.ArrayLike):
    """Converts from Modified Julian Date to Unix Seconds.

    Args:
        mjd (npt.ArrayLike): Modified Julian Date

    Returns:
        unix (npt.ArrayLike): Unix Seconds
    """
    jd = mjd_to_jd(mjd)
    return jd_to_unix(jd)


def ymdhms_to_mjd(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0
):
    """Converts gregorian calendar datetime to Modified Julian Date.

    Args:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int, optional): Hour. Defaults to 0.
        minute (int, optional): Minute. Defaults to 0.
        second (float, optional): Second. Defaults to 0.0.

    Returns:
        mjd (float): Modified Julian Date
    """
    jd = ymdhms_to_jd(year, month, day, hour, minute, second)
    return jd_to_mjd(jd)


def mjd_to_ymdhms(mjd: float):
    """Converts Modified Julian Date to gregorian calendar datetime.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hours
        minute (int): Minutes
        second (float): Seconds
    """
    jd = mjd_to_jd(mjd)
    return jd_to_ymdhms(jd)


def ymdhms_to_unix(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: float = 0.0
):
    """Converts gregorian calendar datetime to Unix Seconds.

    Args:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int, optional): Hour. Defaults to 0.
        minute (int, optional): Minute. Defaults to 0.
        second (float, optional): Second. Defaults to 0.0.

    Returns:
        unix (float): Unix Seconds
    """
    jd = ymdhms_to_jd(year, month, day, hour, minute, second)
    return jd_to_unix(jd)


def unix_to_ymdhms(unix: float):
    """Converts Unix Seconds to gregorian calendar datetime.

    Args:
        unix (float): Unix Seconds

    Returns:
        year (int): Year
        month (int): Month
        day (int): Day
        hour (int): Hours
        minute (int): Minutes
        second (float): Seconds
    """
    jd = unix_to_jd(unix)
    return jd_to_ymdhms(jd)


@njit(cache=True)
def _ymdhms_to_dayofyear_core(year: int, month: int, day: int) -> int:
    """Core day of year calculation (numba-optimized).

    Args:
        year (int): Year
        month (int): Month
        day (int): Day

    Returns:
        int: Day of Year (1-366)
    """
    if _isleap_numba(year):
        days_per_month = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    else:
        days_per_month = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])

    if month == 1:
        return day
    return days_per_month[month - 1] + day


def ymdhms_to_dayofyear(
    year: int | npt.ArrayLike,
    month: int | npt.ArrayLike,
    day: int | npt.ArrayLike
) -> int | npt.NDArray[np.int32]:
    """Converts Year/Month/Day to Day of Year.

    Accepts both scalar and array inputs.

    Args:
        year: Year (scalar or array)
        month: Month (scalar or array)
        day: Day (scalar or array)

    Returns:
        Day of Year (1-366) (scalar or array)
    """
    # Handle scalar case
    if np.ndim(year) == 0:
        return _ymdhms_to_dayofyear_core(int(year), int(month), int(day))

    # Handle array case
    year = np.atleast_1d(year)
    month = np.atleast_1d(month)
    day = np.atleast_1d(day)

    # Broadcast arrays to same shape
    year, month, day = np.broadcast_arrays(year, month, day)

    result = np.empty(year.shape, dtype=np.int32)
    flat_year = year.ravel()
    flat_month = month.ravel()
    flat_day = day.ravel()
    flat_result = result.ravel()

    for i in range(flat_year.size):
        flat_result[i] = _ymdhms_to_dayofyear_core(
            int(flat_year[i]), int(flat_month[i]), int(flat_day[i])
        )

    return result if result.size > 1 else result.item()


@njit(cache=True)
def _dayofyear_to_monthday_core(year: int, doy: int):
    """Core month/day calculation (numba-optimized).

    Args:
        year (int): Year
        doy (int): Day of Year (1-366)

    Returns:
        tuple: (month, day)
    """
    if _isleap_numba(year):
        days_per_month = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    else:
        days_per_month = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])

    month = 1
    for i in range(1, 13):
        if doy > days_per_month[i]:
            month = i + 1

    day = doy - days_per_month[month - 1]

    return month, day


def dayofyear_to_monthday(
    year: int | npt.ArrayLike,
    doy: int | npt.ArrayLike
):
    """Converts Day of Year to Month and Day.

    Accepts both scalar and array inputs.

    Args:
        year: Year (scalar or array)
        doy: Day of Year (1-366) (scalar or array)

    Returns:
        If scalar input:
            tuple: (month, day)
        If array input:
            tuple of arrays: (months, days)
    """
    # Handle scalar case
    if np.ndim(year) == 0:
        month, day = _dayofyear_to_monthday_core(int(year), int(doy))
        return int(month), int(day)

    # Handle array case
    year = np.atleast_1d(year)
    doy = np.atleast_1d(doy)

    # Broadcast arrays to same shape
    year, doy = np.broadcast_arrays(year, doy)

    n = year.size
    months = np.empty(n, dtype=np.int32)
    days = np.empty(n, dtype=np.int32)

    flat_year = year.ravel()
    flat_doy = doy.ravel()

    for i in range(n):
        mo, d = _dayofyear_to_monthday_core(int(flat_year[i]), int(flat_doy[i]))
        months[i] = mo
        days[i] = d

    # Reshape to match input shape
    if year.ndim > 1:
        months = months.reshape(year.shape)
        days = days.reshape(year.shape)

    return months, days
