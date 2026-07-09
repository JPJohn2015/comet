# python imports
from dataclasses import dataclass
import numpy as np

# --------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Constants:
    # TIME: Various time quantities
    MINUTE = 60         # [sec]
    HOUR = 3600         # [sec]
    DAY = 86400         # [sec]
    WEEK = 7            # [days]
    MONTH = 30          # [days]
    YEAR = 365.25       # [days]
    JULIAN_CENTURY = 36525

    # ANGLE: Angle conversions
    DEG2RAD = np.pi/180    
    RAD2DEG = 180/np.pi
    DEG2AS = 3600
    AS2DEG = 1/3600
    AS2RAD = AS2DEG*DEG2RAD
    RAD2AS = 1/AS2RAD

    # TIME SYSTEMS: Conversions between time systems
    UTC_TAI = 37        # [sec]
    TAI_UTC = -37       # [sec]
    TAI_TT = 32.184     # [sec]
    TT_TAI = -32.184    # [sec]
    UTC_TT = 69.184     # [sec]
    TT_UTC = -69.184    # [sec]

    # EPOCH: Epochs of various time systems
    UNIX0 = 2440587.5   # [days] January 1, 1970 GMT
    MJD0 = 2400000.5    # [days] November 17, 1858 GMT
    J2000 = 2451545.0   # [days] January 1, 2000 GMT
    MJ2000 = J2000 - MJD0

    # GRAVITY: Gravitational Parameters of planets
    MU_EARTH = 398600.4418      # [km^3/s^2]
    MU_MOON = 4904.86959        # [km^3/s^2]
    MU_SUN = 1.327124400189e11  # [km^3/s^2]

    # RADIUS: Radius of planets
    RADIUS_EARTH = 6378.14      # [km]
    RADIUS_MOON = 1737.4        # [km]
    RADIUS_SUN = 695700.0       # [km]

    # ASTRONOMY: Solar System constants
    AU = 1.495978707e8          # [km]

    # EARTH: Gravity properties
    SURFACE_GRAVITY = 9.807
    J2_EARTH = 0.001082
    J3_EARTH = -0.0000025
    OMEGA_EARTH = 7.292115e-5
    FLATTENING_EARTH = 298.257223563
    A_EARTH = 6378137.0
    B_EARTH = A_EARTH*(1 - 1/FLATTENING_EARTH)
    E_SQ_EARTH = 1 - (B_EARTH**2/A_EARTH**2)
    EP_SQ_EARTH = (A_EARTH**2 - B_EARTH**2)/B_EARTH**2

    # SUN: Solar properties
    SOLAR_PRESSURE = 4.57e-6

    # CALENDAR: Calendar Days
    DAYS_PER_MONTH = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    DAYS_PER_MONTH_LEAP = [0,31,29,31,30,31,30,31,31,30,31,30,31]



