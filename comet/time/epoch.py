# python imports
import numpy as np
from enum import Enum
from datetime import datetime
from calendar import isleap

# COMET imports
from comet.utilities.constants import Constants as c
from comet.time.time_conversions import *
from comet.time.duration import Duration


# ---------------------------------------------------------------------------------------------------------------------------
class TimeSystem(Enum):
    """Enum for different time systems.

    TimeSystem options are: UTC, UT1, TT, TAI
    """

    # TimeSystem Enums
    UTC = "UTC"
    UT1 = "UT1"
    TT = "TT"
    TAI = "TAI"


# --------------------------------------------------------------------------------------------------------------------------
class Epoch:
    """Class that represents a specific instance of time. Times should be provided in UTC and are stored
    interally in UTC.

    Example Constructions:
        * epoch = Epoch(year, month, day)
        * epoch = Epoch(year, month, day, hour, minute, second)
        * epoch = Epoch(julian_date)
        * epoch = Epoch(str)
    """

    # ----------------------------------------------------------------------------------------------------------------------
    # Class Construction
    # ----------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        year: int = 2000,
        month: int = 1,
        day: int = 1,
        hour: int = 12,
        minute: int = 0,
        second: int | float = 0.0,
    ):
        """Defines Epoch for specified times. Times should be provided in UTC and are stored
        interally in UTC.

        Construct from year, month, and day.

        Args:
            year (int, optional): Years. Defaults to 2000.
            month (int, optional): Months. Defaults to 0.
            days (int, optional): Days. Defaults to 0.

        Construct from year, month, day, hour, minute and seconds values.

        Args:
            year (int, optional): Years. Defaults to 2000.
            month (int, optional): Months. Defaults to 0.
            days (int, optional): Days. Defaults to 0.
            hours (int, optional): Hours. Defaults to 0.
            minutes (int, optional): Minutes. Defaults to 0.
            seconds (int|float, optional): Seconds. Defaults to 0.0.

        Construct from julian date.

        Args:
            julian_date (float): Julian Date.

        Construct from string representation.

        Args:
            days (str): String representation for Epoch formatted as YYYY-MM-DDTHH:MM:SS.sss
        """
        # Construct Class
        if isinstance(year, str):
            try:
                # String Construction
                ymd_hms = year.strip("Z").split("T")
                ymd = ymd_hms[0].split("-")
                hms = ymd_hms[1].split(":")

                # Store Epoch as Julian Date
                self._jd = ymdhms_to_jd(
                    int(ymd[0]), int(ymd[1]), int(ymd[2]), int(hms[0]), int(hms[1]), float(hms[2])
                )
            except:
                raise ValueError("Epoch(): Invalid string Epoch format: YYYY-MM-DDTHH:MM:SS.sss")
        elif isinstance(year, float | np.float64):
            # Julian Date construction
            self._jd = year
        else:
            # Normal Construction
            if not isinstance(year, int | np.int64):
                raise TypeError("Epoch(): year must be an integer")
            if not isinstance(month, int | np.int64):
                raise TypeError("Epoch(): month must be an integer")
            if not isinstance(day, int | np.int64):
                raise TypeError("Epoch(): day must be an integer")
            if not isinstance(hour, int | np.int64):
                raise TypeError("Epoch(): hour must be an integer")
            if not isinstance(minute, int | np.int64):
                raise TypeError("Epoch(): minute must be an integer")
            if not isinstance(second, int | float | np.int64 | np.float64):
                raise TypeError("Epoch(): second must be an integer or float")

            # Store Epoch as Julian Date
            self._jd = ymdhms_to_jd(year, month, day, hour, minute, second)

        # Calculate year, month, day, hour, minute and second as standard bounds
        self.__update()

    # ----------------------------------------------------------------------------------------------------------------------
    # Class Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def copy(self):
        """Returns a copy of the Epoch.

        Returns:
            Epoch: Copy of the Epoch.
        """
        return Epoch(self.year, self.month, self.day, self.hour, self.minute, self.second)

    # ----------------------------------------------------------------------------------------------------------------------
    def julian_date(self, timesystem: TimeSystem | str = TimeSystem.UTC):
        """Returns the Julian Date in the specified TimeSystem.

        Args:
            timesystem (TimeSystem|str): Time System to return Julian Date. Defaults to UTC.

        Returns:
            jd (float): Julian Date.
        """
        # Return Julian Date in the proper time system
        match timesystem:
            # UTC
            case TimeSystem.UTC | "utc" | "UTC":
                return self._jd
            # TT
            case TimeSystem.TT | "tt" | "TT":
                return self._jd + (c.UTC_TT / c.DAY)
            # TAI
            case TimeSystem.TAI | "tai" | "TAI":
                return self._jd + (c.UTC_TAI / c.DAY)
            # UT1
            case TimeSystem.UT1 | "ut1" | "UT1":
                raise NotImplementedError("Epoch(): UT1 has not been implemented")
            case _:
                raise Exception("Epoch(): Invalid TimeSystem")

    # ----------------------------------------------------------------------------------------------------------------------
    def modified_julian_date(self, timesystem: TimeSystem | str = TimeSystem.UTC):
        """Returns the Modified Julian Date in the specified TimeSystem.

        Args:
            timesystem (TimeSystem|str): Time System to return Modfied Julian Date. Defaults to UTC.

        Returns:
            mjd (float): Julian Date.
        """
        # Return Modified Julian Date in the proper time system
        match timesystem:
            # UTC
            case TimeSystem.UTC | "utc" | "UTC":
                return jd_to_mjd(self._jd)
            # TT
            case TimeSystem.TT | "tt" | "TT":
                return jd_to_mjd(self._jd) + (c.UTC_TT / c.DAY)
            # TAI
            case TimeSystem.TAI | "tai" | "TAI":
                return jd_to_mjd(self._jd) + (c.UTC_TAI / c.DAY)
            # UT1
            case TimeSystem.UT1 | "ut1" | "UT1":
                raise NotImplementedError("Epoch(): UT1 has not been implemented")
            case _:
                raise Exception("Epoch(): Invalid TimeSystem")

    # ----------------------------------------------------------------------------------------------------------------------
    def unix(self, timesystem: TimeSystem | str = TimeSystem.UTC):
        """Returns UNIX Seconds in the specified TimeSystem.

        Args:
            timesystem (TimeSystem|str): Time System to return UNIX Seconds. Defaults to UTC.

        Returns:
            unix (float): Unix Seconds.
        """
        # Return Modified Julian Date in the proper time system
        match timesystem:
            # UTC
            case TimeSystem.UTC | "utc" | "UTC":
                return jd_to_unix(self._jd)
            # TT
            case TimeSystem.TT | "tt" | "TT":
                return jd_to_unix(self._jd) + c.UTC_TT
            # TAI
            case TimeSystem.TAI | "tai" | "TAI":
                return jd_to_unix(self._jd) + c.UTC_TAI
            # UT1
            case TimeSystem.UT1 | "ut1" | "UT1":
                raise NotImplementedError("Epoch(): UT1 has not been implemented")
            case _:
                raise Exception("Epoch(): Invalid TimeSystem")

    # ----------------------------------------------------------------------------------------------------------------------
    def datetime(self):
        """Returns a Datetime object

        Returns:
            datetime (datetime): Datetime object.
        """
        # Calculate YMDHMS + Microseconds
        y, mo, d, h, m, s = jd_to_ymdhms(self._jd)
        ms = (s - np.floor(s)) * (10**6)

        return datetime(y, mo, d, h, m, int(np.floor(s)), int(ms))

    # ----------------------------------------------------------------------------------------------------------------------
    def day_of_year(self):
        """Returns the Day of Year.

        Returns:
            day_of_year (int): Day of Year.
        """
        if isleap(self.year):
            return np.cumsum(c.DAYS_PER_MONTH_LEAP[: (self.month - 1)]) + self.day
        else:
            return np.cumsum(c.DAYS_PER_MONTH[: (self.month - 1)]) + self.day

    # ----------------------------------------------------------------------------------------------------------------------
    def julian_centuries(self, timesystem: TimeSystem | str = TimeSystem.UTC):
        """Returns the Number of Julian Centuries since J2000 in the specified TimeSystem.

        Args:
            timesystem (TimeSystem|str): Time System to return Julian Centuries. Defaults to UTC.

        Returns:
            centuries (float): Julian Centuries since J2000.
        """
        # Calculate Julian Date in the proper time system
        match timesystem:
            # UTC
            case TimeSystem.UTC | "utc" | "UTC":
                jd = self._jd
            # TT
            case TimeSystem.TT | "tt" | "TT":
                jd = self._jd + (c.UTC_TT / c.DAY)
            # TAI
            case TimeSystem.TAI | "tai" | "TAI":
                jd = self._jd + (c.UTC_TAI / c.DAY)
            # UT1
            case TimeSystem.UT1 | "ut1" | "UT1":
                raise NotImplementedError("Epoch(): UT1 has not been implemented")
            case _:
                raise Exception("Epoch(): Invalid TimeSystem")

        return (jd - c.J2000) / c.JULIAN_CENTURY

    # ----------------------------------------------------------------------------------------------------------------------
    def to_dict(self):
        """Method that creates a dictionary of required inputs for Epoch construction.

        Returns:
            constructor (dict): dictionary of required inputs for Epoch construction.
        """
        return {"type": "Epoch", "jd": self._jd}

    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def from_dict(dict):
        """Method that creates an Epoch from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            epoch (Epoch): Epoch
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "Epoch":
            raise ValueError("Epoch(): Invalid construction dictionary")

        return Epoch(dict["jd"])

    # ----------------------------------------------------------------------------------------------------------------------
    def __update(self):
        """Recalculates year, month, day, hour, minute and second values after class is updated."""
        # Calculate YMDHMS
        year, month, day, hour, minute, second = jd_to_ymdhms(self._jd)
        ymdhms = self.__date_round([year, month, day, hour, minute, second])

        # Update parameters
        self.year = int(ymdhms[0])
        self.month = int(ymdhms[1])
        self.day = int(ymdhms[2])
        self.hour = int(ymdhms[3])
        self.minute = int(ymdhms[4])
        self.second = float(np.round(ymdhms[5], decimals=3))

    # ----------------------------------------------------------------------------------------------------------------------
    def __date_round(self, ymdhms: np.ndarray):
        """Recalculates overflow of time properties.

        Args:
            ymdhms (list): list of Year, Month, Day, Hour, Minute, Second.

        Returns:
            ymdhms (list): Properly Rounded list of Year, Month, Day, Hour, Minute, Second.
        """
        # Adjust overflow values
        ymdhms = date_round(ymdhms)

        return ymdhms

    # ----------------------------------------------------------------------------------------------------------------------
    # Operator Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, Epoch):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Epochs
        return self._jd == other._jd

    # ----------------------------------------------------------------------------------------------------------------------
    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, Epoch):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Epochs
        return self._jd != other._jd

    # ----------------------------------------------------------------------------------------------------------------------
    def __lt__(self, other) -> bool:
        """Override Less Than operator."""
        # Error checking
        if not isinstance(other, Epoch):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Epochs
        return self._jd < other._jd

    # ----------------------------------------------------------------------------------------------------------------------
    def __le__(self, other) -> bool:
        """Override Less Than or Equal To operator."""
        # Error checking
        if not isinstance(other, Epoch):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Epochs
        return self._jd <= other._jd

    # ----------------------------------------------------------------------------------------------------------------------
    def __gt__(self, other) -> bool:
        """Override Greater Than operator."""
        # Error checking
        if not isinstance(other, Epoch):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Epochs
        return self._jd > other._jd

    # ----------------------------------------------------------------------------------------------------------------------
    def __ge__(self, other) -> bool:
        """Override Greater Than or Equal To operator."""
        # Error checking
        if not isinstance(other, Epoch):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Epochs
        return self._jd >= other._jd

    # ----------------------------------------------------------------------------------------------------------------------
    def __iadd__(self, duration):
        """Override += operator for adding Durations to Epochs."""
        # Error checking
        if not isinstance(duration, Duration):
            raise NotImplementedError(
                f"Iterative Addition is not defined between Epochs and {type(duration)}"
            )

        # Add new Duration to current Epoch
        self._jd += duration._total_seconds / c.DAY
        self.__update()

        return self

    # ----------------------------------------------------------------------------------------------------------------------
    def __add__(self, duration):
        """Define Addition for adding Durations to Epochs."""
        # Error checking
        if not isinstance(duration, Duration):
            raise NotImplementedError(
                f"Addition is not defined between Epochs and {type(duration)}"
            )

        # Add new Duration to current Epoch
        jd = self._jd + duration._total_seconds / c.DAY

        return Epoch(jd)

    # ----------------------------------------------------------------------------------------------------------------------
    def __isub__(self, duration):
        """Override -= operator for subtracting Durations from Epochs."""
        # Error checking
        raise NotImplementedError(
            f"Iterative Subtraction is not defined between Epochs and {type(duration)}"
        )

    # ----------------------------------------------------------------------------------------------------------------------
    def __sub__(self, time):
        """Define Substraction for subtracting Durations from Epochs or Epochs from Epochs."""
        # Error checking
        if not isinstance(time, Duration | Epoch):
            raise NotImplementedError(
                f"Subtraction is not defined between Durations and {type(time)}"
            )

        # Subtract Epoch - Duration
        if isinstance(time, Duration):
            return Epoch(self._jd - time._total_seconds / c.DAY)
        # Subtract Epoch - Epoch
        elif isinstance(time, Epoch):
            dt = self._jd - time._jd
            return Duration(seconds=dt * c.DAY)

    # ----------------------------------------------------------------------------------------------------------------------
    def __imul__(self, value):
        """Override *= operator for multiplying Epochs."""
        # Error checking
        raise NotImplementedError(
            f"Iterative Multiplication is not defined between Epochs and {type(value)}"
        )

    # ----------------------------------------------------------------------------------------------------------------------
    def __mul__(self, value):
        """Define Multiplication for multiplying Epochs."""
        # Error checking
        raise NotImplementedError(f"Multiplication is not defined between Epochs and {type(value)}")

    # ----------------------------------------------------------------------------------------------------------------------
    def __rmul__(self, value):
        """Define Reverse Multiplication for multiplying Epochs."""
        # Error checking
        raise NotImplementedError(
            f"Reverse Multiplication is not defined between Epochs and {type(value)}"
        )

    # ----------------------------------------------------------------------------------------------------------------------
    def __truediv__(self, value):
        """Define Division for dividing Durations."""
        # Error checking
        raise NotImplementedError(f"Division is not defined between Epochs and {type(value)}")

    # ----------------------------------------------------------------------------------------------------------------------
    def __rtruediv__(self, value):
        """Define Reverse Division for dividing Durations."""
        # Error checking
        raise NotImplementedError(
            f"Reverse Division is not defined between Epochs and {type(value)}"
        )

    # ----------------------------------------------------------------------------------------------------------------------
    # Representation Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """String Representation of Epoch Class"""
        ymd = f"{self.month:02d}/{self.day:02d}/{self.year:04d}"
        hms = f"{self.hour:02d}:{self.minute:02d}:{self.second:06.3f}"
        return f"{ymd} {hms} UTC"

    # ----------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        """Class Representation of Epoch Class"""
        return f"Epoch({self.year}, {self.month}, {self.day}, {self.hour}, {self.minute}, {self.second:06.3f})"
