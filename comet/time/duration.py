# python imports
import numpy as np

# COMET imports
from comet.utilities.constants import Constants as c


class Duration:
    """Class that represents a duration of time.

    Example Constructions:
        * duration = Duration(days, hours, minutes, seconds)
        * duration = Duration(str)
    """

    def __init__(self, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int | float = 0.0):
        """Defines Duration for specified times.

        Construct from days, hours, minutes and/or seconds values.

        Args:
            days (int, optional): Days. Defaults to 0.
            hours (int, optional): Hours. Defaults to 0.
            minutes (int, optional): Minutes. Defaults to 0.
            seconds (int | float, optional): Seconds. Defaults to 0.0.

        Construct from string representation.

        Args:
            days (str): String representation for Duration formatted as DDTHH:MM:SS.sss
        """
        # Construct Class
        if isinstance(days, str):
            try:
                # String Construction
                d_hms = days.split("T")
                hms = d_hms[1].split(":")

                # Store Duration as total seconds
                self._total_seconds = np.round(
                    int(d_hms[0]) * c.DAY
                    + int(hms[0]) * c.HOUR
                    + int(hms[1]) * c.MINUTE
                    + float(hms[2]),
                    decimals=3,
                )
            except:
                raise ValueError("Duration(): Invalid string duration format: DDTHH:MM:SS.sss")
        else:
            # Normal Construction
            if not isinstance(days, int | np.int64):
                raise TypeError("Duration(): days must be an integer")
            if not isinstance(hours, int | np.int64):
                raise TypeError("Duration(): hours must be an integer")
            if not isinstance(minutes, int | np.int64):
                raise TypeError("Duration(): minutes must be an integer")
            if not isinstance(seconds, int | float | np.int64 | np.float64):
                raise TypeError("Duration(): seconds must be an integer or float")

            # Store Duration as total seconds
            self._total_seconds = np.round(
                days * c.DAY + hours * c.HOUR + minutes * c.MINUTE + seconds, decimals=3
            )

        # Calculate days, hours, minutes and seconds as standard bounds
        self.__update()

        # Determine if Duration is Negative
        self.is_negative = self._total_seconds < 0

    def copy(self):
        """Returns a copy of the Duration.

        Returns:
            Duration: Copy of the Duration.
        """
        return Duration(seconds=self._total_seconds)

    def total_days(self) -> float:
        """Returns total number of days in Duration.

        Returns:
            float: Total number of Days.
        """
        return self._total_seconds / c.DAY

    def total_hours(self) -> float:
        """Returns total number of hours in Duration.

        Returns:
            float: Total number of Hours.
        """
        return self._total_seconds / c.HOUR

    def total_minutes(self) -> float:
        """Returns total number of minutes in Duration.

        Returns:
            float: Total number of Minutes.
        """
        return self._total_seconds / c.MINUTE

    def total_seconds(self) -> float:
        """Returns total number of seconds in Duration.

        Returns:
            float: Total number of Seconds.
        """
        return self._total_seconds

    def to_dict(self):
        """Method that creates a dictionary of required inputs for Duration construction.

        Returns:
            constructor (dict): dictionary of required inputs for Duration construction.
        """
        return {"type": "Duration", "total_seconds": float(self._total_seconds)}

    @staticmethod
    def from_dict(dict):
        """Method that creates a Duration from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            duration (Duration): Duration
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "Duration":
            raise ValueError("Duration(): Invalid construction dictionary")

        return Duration(seconds=dict["total_seconds"])

    def __update(self):
        """Recalculates days, hours, minutes and seconds values after class is updated."""
        # Reduce Duration time values to standard bounds
        self.days, remainder = np.divmod(np.abs(self._total_seconds), c.DAY)
        self.hours, remainder = np.divmod(remainder, c.HOUR)
        self.minutes, self.seconds = np.divmod(remainder, c.MINUTE)

        # Check the significant figures on seconds, adjust values
        if self.seconds >= 59.999:
            self.seconds = 0.0
            self.minutes += 1
        if self.minutes == 60:
            self.minutes = 0
            self.hours += 1
        if self.hours == 24:
            self.hours = 0
            self.days += 1

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, Duration):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Durations
        return self._total_seconds == other._total_seconds

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, Duration):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Durations
        return self._total_seconds != other._total_seconds

    def __lt__(self, other) -> bool:
        """Override Less Than operator."""
        # Error checking
        if not isinstance(other, Duration):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Durations
        return self._total_seconds < other._total_seconds

    def __le__(self, other) -> bool:
        """Override Less Than or Equal To operator."""
        # Error checking
        if not isinstance(other, Duration):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Durations
        return self._total_seconds <= other._total_seconds

    def __gt__(self, other) -> bool:
        """Override Greater Than operator."""
        # Error checking
        if not isinstance(other, Duration):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Durations
        return self._total_seconds > other._total_seconds

    def __ge__(self, other) -> bool:
        """Override Greater Than or Equal To operator."""
        # Error checking
        if not isinstance(other, Duration):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Durations
        return self._total_seconds >= other._total_seconds

    def __iadd__(self, duration):
        """Override += operator for adding Durations."""
        # Error checking
        if not isinstance(duration, Duration):
            raise NotImplementedError(
                f"Iterative Addition is not defined between Durations and {type(duration)}"
            )

        # Add new duration to current duration
        self._total_seconds += duration._total_seconds
        self.__update()

        return self

    def __add__(self, duration):
        """Define Addition for adding Durations."""
        # Error checking
        if not isinstance(duration, Duration):
            raise NotImplementedError(
                f"Addition is not defined between Durations and {type(duration)}"
            )

        # Add total seconds
        total_seconds = self._total_seconds + duration._total_seconds

        return Duration(seconds=total_seconds)

    def __isub__(self, duration):
        """Override -= operator for subtracting Durations."""
        # Error checking
        if not isinstance(duration, Duration):
            raise NotImplementedError(
                f"Iterative Subtraction is not defined between Durations and {type(duration)}"
            )

        # Subtract new duration to current duration
        self._total_seconds -= duration._total_seconds
        self.__update()

        return self

    def __sub__(self, duration):
        """Define Subtraction for subtracting Durations."""
        # Error checking
        if not isinstance(duration, Duration):
            raise NotImplementedError(
                f"Subtraction is not defined between Durations and {type(duration)}"
            )

        # Subtract total seconds
        total_seconds = self._total_seconds - duration._total_seconds

        return Duration(seconds=total_seconds)

    def __imul__(self, value):
        """Override *= operator for multiplying Durations."""
        # Error checking
        if not isinstance(value, Duration | float | int):
            raise NotImplementedError(
                f"Iterative Multiplication is not defined between Durations and {type(value)}"
            )

        # Multiply new duration to current duration
        if isinstance(value, Duration):
            self._total_seconds = self._total_seconds * value._total_seconds
        elif isinstance(value, float | int):
            self._total_seconds = self._total_seconds * value
        self.__update()

        return self

    def __mul__(self, value):
        """Define Multiplication for multiplying Durations."""
        # Error checking
        if not isinstance(value, Duration | float | int):
            raise NotImplementedError(
                f"Multiplication is not defined between Durations and {type(value)}"
            )

        # Multiply new duration to current duration
        if isinstance(value, Duration):
            total_seconds = self._total_seconds * value._total_seconds
        elif isinstance(value, float | int):
            total_seconds = self._total_seconds * value

        return Duration(seconds=total_seconds)

    def __rmul__(self, value):
        """Define Reverse Multiplication for multiplying Durations."""
        # Error checking
        if not isinstance(value, Duration | float | int):
            raise NotImplementedError(
                f"Reverse Multiplication is not defined between Durations and {type(value)}"
            )

        # Multiply new duration to current duration
        if isinstance(value, Duration):
            total_seconds = self._total_seconds * value._total_seconds
        elif isinstance(value, float | int):
            total_seconds = self._total_seconds * value

        return Duration(seconds=total_seconds)

    def __truediv__(self, value):
        """Define Division for dividing Durations."""
        # Error checking
        if not isinstance(value, Duration | float | int):
            raise NotImplementedError(
                f"Division is not defined between Durations and {type(value)}"
            )

        # Divide Durations
        if isinstance(value, Duration):
            return self._total_seconds / value._total_seconds

        elif isinstance(value, float | int):
            return Duration(seconds=self._total_seconds / value)

    def __rtruediv__(self, value):
        """Define Reverse Division for dividing Durations."""
        # Error checking
        if not isinstance(value, Duration | float | int):
            raise NotImplementedError(
                f"Riverse Division is not defined between Durations and {type(value)}"
            )

        # Divide Durations
        if isinstance(value, Duration):
            return value._total_seconds / self._total_seconds
        elif isinstance(value, float | int):
            return Duration(seconds=value / self._total_seconds)

    def __str__(self):
        """String Representation of Duration Class"""
        if self.is_negative:
            return f"-{int(self.days)}d {int(self.hours)}h {int(self.minutes)}m {float(self.seconds):06.3f}s"
        else:
            return f"{int(self.days)}d {int(self.hours)}h {int(self.minutes)}m {float(self.seconds):06.3f}s"

    def __repr__(self):
        """Class Representation of Duration Class"""
        if self.is_negative:
            return f"Duration(-{int(self.days)}, -{int(self.hours)}, -{int(self.minutes)}, -{float(self.seconds):06.3f})"
        else:
            return f"Duration({int(self.days)}, {int(self.hours)}, {int(self.minutes)}, {float(self.seconds):06.3f})"
