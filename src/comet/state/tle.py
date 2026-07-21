# python imports
import numpy as np

# COMET imports
from comet.state.state_conversions import elements_to_cartesian
from comet.utilities.astro import (
    mean_motion_to_semimajor_axis,
    mean_to_true_anomaly,
    semimajoraxis_to_period,
    mean_to_eccentric_anomaly,
)
from comet.utilities.constants import Constants as c
from comet.time.epoch import Epoch
from comet.state.state import State
from comet.state.elements import Elements
from comet.time.time_conversions import ymdhms_to_jd


class TLE:
    """Class that represents a Two-Lined Element Set.

    Example Constructions:
        * tle = TLE(line1, line2)
        * tle = TLE([line1, line2])
    """

    def __init__(self, *args):
        """Defines a Two-Lined Element Set.

        Construct from two TLE lines as strings.

        Args:
            line1 (str): TLE Line 1.
            line2 (str): TLE Line 2.

        Construct from a list/tuple of two TLE lines.

        Args:
            lines (list|tuple): List of two TLE lines.
        """
        if len(args) == 2:
            if not isinstance(args[0], str) or not isinstance(args[1], str):
                raise TypeError("TLE(): Both lines must be strings")
            self._line1 = args[0].strip()
            self._line2 = args[1].strip()
        elif len(args) == 1:
            if not isinstance(args[0], (list, tuple)) or len(args[0]) != 2:
                raise TypeError("TLE(): Input must be a list/tuple of 2 TLE lines")
            if not isinstance(args[0][0], str) or not isinstance(args[0][1], str):
                raise TypeError("TLE(): Both lines must be strings")
            self._line1 = args[0][0].strip()
            self._line2 = args[0][1].strip()
        else:
            raise ValueError("TLE(): Invalid number of inputs")

        # Basic validation
        if len(self._line1) < 69:
            raise ValueError("TLE(): Line 1 must be at least 69 characters")
        if len(self._line2) < 69:
            raise ValueError("TLE(): Line 2 must be at least 69 characters")
        if self._line1[0] != '1':
            raise ValueError("TLE(): Line 1 must start with '1'")
        if self._line2[0] != '2':
            raise ValueError("TLE(): Line 2 must start with '2'")

    def copy(self):
        """Returns a copy of the TLE.

        Returns:
            tle (TLE): Copy of the TLE.
        """
        return TLE(self._line1, self._line2)

    @property
    def line1(self) -> str:
        """TLE line 1."""
        return self._line1

    @property
    def line2(self) -> str:
        """TLE line 2."""
        return self._line2

    @property
    def lines(self) -> tuple:
        """Both TLE lines as a tuple."""
        return (self._line1, self._line2)

    def norad_id(self) -> int:
        """Returns the NORAD ID.

        Returns:
            norad (int): NORAD ID.
        """
        return int(self._line1[2:7])

    def classification(self) -> str:
        """Returns the TLE Classification.

        Returns:
            classification (str): Classification.
        """
        return self._line1[7]

    def launch_year(self) -> int:
        """Returns the Object Launch Year.

        Returns:
            launch_year (int): Launch Year.
        """
        return int(self._line1[8:11])

    def launch_number(self) -> int:
        """Returns the Object Launch Number.

        Returns:
            launch_number (int): Launch Number.
        """
        return int(self._line1[11:14])

    def designator(self) -> str:
        """Returns the Object International Designator.

        Returns:
            designator (str): International Designator.
        """
        return self._line1[14]

    def year(self) -> int:
        """Returns the TLE Epoch Year.

        Returns:
            year (int): TLE Epoch Year.
        """
        return int(self._line1[18:20])

    def day_of_year(self) -> float:
        """Returns the TLE Epoch Day of Year.

        Returns:
            day_of_year (float): TLE Epoch Day of Year.
        """
        return float(self._line1[20:32])

    def epoch(self):
        """Returns the TLE Epoch.

        Returns:
            epoch (Epoch): TLE Epoch.
        """
        # Calculate Julian Date and convert to Epoch
        # TLE years are 2-digit: 57-99 = 1957-1999, 00-56 = 2000-2056
        two_digit_year = self.year()
        if two_digit_year >= 57:
            full_year = 1900 + two_digit_year
        else:
            full_year = 2000 + two_digit_year

        jd_year_start = ymdhms_to_jd(full_year, 1, 1, 0, 0, 0.0)
        jd = jd_year_start + self.day_of_year() - 1
        return Epoch(jd)

    def inclination(self) -> float:
        """Returns the TLE Inclination in rad.

        Returns:
            inc (float): TLE Inclination in rad.
        """
        return np.deg2rad(float(self._line2[9:16]))

    def right_ascension(self) -> float:
        """Returns the TLE Right Ascension of Ascending Node in rad.

        Returns:
            raan (float): TLE Right Ascension of Ascending Node in rad.
        """
        return np.deg2rad(float(self._line2[16:25]))

    def eccentricity(self) -> float:
        """Returns the TLE Eccentricity.

        Returns:
            ecc (float): TLE Eccentricity.
        """
        return float("0." + self._line2[26:34])

    def argument_perigee(self) -> float:
        """Returns the TLE Argument of Perigee in rad.

        Returns:
            ap (float): TLE Argument of Perigee in rad.
        """
        return np.deg2rad(float(self._line2[34:42]))

    def mean_anomaly(self) -> float:
        """Returns the TLE Mean Anomaly in rad.

        Returns:
            ma (float): TLE Mean Anomaly in rad.
        """
        return np.deg2rad(float(self._line2[43:51]))

    def true_anomaly(self) -> float:
        """Returns the TLE True Anomaly in rad.

        Returns:
            ta (float): TLE True Anomaly in rad.
        """
        result = mean_to_true_anomaly(self.mean_anomaly(), self.eccentricity())
        return float(np.asarray(result).item())

    def eccentric_anomaly(self) -> float:
        """Returns the TLE Eccentric Anomaly in rad.

        Returns:
            ea (float): TLE Eccentric Anomaly in rad.
        """
        result = mean_to_eccentric_anomaly(self.mean_anomaly(), self.eccentricity())
        return float(np.asarray(result).item())

    def mean_motion(self) -> float:
        """Returns the TLE Mean Motion in rad/s.

        Returns:
            n (float): TLE Mean Motion in rad/s.
        """
        return float(self._line2[52:63]) * (2 * np.pi) / c.DAY

    def semimajor_axis(self) -> float:
        """Returns the TLE Semimajor Axis in km.

        Returns:
            sma (float): TLE Semimajor Axis in km.
        """
        return mean_motion_to_semimajor_axis(self.mean_motion())

    def period(self) -> float:
        """Returns the TLE Period in sec.

        Returns:
            period (float): TLE Period in sec.
        """
        return semimajoraxis_to_period(self.semimajor_axis())

    def to_state(self):
        """Returns the Cartesian State Representation

        Returns:
            state (State): Cartesian State.
        """
        element_array = np.array([
            self.semimajor_axis(),
            self.eccentricity(),
            self.inclination(),
            self.right_ascension(),
            self.argument_perigee(),
            self.true_anomaly(),
        ])
        return State(elements_to_cartesian(element_array))

    def to_elements(self):
        """Returns the Classical Orbital Elements Representation.

        Returns:
            elements (Elements): Classical Orbital Elements.
        """
        element_array = [
            self.semimajor_axis(),
            self.eccentricity(),
            self.inclination(),
            self.right_ascension(),
            self.argument_perigee(),
            self.true_anomaly(),
        ]
        return Elements(element_array)

    def to_dict(self):
        """Method that creates a dictionary of required inputs for TLE construction.

        Returns:
            constructor (dict): dictionary of required inputs for TLE construction.
        """
        return {"type": "TLE", "line1": self._line1, "line2": self._line2}

    @staticmethod
    def from_dict(dict):
        """Method that creates a TLE from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            tle (TLE): TLE
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "TLE":
            raise ValueError("TLE(): Invalid construction dictionary")

        return TLE(dict["line1"], dict["line2"])

    def __hash__(self):
        """Make TLE hashable for use in sets and as dictionary keys."""
        return hash((self._line1, self._line2))

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, TLE):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare TLEs
        return self._line1 == other._line1 and self._line2 == other._line2

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, TLE):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare TLEs
        return self._line1 != other._line1 or self._line2 != other._line2

    def __lt__(self, other) -> bool:
        """Override Less Than operator."""
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __le__(self, other) -> bool:
        """Override Less Than or Equal To operator."""
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __gt__(self, other) -> bool:
        """Override Greater Than operator."""
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __ge__(self, other) -> bool:
        """Override Greater Than or Equal To operator."""
        raise NotImplementedError(f"Comparison is not defined for {type(other)}")

    def __str__(self):
        """String Representation of TLE Class"""
        oe = [
            self.semimajor_axis(),
            self.eccentricity(),
            self.inclination(),
            self.right_ascension(),
            self.argument_perigee(),
            self.true_anomaly(),
        ]
        oe[2:] = np.degrees(oe[2:])
        return f"{oe[0]:9.3f} km, {oe[1]:07.6f}, {oe[2]:6.2f}{chr(176)}, {oe[3]:6.2f}{chr(176)}, {oe[4]:6.2f}{chr(176)}, {oe[5]:6.2f}{chr(176)}"

    def __repr__(self):
        """Class Representation of TLE Class"""
        return f"TLE({self._line1}, {self._line2})"


# Testing
if __name__ == "__main__":
    line1 = "1 59069U 24040A   24153.34215875  .00000159  00000-0  00000-0 0  9994"
    line2 = "2 59069   0.0491 104.1558 0003000 296.0103   6.9785  1.00271391    57"
    tle = TLE(line1, line2)
