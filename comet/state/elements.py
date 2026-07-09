# python imports
import numpy as np

# COMET imports
from comet.utilities.constants import Constants as c
from comet.state.state_conversions import elements_to_cartesian
from comet.utilities.astro import true_to_mean_anomaly, true_to_eccentric_anomaly


class Elements:
    """Class that represents a Classical Orbital Elements Set.

    Example Constructions:
        * elements = Elements(a, e, i, O, w, v)
        * elements = Elements([a, e, i, O, w, v])
    """

    def __init__(self, *args):
        """Defines a Classical Orbital Elements Set.

        Construct from 6 orbital elements km and radians.

        Args:
            a (int|float): Semi-Major Axis in km.
            e (int|float): Eccentricity.
            i (int|float): Inclination in rad.
            O (int|float): Right Ascension of Ascending Node in rad.
            w (int|float): Argument of Perigee in rad.
            v (int|float): True Anomaly in rad.

        Construct from a orbital elements array in km and radians.

        Args:
            elements (list|np.ndarray): Orbital Elements array.
        """
        # Construct Class
        if len(args) == 6:
            # 6 individual components of the Classical Orbital Elements Set
            if not np.all([isinstance(arg, int | float | np.int64 | np.float64) for arg in args]):
                raise TypeError(
                    "Elements(): values must be an integer or float when providing 6 inputs"
                )
            self._raw = np.array(args)

        elif len(args) == 1:
            # Full Elements Vector
            if not isinstance(args[0], list | np.ndarray):
                raise TypeError("Elements(): Orbital Elements must be an Array of length 6")
            if len(args[0]) != 6:
                raise TypeError("Elements(): Orbital Elements must be an Array of length 6")
            if not np.all(
                [isinstance(arg, int | float | np.int64 | np.float64) for arg in args[0]]
            ):
                raise TypeError("Elements(): Values inside State array must be an integer or float")
            self._raw = np.array(args[0])

        else:
            raise ValueError("Elements(): Invalid number of inputs")

    def copy(self):
        """Returns a copy of the Elements.

        Returns:
            elements (Elements): Copy of the Elements.
        """
        return Elements(self._raw)

    def semi_major_axis(self) -> float:
        """Returns the Semi-Major Axis of the Elements Set.

        Returns:
            sma (float): Semi-Major Axis in km.
        """
        return self._raw[0]

    def eccentricity(self) -> float:
        """Returns the Eccentricity of the Elements Set.

        Returns:
            ecc (float): Eccentricity.
        """
        return self._raw[1]

    def inclination(self) -> float:
        """Returns the Inclination of the Elements Set.

        Returns:
            inc (float): Inclination in rad.
        """
        return self._raw[2]

    def right_ascension(self) -> float:
        """Returns the Right Ascension of Ascending Node of the Elements Set.

        Returns:
            raan (float): Right Ascension of Ascending Node in rad.
        """
        return self._raw[3]

    def argument_perigee(self) -> float:
        """Returns the Argument of Perigee of the Elements Set.

        Returns:
            ap (float): Argument of Perigee in rad.
        """
        return self._raw[4]

    def true_anomaly(self) -> float:
        """Returns the True Anomaly of the Elements Set.

        Returns:
            ta (float): True Anomaly in rad.
        """
        return self._raw[5]

    def mean_anomaly(self) -> float:
        """Returns the Mean Anomaly of the Elements Set.

        Returns:
            ma (float): Mean Anomaly in rad.
        """
        return true_to_mean_anomaly(self._raw[5], self._raw[1])

    def eccentric_anomaly(self) -> float:
        """Returns the Eccentric Anomaly of the Elements Set.

        Returns:
            ea (float): Eccentric Anomaly in rad.
        """
        return true_to_eccentric_anomaly(self._raw[5], self._raw[1])

    def flight_path_angle(self) -> float:
        """Returns the Flight Path Angle of the Elements Set.

        Returns:
            fpa (float): Flight Path Angle in rad.
        """
        return np.arctan(
            (self._raw[1] * np.sin(self._raw[5])) / (1 + self._raw[1] * np.cos(self._raw[5]))
        )

    def longitude_of_pariapsis(self) -> float:
        """Returns the Longitude of Pariapsis of the Elements Set.

        Returns:
            lp (float): Longitude of Pariapsis in rad.
        """
        return self._raw[3] + self._raw[4]

    def true_longitude(self) -> float:
        """Returns the True Longitude of the Elements Set.

        Returns:
            tl (float): True Longitude in rad.
        """
        return self._raw[3] + self._raw[4] + self._raw[5]

    def apogee(self) -> float:
        """Returns the Orbital Apogee in km.

        Returns:
            apogee (float): Orbital Apogee.
        """
        return self._raw[0] * (1 + self._raw[1])

    def perigee(self) -> float:
        """Returns the Orbital Perigee in km.

        Returns:
            perigee (float): Orbital Perigee.
        """
        return self._raw[0] * (1 - self._raw[1])

    def semi_parameter(self) -> float:
        """Returns the Orbital Semi-Parameter in km.

        Returns:
            p (float): Semi-Parameter.
        """
        return self._raw[0] * (1 - self._raw[1] ** 2)

    def specific_energy(self) -> float:
        """Returns the Orbital Specific Energy in km^2/s^2.

        Returns:
            specific_energy (float): Specific Energy.
        """
        return -c.MU_EARTH / (2 * self._raw[0])

    def radius(self) -> float:
        """Returns the current Orbital Radius in km.

        Returns:
            radius (float): Current Orbital Radius.
        """
        return self.semi_parameter() / (1 + self._raw[1] * np.cos(self._raw[5]))

    def mean_motion(self) -> float:
        """Returns the Orbital Mean Motion in rad/s.

        Returns:
            mean_motion (float): Mean Motion.
        """
        return np.sqrt(c.MU_EARTH / (self._raw[0] ** 3))

    def period(self) -> float:
        """Returns the Orbital Period in sec.

        Returns:
            period (float): Orbital Period.
        """
        return 2 * np.pi * np.sqrt((self._raw[0] ** 3) / c.MU_EARTH)

    def inside_earth(self) -> bool:
        """Returns whether the orbit is currently inside Earth's Radius.

        Returns:
            inside (bool): Inside Earth.
        """
        return self.radius() <= c.RADIUS_EARTH

    def to_state(self):
        """Returns the Cartesian State Representation

        Returns:
            state (State): Cartesian State.
        """
        from comet.state.state import State

        return State(elements_to_cartesian(self._raw))

    def to_dict(self):
        """Method that creates a dictionary of required inputs for Elements construction.

        Returns:
            constructor (dict): dictionary of required inputs for Elements construction.
        """
        return {"type": "Elements", "vector": self._raw.tolist()}

    @staticmethod
    def from_dict(dict):
        """Method that creates a State from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            elements (Elements): Elements
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "Elements":
            raise ValueError("Elements(): Invalid construction dictionary")

        return Elements(dict["vector"])

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, Elements | list | np.ndarray):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare states
        if isinstance(other, Elements):
            return np.all(self._raw == other._raw)
        elif isinstance(other, list | np.ndarray):
            return np.all(self._raw == other)

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, Elements | list | np.ndarray):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare states
        if isinstance(other, Elements):
            return np.any(self._raw != other._raw)
        elif isinstance(other, list | np.ndarray):
            return np.any(self._raw != other)

    def __getitem__(self, i):
        """Define key indexing for getting components in Elements."""
        return self._raw[i]

    def __setitem__(self, i, value):
        """Define key indexing for setting components in Elements."""
        self._raw[i] = value

    def __len__(self):
        """Define length of Elements."""
        return len(self._raw)

    def __str__(self):
        """String Representation of Elements Class"""
        oe = self.copy()
        oe[2:] = np.degrees(oe[2:])
        return f"{oe[0]:9.3f} km, {oe[1]:07.6f}, {oe[2]:6.2f}{chr(176)}, {oe[3]:6.2f}{chr(176)}, {oe[4]:6.2f}{chr(176)}, {oe[5]:6.2f}{chr(176)}"

    def __repr__(self):
        """Class Representation of Elements Class"""
        return f"Elements({self._raw})"
