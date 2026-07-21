# python imports
import numpy as np
import numpy.typing as npt

# COMET imports
from comet.time import Epoch
from comet.frames.transformations import lla_to_ecef, lla_to_eci


class LLA:
    """Class that represents a Latitude, Longitude and Altitude coordinate.

    Example Constructions:
        * lla = LLA([lat, long, alt])
        * lla = LLA(lat, long, alt)
    """

    def __init__(self, *args):
        """Defines a Latitude, Longitude and Altitude geodetic state.

        Construct from 3 components in deg, deg, km.

        Args:
            lat (int|float): Geodetic Latitude in deg.
            long (int|float): Geodetic Longitude in deg.
            alt (int|float): Geodetic Altitude in km.

        Construct from an array of [latitude, longitude and altitude] in deg, deg, km.

        Args:
            lla (list|np.ndarray): Latitude, Longitude Altitude array.
        """
        # Construct Class
        if len(args) == 3:
            # 3 individual components for Latitude, Longitude, and Altitude
            if not np.all([isinstance(arg, int | float | np.int64 | np.float64) for arg in args]):
                raise TypeError("LLA(): values must be an integer or float when providing 3 inputs")
            self._raw = np.array([args[0], args[1], args[2]])

        elif len(args) == 1:
            # Full LLA Vector
            if not isinstance(args[0], list | np.ndarray):
                raise TypeError("LLA(): LLA must be an Array of length 3")
            if len(args[0]) != 3:
                raise TypeError("LLA(): LLA must be an Array of length 3")
            if not np.all(
                [isinstance(arg, int | float | np.int64 | np.float64) for arg in args[0]]
            ):
                raise TypeError("LLA(): Values inside LLA array must be an integer or float")
            self._raw = np.array(args[0])

        else:
            raise ValueError("LLA(): Invalid number of inputs")

    def copy(self):
        """Returns a copy of the LLA Coordinates.

        Returns:
            lla: Copy of the LLA.
        """
        return LLA(self._raw)

    @property
    def lat(self) -> float:
        """Geodetic latitude in deg."""
        return self._raw[0]

    @property
    def long(self) -> float:
        """Geodetic longitude in deg."""
        return self._raw[1]

    @property
    def alt(self) -> float:
        """Geodetic altitude in km."""
        return self._raw[2]

    def get_latitude(self) -> float:
        """Returns the Geodetic Latitude.

        Returns:
            latitude (float): Geodetic Latitude in deg.
        """
        return self.lat

    def get_longitude(self) -> float:
        """Returns the Geodetic Longitude.

        Returns:
            longitude (float): Geodetic Longitude in deg.
        """
        return self.long

    def get_altitude(self) -> float:
        """Returns the Geodetic Altitude.

        Returns:
            altitude (float): Geodetic Altitude in km.
        """
        return self.alt

    def ecef_position(self) -> np.ndarray:
        """Returns the ECEF position of the LLA Coordinates.

        Returns:
            ecef (np.ndarray): ECEF position in km.
        """
        state = lla_to_ecef(self._raw)
        return state[:3] if state.ndim == 1 else state[:, :3]

    def eci_position(self, epoch: npt.ArrayLike) -> np.ndarray:
        """Returns the ECI position at the specified Epoch of the LLA Coordinates.

        Returns:
            eci (np.ndarray): ECI position in km.
        """
        if isinstance(epoch, Epoch):
            lla = self._raw
        else:
            lla = self._raw[np.newaxis, ...].repeat(len(epoch), axis=0)

        state = lla_to_eci(epoch, lla)
        return state[:3] if state.ndim == 1 else state[:, :3]

    def to_dict(self):
        """Method that creates a dictionary of required inputs for LLA construction.

        Returns:
            constructor (dict): dictionary of required inputs for LLA construction.
        """
        return {"type": "LLA", "coordinates": self._raw.tolist()}

    @staticmethod
    def from_dict(dict):
        """Method that creates a LLA from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            lla (LLA): LLA
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "LLA":
            raise ValueError("LLA(): Invalid construction dictionary")

        return LLA(dict["coordinates"])

    def __hash__(self):
        """Make LLA hashable for use in sets and as dictionary keys."""
        return hash(tuple(self._raw))

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, LLA | list | np.ndarray):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare LLA coordinates
        if isinstance(other, LLA):
            return np.allclose(self._raw, other._raw)
        elif isinstance(other, list | np.ndarray):
            return np.allclose(self._raw, other)

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, LLA | list | np.ndarray):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare LLA coordinates
        if isinstance(other, LLA):
            return np.any(self._raw != other._raw)
        elif isinstance(other, list | np.ndarray):
            return np.any(self._raw != other)

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

    def __getitem__(self, i):
        """Define key indexing for getting components in LLA Coordinates."""
        return self._raw[i]

    def __setitem__(self, i, value):
        """Define key indexing for setting components in LLA Coordinates."""
        self._raw[i] = value

    def __len__(self):
        """Define length of LLA Coordinates."""
        return len(self._raw)

    def __str__(self):
        """String Representation of LLA Coordinates Class"""
        return f"({self.lat}{chr(176)}, {self.long}{chr(176)}, {self.alt} km)"

    def __repr__(self):
        """Class Representation of LLA Coordinates Class"""
        return f"LLA({self._raw})"
