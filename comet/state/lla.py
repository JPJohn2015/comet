# python imports
import numpy as np
import numpy.typing as npt

# COMET imports
from comet.time.epoch import Epoch
from comet.frames.transformations import lla_to_ecef, lla_to_eci

# --------------------------------------------------------------------------------------------------------------------------
class LLA:
    """Class that represents a Latitude, Longitude and Altitude coordinate.

    Example Constructions:
        * lla = LLA([lat, long, alt])
        * lla = LLA(lat, long, alt)
    """
    # ----------------------------------------------------------------------------------------------------------------------
    # Class Construction
    # ----------------------------------------------------------------------------------------------------------------------
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
            # 3 individual components for Position and Velocity
            if not np.all([isinstance(arg, int|float|np.int64|np.float64) for arg in args]):
                raise TypeError('LLA(): values must be an integer or float when providing 3 inputs')
            self.lat = args[0]
            self.long = args[1]
            self.alt = args[2]
            self._raw = np.array([self.lat, self.long, self.alt])

        elif len(args) == 1:
            # Full State Vector
            if not isinstance(args[0], list|np.ndarray):
                raise TypeError('LLA(): LLA must be an Array of length 3')
            if len(args[0]) != 3:
                raise TypeError('LLA(): LLA must be an Array of length 3')
            if not np.all([isinstance(arg, int|float|np.int64|np.float64) for arg in args[0]]):
                raise TypeError('LLA(): Values inside LLA array must be an integer or float')
            self.lat = args[0][0]
            self.long = args[0][1]
            self.alt = args[0][2]
            self._raw = np.array([self.lat, self.long, self.alt])
        
        else:
            raise ValueError('LLA(): Invalid number of inputs')
        
    # ----------------------------------------------------------------------------------------------------------------------
    # Class Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def copy(self):
        """Returns a copy of the LLA Coordinates.

        Returns:
            lla: Copy of the LLA.
        """
        return LLA(self.lat, self.long, self.alt)
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_latitude(self) -> float:
        """Returns the Geodetic Latitude.

        Returns:
            latitude (float): Geodetic Latitude in deg.
        """
        return self.lat
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_longitude(self) -> float:
        """Returns the Geodetic Longitude.

        Returns:
            longitude (float): Geodetic Longitude in deg.
        """
        return self.long
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_altitude(self) -> float:
        """Returns the Geodetic Altitude.

        Returns:
            altitude (float): Geodetic Altitude in km.
        """
        return self.alt
    
    # ----------------------------------------------------------------------------------------------------------------------
    def ecef_position(self) -> np.ndarray:
        """Returns the ECEF position of the LLA Coordinates.

        Returns:
            ecef (np.ndarray): ECEF State position in km.
        """
        return lla_to_ecef(self._raw)
    
    # ----------------------------------------------------------------------------------------------------------------------
    def eci_position(self, epoch: npt.ArrayLike) -> np.ndarray:
        """Returns the ECI position at the specified Epoch of the LLA Coordinates.

        Returns:
            eci (np.ndarray): ECI State position in km.
        """
        if isinstance(epoch, Epoch):
            lla = self._raw
        else:
            lla = self._raw[np.newaxis,...].repeat(len(epoch), axis=0)

        return lla_to_eci(epoch, lla)
    
    # ----------------------------------------------------------------------------------------------------------------------
    def to_dict(self):
        """Method that creates a dictionary of required inputs for LLA construction.

        Returns:
            constructor (dict): dictionary of required inputs for LLA construction.
        """
        return {
            "type": 'LLA',
            "coordinates": self._raw.tolist()
            }
    
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def from_dict(dict):
        """Method that creates a LLA from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            lla (LLA): LLA
        """
        # Check that dictionary of construction is of the correct type
        if dict['type'] != 'LLA':
            raise ValueError('LLA(): Invalid construction dictionary')

        return LLA(dict["coordinates"])
    
    # ----------------------------------------------------------------------------------------------------------------------
    # Operator Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        """Override Equality operator.
        """
        # Error checking
        if not isinstance(other, LLA):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")
        
        # Compare LLA coordinates
        if isinstance(other, LLA):
            return np.all(self._raw == other._raw)
        elif isinstance(other, list|np.ndarray):
            return np.all(self._raw == other)
        
    # ----------------------------------------------------------------------------------------------------------------------
    def __ne__(self, other) -> bool:
        """Override Non-Equality operator.
        """
        # Error checking
        if not isinstance(other, LLA|list|np.ndarray):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")
        
        # Compare LLA coordinates
        if isinstance(other, LLA):
            return np.any(self._raw != other._raw)
        elif isinstance(other, list|np.ndarray):
            return np.any(self._raw != other)
    
    # ----------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, i):
        """Define key indexing for getting components in LLA Coordinates.
        """
        return self._raw[i]
    
    # ----------------------------------------------------------------------------------------------------------------------
    def __setitem__(self, i, value):
        """Define key indexing for setting components in LLA Coordinates.
        """
        self._raw[i] = value
    
    # ----------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        """Define length of LLA Coordinates.
        """
        return len(self._raw)
    
    # ----------------------------------------------------------------------------------------------------------------------
    # Representation Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """String Representation of LLA Coordinates Class"""
        return f'({self.lat}{chr(176)}, {self.long}{chr(176)}, {self.alt} km)'
    
    # ----------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        """Class Representation of LLA Coordinates Class"""
        return f'LLA({self._raw})'    
