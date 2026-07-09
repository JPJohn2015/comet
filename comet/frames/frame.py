# python imports
from enum import Enum


class ManeuverFrame(str, Enum):
    """Enum for which Coordinate Frame the Maneuver is to take place.

    ManeuverFrame options are: ECI, RIC.
    """

    ECI = "ECI"
    RIC = "RIC"


class StateFrame(str, Enum):
    """Enum for which Coordinate Frame the State Data is represented.

    StateFrame options are: ECI, ECEF, LLA.
    """

    ECI = "ECI"
    ECEF = "ECEF"
    LLA = "LLA"
