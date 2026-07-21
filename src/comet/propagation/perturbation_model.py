# python imports
from enum import Enum

# COMET imports
# from poliastro.earth.atmosphere.coesa62 import COESA62
# from poliastro.earth.atmosphere.coesa76 import COESA76
# from poliastro.earth.atmosphere.jacchia import Jacchia77


class AtmosphereModel(Enum):
    """Enum for which Atmospheric Drag Model to use.

    AtmosphereModel options are: NONE, EXPONENTIAL, COESA76
    """

    # Status Enums
    EXPONENTIAL = 0
    COESA76 = 1


class NBodyModel(Enum):
    """Enum for which Celestial Bodies to use.

    NBodyModel options are: NONE, MOON, SUN, SUNandMOON
    """

    # Status Enums
    MOON = 0
    SUN = 1
    SUNandMOON = 2


class SRPModel(Enum):
    """Enum for which Solar Radiation Pressure to use.

    SRPModel options are: NONE, SUN, ALBEDO, SUNandALBEDO
    """

    # Status Enums
    SUN = 0
    ALBEDO = 1
    SUNandALBEDO = 2


class GravityPotentialModel(Enum):
    """Enum for which Gravitational Potential Model to use.

    GravityPotentialModel options are: NONE, J2, J2andJ3
    """

    # Status Enums
    J2 = 0
    J2andJ3 = 1
