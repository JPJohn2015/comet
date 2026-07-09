# python imports
from enum import Enum

# ---------------------------------------------------------------------------------------------------------------------------
class CelestialFidelity(Enum):
    """Enum for Fidelity of calculations of Celestial Bodies.

    Integrator options are: LoFi, HiFi
    """
    # Status Enums
    LoFi = 0
    HiFi = 1