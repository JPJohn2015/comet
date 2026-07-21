"""Asset management for COMET library.

This module provides classes for representing space and ground assets:
- Asset: Base class for all assets
- Satellite: Orbital assets with propagation and maneuver capabilities
- Groundstation: Fixed or mobile ground-based assets
- AssetArray: Collection of assets for batch operations
- SatelliteArray: Collection of satellites
- GroundstationArray: Collection of ground stations

Assets maintain state over time and can be propagated through timelines.
"""

from .asset import Asset, AssetArray
from .satellite import Satellite, SatelliteArray
from .groundstation import Groundstation, GroundstationArray

__all__ = [
    "Asset",
    "AssetArray",
    "Satellite",
    "SatelliteArray",
    "Groundstation",
    "GroundstationArray",
]
