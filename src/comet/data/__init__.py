"""Data management and external data sources for COMET library.

This module provides managers for downloading and accessing external data:
- DataManager: Generic data download and caching
- SpaceTrackManager: Space-Track.org TLE data access

The data directory also contains:
- iau/: IAU2000 Earth orientation tables
- tle/: Cached TLE data files
- download_paths.json: URLs for downloadable data sources
"""

from .managers.data_manager import DataManager
from .managers.spacetrack_manager import SpaceTrackManager

__all__ = ["DataManager", "SpaceTrackManager"]
