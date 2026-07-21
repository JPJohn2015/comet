"""Data managers for COMET library.

Managers for external data sources:
- DataManager: Generic HTTP download with caching
- SpaceTrackManager: Space-Track.org API client for TLE data
"""

from .data_manager import DataManager
from .spacetrack_manager import SpaceTrackManager

__all__ = ["DataManager", "SpaceTrackManager"]
