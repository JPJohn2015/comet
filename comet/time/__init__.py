"""Time module for COMET library.

This module contains classes and functions for handling time and epochs.
"""

from .epoch import Epoch, TimeSystem
from .duration import Duration
from .timeline import Timeline, TIMELINE

__all__ = ["Epoch", "TimeSystem", "Duration", "Timeline", "TIMELINE"]
