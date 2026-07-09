"""COMET: Comprehensive Orbit Mechanics and Engineering Toolkit

A Python library for orbital mechanics, satellite operations, and space mission analysis.
"""

__version__ = "0.1.0"
__author__ = "James Johnson"

# Import main modules for convenience
from . import time
from . import state
from . import utilities
from . import frames

__all__ = ["time", "state", "utilities", "frames"]
