# python imports
import numpy as np
import itertools

# COMET imports
from comet.assets.asset import Asset
from comet.utilities.constants import Constants as c
from comet.state.state import State
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.propagation.propagator import Propagator, SpacePropagator
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.time.timeline import TIMELINE
from comet.frames.transformations import eci_to_ecef, ecef_to_lla
from comet.frames.frame import StateFrame


class Satellite(Asset):
    """Class that defines a Satellite Asset."""

    def __init__(self, propagator: Propagator = None, **kwargs):

        # Initialize superclass constructor
        super().__init__(propagator, **kwargs)


# Testing
if __name__ == "__main__":
    # Parse Timeline
    TIMELINE.update(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

    # Create Satellite Properties for Propagator
    start_elements = Elements(6378 + 780, 0, np.pi / 4, 0, 0, 0)

    propagator = SpacePropagator(epoch=TIMELINE.start, state=start_elements)
    sat = Satellite(propagator=propagator)

    sat._sample_state()
