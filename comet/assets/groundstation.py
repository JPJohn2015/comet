# python imports
import numpy as np

# COMET imports
from comet.assets.asset import Asset
from comet.state.lla import LLA
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.propagation.propagator import Propagator, GroundPropagator
from comet.time.timeline import TIMELINE


class Groundstation(Asset):
    """Class that defines a Groundstation Asset."""

    def __init__(self, propagator: Propagator = None, **kwargs):

        # Initialize superclass constructor
        super().__init__(propagator, **kwargs)


# Testing
if __name__ == "__main__":
    # Parse Timeline
    TIMELINE.update(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

    # Create Satellite Properties for Propagator
    propagator = GroundPropagator(epoch=TIMELINE.start, state=LLA(0, 0, 0))
    site = Groundstation(propagator=propagator)

    site._sample_state()
