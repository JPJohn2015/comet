# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets.groundstation import Groundstation
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE
from comet.propagation.propagator import Propagator
from comet.frames.frame import StateFrame


class TestGroundstationConstruction:
    """Test Groundstation construction."""

    def test_groundstation_no_propagator(self):
        """Test Groundstation construction without propagator."""
        gs = Groundstation()
        assert gs.id is not None
        assert gs._propagator is None
        assert isinstance(gs, Groundstation)

    def test_groundstation_with_propagator(self):
        """Test Groundstation construction with Propagator."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        # Use Propagator with simple orbit for testing
        elements = Elements(6378 + 500, 0.0, 0, 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        gs = Groundstation(propagator=propagator)
        assert gs._propagator is propagator
        assert gs.id is not None

    def test_groundstation_inherits_from_asset(self):
        """Test that Groundstation inherits Asset functionality."""
        gs = Groundstation()
        # Should have Asset methods
        assert hasattr(gs, 'get_state')
        assert hasattr(gs, 'get_position')
        assert hasattr(gs, 'get_velocity')
        assert hasattr(gs, 'timeline')


class TestGroundstationGetState:
    """Test Groundstation state retrieval."""

    def test_get_state_eci_with_propagator(self):
        """Test get_state with ECI frame and Propagator."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.0, 0, 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        gs = Groundstation(propagator=propagator)

        state = gs.get_state(StateFrame.ECI)
        assert state is not None
        assert state.shape[1] == 6

    def test_get_state_ecef_with_propagator(self):
        """Test get_state with ECEF frame."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.0, 0, 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        gs = Groundstation(propagator=propagator)

        state = gs.get_state(StateFrame.ECEF)
        assert state is not None
        assert state.shape[1] == 6


class TestGroundstationGetPosition:
    """Test Groundstation position retrieval."""

    def test_get_position_eci(self):
        """Test get_position in ECI frame."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.0, 0, 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        gs = Groundstation(propagator=propagator)

        position = gs.get_position(StateFrame.ECI)
        assert position is not None
        assert position.shape[1] == 3


class TestGroundstationMultipleInstances:
    """Test multiple Groundstation instances."""

    def test_multiple_groundstations_unique_ids(self):
        """Test that multiple Groundstations have unique IDs."""
        gs1 = Groundstation()
        gs2 = Groundstation()
        gs3 = Groundstation()
        assert gs1.id != gs2.id
        assert gs2.id != gs3.id
        assert gs1.id != gs3.id
