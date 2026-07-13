# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets.satellite import Satellite
from comet.state.elements import Elements
from comet.state.state import State
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE
from comet.propagation.propagator import Propagator, SpacePropagator
from comet.frames.frame import StateFrame


class TestSatelliteConstruction:
    """Test Satellite construction."""

    def test_satellite_no_propagator(self):
        """Test Satellite construction without propagator."""
        sat = Satellite()
        assert sat.id is not None
        assert sat._propagator is None
        assert isinstance(sat, Satellite)

    def test_satellite_with_propagator(self):
        """Test Satellite construction with Propagator."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)
        assert sat._propagator is propagator
        assert sat.id is not None

    def test_satellite_with_space_propagator(self):
        """Test Satellite construction with SpacePropagator."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)
        assert sat._propagator is propagator

    def test_satellite_inherits_from_asset(self):
        """Test that Satellite inherits Asset functionality."""
        sat = Satellite()
        # Should have Asset methods
        assert hasattr(sat, 'get_state')
        assert hasattr(sat, 'get_position')
        assert hasattr(sat, 'get_velocity')
        assert hasattr(sat, 'timeline')


class TestSatelliteGetState:
    """Test Satellite state retrieval."""

    def test_get_state_eci_with_propagator(self):
        """Test get_state with ECI frame and Propagator."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        state = sat.get_state(StateFrame.ECI)
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
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        state = sat.get_state(StateFrame.ECEF)
        assert state is not None
        assert state.shape[1] == 6

    def test_get_state_lla_with_propagator(self):
        """Test get_state with LLA frame."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        lla = sat.get_state(StateFrame.LLA)
        assert lla is not None
        assert lla.shape[1] == 3
        # Check altitude is reasonable for LEO satellite
        altitudes = lla[..., 2]
        assert np.all(altitudes > 400)  # km
        assert np.all(altitudes < 600)  # km


class TestSatelliteGetPosition:
    """Test Satellite position retrieval."""

    def test_get_position_eci(self):
        """Test get_position in ECI frame."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        position = sat.get_position(StateFrame.ECI)
        assert position is not None
        assert position.shape[1] == 3
        # Check position magnitudes are reasonable for LEO
        radii = np.linalg.norm(position, axis=-1)
        assert np.all(radii > 6800)  # km
        assert np.all(radii < 7000)  # km


class TestSatelliteGetVelocity:
    """Test Satellite velocity retrieval."""

    def test_get_velocity_eci(self):
        """Test get_velocity in ECI frame."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        velocity = sat.get_velocity(StateFrame.ECI)
        assert velocity is not None
        assert velocity.shape[1] == 3
        # Check velocity magnitudes are reasonable for LEO
        speeds = np.linalg.norm(velocity, axis=-1)
        assert np.all(speeds > 7.0)  # km/s
        assert np.all(speeds < 8.0)  # km/s


class TestSatelliteOrbitPropagation:
    """Test Satellite orbit propagation behavior."""

    def test_satellite_circular_orbit(self):
        """Test Satellite with circular orbit."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 2, 0, 0),
            Duration(seconds=600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        # Circular orbit: e = 0
        elements = Elements(6378 + 500, 0.0, np.deg2rad(0), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        position = sat.get_position(StateFrame.ECI)
        radii = np.linalg.norm(position, axis=-1)
        # Circular orbit should maintain constant radius
        assert np.std(radii) < 1.0  # km, very small variation

    def test_satellite_elliptical_orbit(self):
        """Test Satellite with elliptical orbit."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 2, 0, 0),
            Duration(seconds=600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        # Elliptical orbit: e = 0.2
        elements = Elements(6378 + 500, 0.2, np.deg2rad(0), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        position = sat.get_position(StateFrame.ECI)
        radii = np.linalg.norm(position, axis=-1)
        # Elliptical orbit should have varying radius
        assert np.std(radii) > 10.0  # km, significant variation


class TestSatelliteMultipleInstances:
    """Test multiple Satellite instances."""

    def test_multiple_satellites_unique_ids(self):
        """Test that multiple Satellites have unique IDs."""
        sat1 = Satellite()
        sat2 = Satellite()
        sat3 = Satellite()
        assert sat1.id != sat2.id
        assert sat2.id != sat3.id
        assert sat1.id != sat3.id

    def test_multiple_satellites_different_orbits(self):
        """Test multiple Satellites with different orbits."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)

        # LEO satellite
        elements1 = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        prop1 = Propagator(epoch=epoch, state=elements1)
        sat1 = Satellite(propagator=prop1)

        # MEO satellite
        elements2 = Elements(6378 + 20000, 0.01, np.deg2rad(55), 0, 0, 0)
        prop2 = Propagator(epoch=epoch, state=elements2)
        sat2 = Satellite(propagator=prop2)

        pos1 = sat1.get_position(StateFrame.ECI)
        pos2 = sat2.get_position(StateFrame.ECI)

        # Radii should be significantly different
        radii1 = np.linalg.norm(pos1, axis=-1)
        radii2 = np.linalg.norm(pos2, axis=-1)
        assert np.all(np.abs(radii2 - radii1) > 10000)  # km
