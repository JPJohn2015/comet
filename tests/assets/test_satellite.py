# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets.satellite import Satellite
from comet.state.elements import Elements
from comet.state.state import State
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE, TimelineMode
from comet.propagation.propagator import Propagator, SpacePropagator
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
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


class TestSatelliteCOEAccessors:
    """Test Satellite classical orbital element accessors."""

    def setup_method(self):
        """Set up test fixtures."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.reset()

    def teardown_method(self):
        """Clean up after tests."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.reset()

    def test_get_coe_batch_mode(self):
        """Test get_coe returns 2D array in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), np.deg2rad(30), np.deg2rad(60), np.deg2rad(90))
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.BATCH)
        coe = sat.get_coe()

        assert coe.ndim == 2
        assert coe.shape[1] == 6
        assert coe.shape[0] > 1  # Multiple time points

    def test_get_coe_stepped_mode(self):
        """Test get_coe returns 1D array in STEPPED mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), np.deg2rad(30), np.deg2rad(60), np.deg2rad(90))
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        coe = sat.get_coe()

        assert coe.ndim == 1
        assert coe.shape[0] == 6

    def test_get_coe_values_reasonable(self):
        """Test get_coe returns reasonable values."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        a = 6878.0  # km
        e = 0.01
        i = np.deg2rad(45)
        raan = np.deg2rad(30)
        omega = np.deg2rad(60)
        nu = np.deg2rad(90)
        elements = Elements(a, e, i, raan, omega, nu)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        coe = sat.get_coe()

        # Check semi-major axis (should be close to input)
        assert np.isclose(coe[0], a, rtol=0.01)
        # Check eccentricity
        assert np.isclose(coe[1], e, rtol=0.1)
        # Check inclination
        assert np.isclose(coe[2], i, atol=np.deg2rad(1))

    def test_get_coe_no_propagator_raises(self):
        """Test get_coe raises ValueError without propagator."""
        sat = Satellite()
        with pytest.raises(ValueError, match="Cannot compute COE without a propagator"):
            sat.get_coe()

    def test_get_a(self):
        """Test get_a returns semi-major axis."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        a = 6878.0
        elements = Elements(a, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        a_result = sat.get_a()

        assert isinstance(a_result, (np.ndarray, float, np.floating))
        assert np.isclose(a_result, a, rtol=0.01)

    def test_get_e(self):
        """Test get_e returns eccentricity."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        e = 0.05
        elements = Elements(6878.0, e, np.deg2rad(45), 0, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        e_result = sat.get_e()

        assert isinstance(e_result, (np.ndarray, float, np.floating))
        assert np.isclose(e_result, e, rtol=0.1)

    def test_get_i(self):
        """Test get_i returns inclination."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        i = np.deg2rad(60)
        elements = Elements(6878.0, 0.01, i, 0, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        i_result = sat.get_i()

        assert isinstance(i_result, (np.ndarray, float, np.floating))
        assert np.isclose(i_result, i, atol=np.deg2rad(1))

    def test_get_raan(self):
        """Test get_raan returns RAAN."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        raan = np.deg2rad(45)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), raan, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        raan_result = sat.get_raan()

        assert isinstance(raan_result, (np.ndarray, float, np.floating))
        assert np.isclose(raan_result, raan, atol=np.deg2rad(1))

    def test_get_omega(self):
        """Test get_omega returns argument of perigee."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        omega = np.deg2rad(90)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, omega, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        omega_result = sat.get_omega()

        assert isinstance(omega_result, (np.ndarray, float, np.floating))
        assert np.isclose(omega_result, omega, atol=np.deg2rad(5))

    def test_get_nu(self):
        """Test get_nu returns true anomaly."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        nu = np.deg2rad(120)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, 0, nu)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        nu_result = sat.get_nu()

        assert isinstance(nu_result, (np.ndarray, float, np.floating))
        # True anomaly changes over time, just check it's valid
        assert 0 <= nu_result <= 2 * np.pi

    def test_individual_accessors_batch_mode(self):
        """Test individual accessors return 1D arrays in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.BATCH)

        a = sat.get_a()
        e = sat.get_e()
        i = sat.get_i()

        assert a.ndim == 1
        assert e.ndim == 1
        assert i.ndim == 1
        assert len(a) > 1

    def test_mode_switch_invalidates_coe_cache(self):
        """Test that mode switch forces COE recalculation."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)
        sat = Satellite(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.BATCH)
        coe_batch = sat.get_coe()
        assert coe_batch.ndim == 2

        TIMELINE.set_mode(TimelineMode.STEPPED)
        coe_stepped = sat.get_coe()
        assert coe_stepped.ndim == 1


class TestSatelliteBuilders:
    """Test Satellite builder classmethods."""

    def test_create_satellite_from_elements(self):
        """Test create_satellite builder with elements."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, 0, 0)

        sat = Satellite.create_satellite(epoch, elements, name="TestSat")

        assert isinstance(sat, Satellite)
        assert sat.name == "TestSat"
        assert sat._propagator is not None
        assert isinstance(sat._propagator, SpacePropagator)

    def test_create_satellite_propagates(self):
        """Test satellite created with builder can propagate."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, 0, 0)

        sat = Satellite.create_satellite(epoch, elements)

        state = sat.get_state(StateFrame.ECI)
        assert state is not None
        assert state.shape[1] == 6

    def test_create_from_state(self):
        """Test create_from_state builder with Cartesian state."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        # Simple circular orbit state
        r = 6878.0  # km
        v = np.sqrt(398600.4418 / r)  # circular velocity
        state = State(
            np.array([r, 0, 0]),
            np.array([0, v, 0])
        )

        sat = Satellite.create_from_state(epoch, state, name="FromState")

        assert isinstance(sat, Satellite)
        assert sat.name == "FromState"
        assert sat._propagator is not None

    def test_builder_with_force_model(self):
        """Test builder accepts force_model parameter."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, 0, 0)
        force_model = ForceModel()

        sat = Satellite.create_satellite(epoch, elements, force_model=force_model)

        assert isinstance(sat, Satellite)
        assert sat._propagator is not None

    def test_builder_with_sat_properties(self):
        """Test builder accepts sat_properties parameter."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6878.0, 0.01, np.deg2rad(45), 0, 0, 0)
        sat_props = SatelliteProperties()

        sat = Satellite.create_satellite(epoch, elements, sat_properties=sat_props)

        assert isinstance(sat, Satellite)
        assert sat._propagator is not None


class TestSatelliteCOERoundTrip:
    """Test round-trip: elements -> satellite -> get_coe()."""

    def setup_method(self):
        """Set up test fixtures."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 30, 0),
            Duration(seconds=60)
        )
        TIMELINE.set_mode(TimelineMode.STEPPED)
        TIMELINE.reset()

    def teardown_method(self):
        """Clean up after tests."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.reset()

    def test_round_trip_circular_orbit(self):
        """Test round-trip for circular orbit."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        a = 6878.0
        e = 0.0
        i = np.deg2rad(45)
        raan = np.deg2rad(30)
        omega = np.deg2rad(60)
        nu = np.deg2rad(90)

        elements = Elements(a, e, i, raan, omega, nu)
        sat = Satellite.create_satellite(epoch, elements)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        coe = sat.get_coe()

        # Check semi-major axis matches
        assert np.isclose(coe[0], a, rtol=0.01)
        # Check eccentricity
        assert np.isclose(coe[1], e, atol=0.001)
        # Check inclination
        assert np.isclose(coe[2], i, atol=np.deg2rad(0.5))

    def test_round_trip_elliptical_orbit(self):
        """Test round-trip for elliptical orbit."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        a = 7000.0
        e = 0.1
        i = np.deg2rad(55)

        elements = Elements(a, e, i, 0, 0, 0)
        sat = Satellite.create_satellite(epoch, elements)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        coe = sat.get_coe()

        # Check semi-major axis
        assert np.isclose(coe[0], a, rtol=0.01)
        # Check eccentricity
        assert np.isclose(coe[1], e, rtol=0.1)
        # Check inclination
        assert np.isclose(coe[2], i, atol=np.deg2rad(1))
