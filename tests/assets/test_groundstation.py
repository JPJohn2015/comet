# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets import Groundstation
from comet.state import Elements
from comet.state import LLA
from comet.time import Epoch
from comet.time import Duration
from comet.time import TIMELINE, TimelineMode
from comet.propagation.propagator import Propagator, GroundPropagator
from comet.frames import StateFrame


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
        """Test get_state with ECI frame and GroundPropagator."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        lla = LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01)
        propagator = GroundPropagator(epoch=epoch, state=lla)
        gs = Groundstation(propagator=propagator)

        state = gs.get_state(StateFrame.ECI)
        assert state is not None
        assert state.shape[1] == 6

    def test_get_state_ecef_with_propagator(self):
        """Test get_state with ECEF frame and GroundPropagator."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        lla = LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01)
        propagator = GroundPropagator(epoch=epoch, state=lla)
        gs = Groundstation(propagator=propagator)

        state = gs.get_state(StateFrame.ECEF)
        assert state is not None
        assert state.shape[1] == 6


class TestGroundstationGetPosition:
    """Test Groundstation position retrieval."""

    def test_get_position_eci(self):
        """Test get_position in ECI frame with GroundPropagator."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        lla = LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01)
        propagator = GroundPropagator(epoch=epoch, state=lla)
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


class TestGroundstationLLAAccessors:
    """Test Groundstation LLA accessors."""

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

    def test_get_lla_batch_mode(self):
        """Test get_lla returns 2D array in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        # Kennedy Space Center coordinates
        lla = LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01)
        propagator = GroundPropagator(epoch=epoch, state=lla)
        gs = Groundstation(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.BATCH)
        lla_result = gs.get_lla()

        assert lla_result.ndim == 2
        assert lla_result.shape[1] == 3
        assert lla_result.shape[0] > 1  # Multiple time points

    def test_get_lla_stepped_mode(self):
        """Test get_lla returns 1D array in STEPPED mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        lla = LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01)
        propagator = GroundPropagator(epoch=epoch, state=lla)
        gs = Groundstation(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        lla_result = gs.get_lla()

        assert lla_result.ndim == 1
        assert lla_result.shape[0] == 3

    def test_get_lla_values_constant(self):
        """Test get_lla returns constant values over time in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        lat = np.deg2rad(28.5)
        lon = np.deg2rad(-80.6)
        alt = 0.01
        lla = LLA(lat, lon, alt)
        propagator = GroundPropagator(epoch=epoch, state=lla)
        gs = Groundstation(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.BATCH)
        lla_result = gs.get_lla()

        # Check all time points have same LLA (ground station doesn't move)
        # Use looser tolerance for latitude/longitude (1 degree) since ground stations are stationary
        for i in range(lla_result.shape[0]):
            assert np.isclose(lla_result[i, 0], lat, atol=np.deg2rad(1.0))
            assert np.isclose(lla_result[i, 1], lon, atol=np.deg2rad(1.0))
            assert np.isclose(lla_result[i, 2], alt, atol=0.1)

    def test_get_lla_no_propagator_raises(self):
        """Test get_lla raises ValueError without propagator."""
        gs = Groundstation()
        with pytest.raises(ValueError, match="Cannot get LLA without a propagator"):
            gs.get_lla()

    def test_mode_switch_with_lla(self):
        """Test that mode switch affects LLA accessor output shape."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        lla = LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01)
        propagator = GroundPropagator(epoch=epoch, state=lla)
        gs = Groundstation(propagator=propagator)

        TIMELINE.set_mode(TimelineMode.BATCH)
        lla_batch = gs.get_lla()
        assert lla_batch.ndim == 2

        TIMELINE.set_mode(TimelineMode.STEPPED)
        lla_stepped = gs.get_lla()
        assert lla_stepped.ndim == 1


class TestGroundstationOrbitOnlyMethodsDisabled:
    """Test that orbit-only methods are disabled for Groundstation."""

    def test_get_coe_raises_not_implemented(self):
        """Test get_coe raises NotImplementedError."""
        gs = Groundstation()
        with pytest.raises(NotImplementedError, match="not applicable to ground-based assets"):
            gs.get_coe()

    def test_get_a_raises_not_implemented(self):
        """Test get_a raises NotImplementedError."""
        gs = Groundstation()
        with pytest.raises(NotImplementedError, match="not applicable to ground-based assets"):
            gs.get_a()

    def test_get_e_raises_not_implemented(self):
        """Test get_e raises NotImplementedError."""
        gs = Groundstation()
        with pytest.raises(NotImplementedError, match="not applicable to ground-based assets"):
            gs.get_e()

    def test_get_i_raises_not_implemented(self):
        """Test get_i raises NotImplementedError."""
        gs = Groundstation()
        with pytest.raises(NotImplementedError, match="not applicable to ground-based assets"):
            gs.get_i()

    def test_get_raan_raises_not_implemented(self):
        """Test get_raan raises NotImplementedError."""
        gs = Groundstation()
        with pytest.raises(NotImplementedError, match="not applicable to ground-based assets"):
            gs.get_raan()

    def test_get_omega_raises_not_implemented(self):
        """Test get_omega raises NotImplementedError."""
        gs = Groundstation()
        with pytest.raises(NotImplementedError, match="not applicable to ground-based assets"):
            gs.get_omega()

    def test_get_nu_raises_not_implemented(self):
        """Test get_nu raises NotImplementedError."""
        gs = Groundstation()
        with pytest.raises(NotImplementedError, match="not applicable to ground-based assets"):
            gs.get_nu()


class TestGroundstationBuilder:
    """Test Groundstation builder classmethod."""

    def teardown_method(self):
        """Clean up after tests."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.reset()

    def test_create_groundstation(self):
        """Test create_groundstation builder."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        # Kennedy Space Center coordinates
        lla = LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01)

        gs = Groundstation.create_groundstation(epoch, lla, name="KSC")

        assert isinstance(gs, Groundstation)
        assert gs.name == "KSC"
        assert gs._propagator is not None
        assert isinstance(gs._propagator, GroundPropagator)

    def test_create_groundstation_can_get_lla(self):
        """Test ground station created with builder can get LLA."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        TIMELINE.set_mode(TimelineMode.STEPPED)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        lat = np.deg2rad(28.5)
        lon = np.deg2rad(-80.6)
        alt = 0.01
        lla = LLA(lat, lon, alt)

        gs = Groundstation.create_groundstation(epoch, lla)

        lla_result = gs.get_lla()
        assert lla_result is not None
        assert lla_result.shape[0] == 3

        # Verify LLA values match input
        assert np.isclose(lla_result[0], lat, atol=np.deg2rad(0.1))
        assert np.isclose(lla_result[1], lon, atol=np.deg2rad(0.1))
        assert np.isclose(lla_result[2], alt, atol=0.001)

    def test_builder_multiple_groundstations(self):
        """Test creating multiple ground stations with builder."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)

        # Kennedy Space Center
        ksc = Groundstation.create_groundstation(
            epoch,
            LLA(np.deg2rad(28.5), np.deg2rad(-80.6), 0.01),
            name="KSC"
        )

        # Vandenberg Space Force Base
        vsfb = Groundstation.create_groundstation(
            epoch,
            LLA(np.deg2rad(34.7), np.deg2rad(-120.5), 0.1),
            name="VSFB"
        )

        assert ksc.name == "KSC"
        assert vsfb.name == "VSFB"
        assert ksc.id != vsfb.id
