# python imports
import pytest
import numpy as np

# comet imports
from comet.propagation.propagator import Propagator, Integrator
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.state.state import State
from comet.state.elements import Elements
from comet.time.epoch import Epoch


class TestIntegratorEnum:
    """Tests for Integrator enum."""

    def test_integrator_values(self):
        """Test Integrator has correct values."""
        assert Integrator.DOPRI5.value == "dopri5"
        assert Integrator.DOP853.value == "dop853"
        assert Integrator.LSODA.value == "lsoda"
        assert Integrator.VODE.value == "vode"

    def test_integrator_membership(self):
        """Test Integrator membership."""
        assert Integrator.DOPRI5 in Integrator
        assert Integrator.DOP853 in Integrator
        assert Integrator.LSODA in Integrator
        assert Integrator.VODE in Integrator

    def test_integrator_count(self):
        """Test Integrator has exactly 4 members."""
        assert len(Integrator) == 4


class TestPropagatorConstruction:
    """Tests for Propagator construction."""

    def test_default_construction_with_state(self):
        """Test Propagator with default parameters and State."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        prop = Propagator(epoch, state)

        assert prop.epoch == epoch
        assert prop.state == state
        assert isinstance(prop.force_model, ForceModel)
        assert isinstance(prop.sat_properties, SatelliteProperties)
        assert prop.integrator == Integrator.DOPRI5
        assert prop.abs_tol == 1e-6
        assert prop.rel_tol == 1e-6
        assert prop.min_step == 10.0
        assert prop.max_step == 60.0

    def test_construction_with_elements(self):
        """Test Propagator construction with Elements."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        elements = Elements(7000.0, 0.001, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        prop = Propagator(epoch, elements)

        assert prop.epoch == epoch
        assert prop.state == elements

    def test_construction_with_custom_force_model(self):
        """Test Propagator with custom ForceModel."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel(drag=True, gravity=True)
        prop = Propagator(epoch, state, force_model=fm)

        assert prop.force_model.drag is True
        assert prop.force_model.gravity is True

    def test_construction_with_custom_sat_properties(self):
        """Test Propagator with custom SatelliteProperties."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        sp = SatelliteProperties(dry_mass=500.0, area=10.0)
        prop = Propagator(epoch, state, sat_properties=sp)

        assert prop.sat_properties.dry_mass == 500.0
        assert prop.sat_properties.area == 10.0

    def test_construction_with_custom_integrator_settings(self):
        """Test Propagator with custom integrator settings."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        prop = Propagator(
            epoch,
            state,
            integrator=Integrator.DOP853,
            abs_tol=1e-8,
            rel_tol=1e-8,
            min_step=1.0,
            max_step=120.0,
        )

        assert prop.integrator == Integrator.DOP853
        assert prop.abs_tol == 1e-8
        assert prop.rel_tol == 1e-8
        assert prop.min_step == 1.0
        assert prop.max_step == 120.0


class TestPropagatorCopy:
    """Tests for Propagator copy method."""

    def test_copy_creates_new_instance(self):
        """Test copy creates a new Propagator instance."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        prop1 = Propagator(epoch, state)
        prop2 = prop1.copy()

        assert prop1 is not prop2
        assert prop1.epoch == prop2.epoch
        # Note: state comparison might not work with == depending on State implementation

    def test_copy_is_independent(self):
        """Test that modifying copy doesn't affect original."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        prop1 = Propagator(epoch, state, abs_tol=1e-6)
        prop2 = prop1.copy()

        # Modify copy's attributes
        prop2.abs_tol = 1e-8

        # Original should be unchanged
        assert prop1.abs_tol == 1e-6
        assert prop2.abs_tol == 1e-8


class TestPropagatorGetState:
    """Tests for Propagator get_state method."""

    def test_get_state_returns_state(self):
        """Test get_state returns State when constructed with State."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        prop = Propagator(epoch, state)

        returned_state = prop.get_state()
        assert isinstance(returned_state, State)
        assert returned_state == state

    def test_get_state_converts_elements(self):
        """Test get_state converts Elements to State."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        elements = Elements(7000.0, 0.001, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        prop = Propagator(epoch, elements)

        returned_state = prop.get_state()
        assert isinstance(returned_state, State)
        # Check that conversion happened (position should be near 7000 km radius)
        pos = returned_state.position()
        assert 6900 < np.linalg.norm(pos) < 7100


class TestPropagatorBasicFunctionality:
    """Basic functionality tests for Propagator (without full propagation)."""

    def test_propagator_attributes_accessible(self):
        """Test that all Propagator attributes are accessible."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel(gravity=True)
        sp = SatelliteProperties(dry_mass=200.0)
        prop = Propagator(
            epoch,
            state,
            force_model=fm,
            sat_properties=sp,
            integrator=Integrator.LSODA,
            abs_tol=1e-7,
            rel_tol=1e-7,
            min_step=5.0,
            max_step=100.0,
        )

        # Check all attributes
        assert prop.epoch == epoch
        assert prop.state == state
        assert prop.force_model == fm
        assert prop.sat_properties == sp
        assert prop.integrator == Integrator.LSODA
        assert prop.abs_tol == 1e-7
        assert prop.rel_tol == 1e-7
        assert prop.min_step == 5.0
        assert prop.max_step == 100.0

    def test_propagator_with_different_integrators(self):
        """Test Propagator can be constructed with different integrators."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])

        for integrator in [Integrator.DOPRI5, Integrator.DOP853, Integrator.LSODA, Integrator.VODE]:
            prop = Propagator(epoch, state, integrator=integrator)
            assert prop.integrator == integrator


class TestPropagatorIntegration:
    """Integration tests for Propagator (these test actual propagation behavior in pipeline test)."""

    def test_propagator_initialization_valid(self):
        """Test that Propagator initializes with valid parameters."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel()
        sp = SatelliteProperties()

        # Should not raise
        prop = Propagator(epoch, state, fm, sp)
        assert prop is not None

    def test_propagator_force_model_two_body(self):
        """Test Propagator recognizes two-body force model."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel()  # All False, should be two-body
        prop = Propagator(epoch, state, force_model=fm)

        assert prop.force_model.is_two_body() is True

    def test_propagator_force_model_perturbed(self):
        """Test Propagator recognizes perturbed force model."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel(gravity=True)
        prop = Propagator(epoch, state, force_model=fm)

        assert prop.force_model.is_two_body() is False
