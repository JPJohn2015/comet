"""Comprehensive test suite for the Component class.

This module tests all functionality of the Component class including:
- Construction and basic properties
- Parent linkage and rotation validation
- Mode-transparent state pass-through
- Composite access stub
- Serialization
"""

# python imports
import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

# COMET imports
from comet.components.component import Component
from comet.assets.asset import Asset
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE, TimelineMode
from comet.propagation.propagator import Propagator
from comet.frames.frame import StateFrame


class TestComponentConstruction:
    """Test Component construction."""

    def test_component_default(self):
        """Test Component with default parameters."""
        comp = Component()
        assert comp.mass == 0.0
        assert np.allclose(comp.body_vector, [1, 0, 0])
        assert comp.id is not None
        assert comp.name is not None
        assert comp.parent is None

    def test_component_with_mass(self):
        """Test Component with custom mass."""
        comp = Component(mass=10.5)
        assert comp.mass == 10.5
        assert np.allclose(comp.body_vector, [1, 0, 0])

    def test_component_with_body_vector(self):
        """Test Component with custom body vector."""
        body_vec = np.array([0, 1, 0])
        comp = Component(body_vector=body_vec)
        assert comp.mass == 0.0
        assert np.allclose(comp.body_vector, body_vec)

    def test_component_with_name(self):
        """Test Component with custom name."""
        comp = Component(name="TestComponent")
        assert comp.name == "TestComponent"

    def test_component_with_rotation(self):
        """Test Component with custom rotation matrix."""
        # 90 degree rotation around z-axis
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        comp = Component(parent_to_component_rot=rot)
        assert np.allclose(comp.get_parent_to_component_rot(), rot)

    def test_component_default_rotation_is_identity(self):
        """Test Component default rotation is identity."""
        comp = Component()
        assert np.allclose(comp.get_parent_to_component_rot(), np.eye(3))

    def test_component_unique_ids(self):
        """Test that each Component gets a unique ID."""
        comp1 = Component()
        comp2 = Component()
        comp3 = Component()
        assert comp1.id != comp2.id
        assert comp2.id != comp3.id
        assert comp1.id != comp3.id


class TestComponentRotationValidation:
    """Test Component rotation matrix validation."""

    def test_set_rotation_valid(self):
        """Test setting a valid rotation matrix."""
        comp = Component()
        # 90 degree rotation around x-axis
        rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        comp.set_parent_to_component_rot(rot)
        assert np.allclose(comp.get_parent_to_component_rot(), rot)

    def test_set_rotation_invalid_shape(self):
        """Test that invalid shape raises ValueError."""
        comp = Component()
        with pytest.raises(ValueError, match="must be 3x3"):
            comp.set_parent_to_component_rot(np.eye(4))

    def test_set_rotation_not_orthogonal(self):
        """Test that non-orthogonal matrix raises ValueError."""
        comp = Component()
        not_orthogonal = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="must be orthogonal"):
            comp.set_parent_to_component_rot(not_orthogonal)

    def test_set_rotation_reflection(self):
        """Test that reflection (det=-1) raises ValueError."""
        comp = Component()
        # Reflection matrix (det = -1)
        reflection = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="determinant must be"):
            comp.set_parent_to_component_rot(reflection)

    def test_get_rotation_returns_copy(self):
        """Test that get_rotation returns a copy."""
        comp = Component()
        rot1 = comp.get_parent_to_component_rot()
        rot1[0, 0] = 999  # Modify returned array
        rot2 = comp.get_parent_to_component_rot()
        assert rot2[0, 0] != 999  # Original should be unchanged

    def test_scipy_rotation_matrices_valid(self):
        """Test that scipy-generated rotation matrices are accepted."""
        comp = Component()

        # Test various rotations
        for angles in [[0, 0, 90], [45, 45, 0], [30, 60, 90]]:
            rot = R.from_euler('xyz', angles, degrees=True).as_matrix()
            comp.set_parent_to_component_rot(rot)
            assert np.allclose(comp.get_parent_to_component_rot(), rot)


class TestComponentParentLinkage:
    """Test Component parent linkage."""

    def test_parent_initially_none(self):
        """Test that parent is initially None."""
        comp = Component()
        assert comp.parent is None

    def test_parent_set_by_asset(self):
        """Test that parent is set when added to asset."""
        asset = Asset()
        comp = Component()

        asset.add_component(comp)

        assert comp.parent is asset

    def test_parent_cleared_on_removal(self):
        """Test that parent is cleared when removed from asset."""
        asset = Asset()
        comp = Component()

        asset.add_component(comp)
        assert comp.parent is asset

        asset.remove_component(comp.id)
        assert comp.parent is None


class TestComponentModeTransparentStatePassThrough:
    """Test Component mode-transparent state pass-through."""

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

    def test_get_state_no_parent_raises(self):
        """Test that get_state without parent raises ValueError."""
        comp = Component()
        with pytest.raises(ValueError, match="no parent"):
            comp.get_state()

    def test_get_position_no_parent_raises(self):
        """Test that get_position without parent raises ValueError."""
        comp = Component()
        with pytest.raises(ValueError, match="no parent"):
            comp.get_position()

    def test_get_velocity_no_parent_raises(self):
        """Test that get_velocity without parent raises ValueError."""
        comp = Component()
        with pytest.raises(ValueError, match="no parent"):
            comp.get_velocity()

    def test_get_state_batch_mode_returns_2d(self):
        """Test that get_state returns 2D array in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        TIMELINE.set_mode(TimelineMode.BATCH)

        state = comp.get_state(StateFrame.ECI)

        assert state.ndim == 2
        assert state.shape[1] == 6
        assert state.shape[0] > 1

    def test_get_state_stepped_mode_returns_1d(self):
        """Test that get_state returns 1D array in STEPPED mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        TIMELINE.set_mode(TimelineMode.STEPPED)

        state = comp.get_state(StateFrame.ECI)

        assert state.ndim == 1
        assert state.shape[0] == 6

    def test_get_position_batch_mode_returns_2d(self):
        """Test that get_position returns 2D array in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        TIMELINE.set_mode(TimelineMode.BATCH)

        position = comp.get_position(StateFrame.ECI)

        assert position.ndim == 2
        assert position.shape[1] == 3
        assert position.shape[0] > 1

    def test_get_position_stepped_mode_returns_1d(self):
        """Test that get_position returns 1D array in STEPPED mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        TIMELINE.set_mode(TimelineMode.STEPPED)

        position = comp.get_position(StateFrame.ECI)

        assert position.ndim == 1
        assert position.shape[0] == 3

    def test_get_velocity_batch_mode_returns_2d(self):
        """Test that get_velocity returns 2D array in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        TIMELINE.set_mode(TimelineMode.BATCH)

        velocity = comp.get_velocity(StateFrame.ECI)

        assert velocity.ndim == 2
        assert velocity.shape[1] == 3
        assert velocity.shape[0] > 1

    def test_get_velocity_stepped_mode_returns_1d(self):
        """Test that get_velocity returns 1D array in STEPPED mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        TIMELINE.set_mode(TimelineMode.STEPPED)

        velocity = comp.get_velocity(StateFrame.ECI)

        assert velocity.ndim == 1
        assert velocity.shape[0] == 3

    def test_component_state_matches_parent(self):
        """Test that component state matches parent asset state."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        TIMELINE.set_mode(TimelineMode.BATCH)

        comp_state = comp.get_state(StateFrame.ECI)
        asset_state = asset.get_state(StateFrame.ECI)

        assert np.allclose(comp_state, asset_state)

    def test_mode_switch_transparent(self):
        """Test that mode switch is transparent to component."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        comp = Component()
        asset.add_component(comp)

        # Get state in BATCH mode
        TIMELINE.set_mode(TimelineMode.BATCH)
        batch_state = comp.get_state(StateFrame.ECI)
        assert batch_state.ndim == 2

        # Switch to STEPPED mode
        TIMELINE.set_mode(TimelineMode.STEPPED)
        stepped_state = comp.get_state(StateFrame.ECI)
        assert stepped_state.ndim == 1

        # First row of batch should match stepped (same epoch)
        assert np.allclose(batch_state[0, :], stepped_state, rtol=1e-5)


class TestComponentAccessStub:
    """Test Component composite access stub."""

    def test_get_access_raises_not_implemented(self):
        """Test that get_access raises NotImplementedError."""
        comp = Component()
        with pytest.raises(NotImplementedError, match="Phase 6"):
            comp.get_access()


class TestComponentGetters:
    """Test Component getter methods."""

    def test_get_mass_default(self):
        """Test get_mass with default mass."""
        comp = Component()
        assert comp.get_mass() == 0.0

    def test_get_mass_custom(self):
        """Test get_mass with custom mass."""
        comp = Component(mass=42.3)
        assert comp.get_mass() == 42.3

    def test_get_body_vector_default(self):
        """Test get_body_vector with default vector."""
        comp = Component()
        vec = comp.get_body_vector()
        assert np.allclose(vec, [1, 0, 0])

    def test_get_body_vector_custom(self):
        """Test get_body_vector with custom vector."""
        body_vec = np.array([0.577, 0.577, 0.577])
        comp = Component(body_vector=body_vec)
        vec = comp.get_body_vector()
        assert np.allclose(vec, body_vec)

    def test_get_id_returns_int(self):
        """Test get_id returns an integer."""
        comp = Component()
        comp_id = comp.get_id()
        assert isinstance(comp_id, int)

    def test_get_name_returns_string(self):
        """Test get_name returns a string."""
        comp = Component(name="TestComp")
        assert comp.get_name() == "TestComp"


class TestComponentSerialization:
    """Test Component serialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        comp = Component(mass=15.0, name="Sensor")
        d = comp.to_dict()

        assert d["type"] == "Component"
        assert d["id"] == comp.id
        assert d["name"] == "Sensor"
        assert d["mass"] == 15.0

    def test_to_dict_with_rotation(self):
        """Test to_dict includes rotation matrix."""
        rot = R.from_euler('z', 90, degrees=True).as_matrix()
        comp = Component(parent_to_component_rot=rot)
        d = comp.to_dict()

        assert "parent_to_component_rot" in d
        assert isinstance(d["parent_to_component_rot"], list)

    def test_from_dict_basic(self):
        """Test basic from_dict deserialization."""
        comp1 = Component(mass=20.0, name="Original")
        d = comp1.to_dict()

        comp2 = Component.from_dict(d)

        assert comp2.id == comp1.id
        assert comp2.name == comp1.name
        assert comp2.mass == comp1.mass

    def test_from_dict_with_rotation(self):
        """Test from_dict preserves rotation matrix."""
        rot = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        comp1 = Component(parent_to_component_rot=rot)
        d = comp1.to_dict()

        comp2 = Component.from_dict(d)

        assert np.allclose(comp2.get_parent_to_component_rot(), rot)

    def test_from_dict_invalid_type(self):
        """Test from_dict raises ValueError for invalid type."""
        d = {"type": "NotAComponent"}
        with pytest.raises(ValueError, match="Invalid construction dictionary"):
            Component.from_dict(d)

    def test_serialization_round_trip(self):
        """Test complete serialization round trip."""
        rot = R.from_euler('y', 45, degrees=True).as_matrix()
        comp1 = Component(
            mass=25.0,
            body_vector=[0, 1, 0],
            name="RoundTrip",
            parent_to_component_rot=rot
        )

        d = comp1.to_dict()
        comp2 = Component.from_dict(d)

        assert comp2.id == comp1.id
        assert comp2.name == comp1.name
        assert comp2.mass == comp1.mass
        assert np.allclose(comp2.body_vector, comp1.body_vector)
        assert np.allclose(comp2.get_parent_to_component_rot(), comp1.get_parent_to_component_rot())


class TestComponentEdgeCases:
    """Test Component edge cases."""

    def test_component_negative_mass(self):
        """Test Component with negative mass."""
        comp = Component(mass=-5.0)
        assert comp.mass == -5.0

    def test_component_zero_body_vector(self):
        """Test Component with zero body vector."""
        comp = Component(body_vector=[0, 0, 0])
        assert np.allclose(comp.body_vector, [0, 0, 0])

    def test_component_large_mass(self):
        """Test Component with very large mass."""
        comp = Component(mass=1e6)
        assert comp.mass == 1e6

    def test_component_body_vector_not_normalized(self):
        """Test Component accepts non-normalized body vectors."""
        body_vec = [10, 20, 30]
        comp = Component(body_vector=body_vec)
        assert np.allclose(comp.body_vector, body_vec)


class TestComponentMultipleInstances:
    """Test multiple Component instances."""

    def test_multiple_components_independent(self):
        """Test that multiple Components are independent."""
        comp1 = Component(mass=10.0, body_vector=[1, 0, 0])
        comp2 = Component(mass=20.0, body_vector=[0, 1, 0])

        assert comp1.mass != comp2.mass
        assert not np.allclose(comp1.body_vector, comp2.body_vector)
        assert comp1.id != comp2.id

    def test_component_modification_doesnt_affect_others(self):
        """Test modifying one Component doesn't affect others."""
        comp1 = Component(mass=10.0)
        comp2 = Component(mass=10.0)

        comp1.mass = 20.0
        assert comp2.mass == 10.0
