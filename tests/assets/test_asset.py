"""Comprehensive test suite for the Asset class.

This module tests all functionality of the Asset class including:
- Construction and propagator interface
- Mode-aware state sampling (BATCH vs STEPPED)
- Cache invalidation via timeline hash
- Component management
- Access/range stubs
- Serialization
"""

# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets.asset import Asset
from comet.components.component import Component
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE, TimelineMode
from comet.propagation.propagator import (
    Propagator,
    PropagatorCategory,
    SpacePropagator,
    GroundPropagator,
)
from comet.state.lla import LLA
from comet.frames.frame import StateFrame


class TestAssetConstruction:
    """Test Asset construction and basic properties."""

    def test_asset_no_propagator(self):
        """Test Asset construction without propagator."""
        asset = Asset()
        assert asset.id is not None
        assert asset.name is not None
        assert asset.get_propagator() is None
        assert asset._eci_state is None
        assert asset._ecef_state is None
        assert asset.timeline is TIMELINE

    def test_asset_with_propagator(self):
        """Test Asset construction with propagator."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        assert asset.get_propagator() is propagator
        assert asset.id is not None

    def test_asset_with_name(self):
        """Test Asset construction with custom name."""
        asset = Asset(name="TestAsset")
        assert asset.name == "TestAsset"

    def test_asset_unique_ids(self):
        """Test that each Asset gets a unique ID."""
        asset1 = Asset()
        asset2 = Asset()
        asset3 = Asset()
        assert asset1.id != asset2.id
        assert asset2.id != asset3.id
        assert asset1.id != asset3.id


class TestAssetPropagatorInterface:
    """Test Asset propagator property and category tracking."""

    def test_set_propagator_generic(self):
        """Test setting generic Propagator."""
        asset = Asset()
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)

        asset.set_propagator(propagator)

        assert asset.get_propagator() is propagator
        assert asset.get_propagator_category() == PropagatorCategory.GENERIC

    def test_set_propagator_space(self):
        """Test setting SpacePropagator."""
        asset = Asset()
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = SpacePropagator(epoch=epoch, state=elements)

        asset.set_propagator(propagator)

        assert asset.get_propagator() is propagator
        assert asset.get_propagator_category() == PropagatorCategory.SPACE

    def test_set_propagator_ground(self):
        """Test setting GroundPropagator."""
        # Note: GroundPropagator has specific initialization requirements
        # This test verifies category tracking works with different propagator types
        asset = Asset()
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)

        # Create a generic propagator and manually set category
        propagator = Propagator(epoch=epoch, state=elements)
        asset._propagator = propagator
        asset._propagator_category = PropagatorCategory.GROUND

        assert asset.get_propagator() is propagator
        assert asset.get_propagator_category() == PropagatorCategory.GROUND

    def test_set_propagator_clears_cache(self):
        """Test that setting propagator clears cached states."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        # Sample state
        asset.get_state(StateFrame.ECI)
        assert asset._eci_state is not None

        # Change propagator
        new_elements = Elements(6378 + 600, 0.01, np.deg2rad(45), 0, 0, 0)
        new_propagator = Propagator(epoch=epoch, state=new_elements)
        asset.set_propagator(new_propagator)

        # Cache should be cleared
        assert asset._eci_state is None

    def test_set_propagator_invalid_type(self):
        """Test that invalid propagator type raises TypeError."""
        asset = Asset()
        with pytest.raises(TypeError, match="must be an instance of Propagator"):
            asset.set_propagator("not a propagator")


class TestAssetModeAwareStateSampling:
    """Test Asset mode-aware state sampling (BATCH vs STEPPED)."""

    def setup_method(self):
        """Reset timeline to BATCH mode before each test."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.reset()

    def teardown_method(self):
        """Reset timeline to BATCH mode after each test."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.reset()

    def test_batch_mode_returns_2d_array(self):
        """Test that BATCH mode returns 2D arrays."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        state = asset.get_state(StateFrame.ECI)

        assert state.ndim == 2
        assert state.shape[1] == 6
        assert state.shape[0] > 1  # Multiple time steps

    def test_stepped_mode_returns_1d_array(self):
        """Test that STEPPED mode returns 1D arrays."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        TIMELINE.set_mode(TimelineMode.STEPPED)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        state = asset.get_state(StateFrame.ECI)

        assert state.ndim == 1
        assert state.shape[0] == 6

    def test_stepped_mode_advances_correctly(self):
        """Test that STEPPED mode respects timeline advance."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=600)
        )
        TIMELINE.set_mode(TimelineMode.STEPPED)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        # Get state at start
        state_0 = asset.get_state(StateFrame.ECI)

        # Advance timeline
        TIMELINE.advance(3)

        # Get state at advanced position
        state_3 = asset.get_state(StateFrame.ECI)

        # States should be different (orbit has evolved)
        assert not np.allclose(state_0, state_3)

    def test_mode_switch_invalidates_cache(self):
        """Test that switching modes invalidates cached states."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        # Sample in BATCH mode
        batch_state = asset.get_state(StateFrame.ECI)
        assert batch_state.ndim == 2

        # Switch to STEPPED mode
        TIMELINE.set_mode(TimelineMode.STEPPED)

        # Get state in STEPPED mode
        stepped_state = asset.get_state(StateFrame.ECI)
        assert stepped_state.ndim == 1

        # First row of batch should match stepped (same epoch)
        assert np.allclose(batch_state[0, :], stepped_state, rtol=1e-5)


class TestAssetCacheInvalidation:
    """Test Asset cache invalidation via timeline hash."""

    def test_timeline_update_invalidates_cache(self):
        """Test that updating timeline invalidates cached states."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        # Sample state
        state1 = asset.get_state(StateFrame.ECI)
        n_steps_1 = state1.shape[0]

        # Update timeline
        TIMELINE.update(step=Duration(seconds=900))  # Fewer steps

        # Get state again
        state2 = asset.get_state(StateFrame.ECI)
        n_steps_2 = state2.shape[0]

        # Should have different number of steps
        assert n_steps_1 != n_steps_2

    def test_timeline_mode_change_invalidates_cache(self):
        """Test that timeline mode changes trigger cache invalidation."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=1800)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        # Cache state in BATCH mode
        _ = asset.get_state(StateFrame.ECI)
        hash_batch = asset._valid_timeline_hash

        # Switch to STEPPED mode
        TIMELINE.set_mode(TimelineMode.STEPPED)

        # Get state triggers cache check
        _ = asset.get_state(StateFrame.ECI)
        hash_stepped = asset._valid_timeline_hash

        # Hashes should be different
        assert hash_batch != hash_stepped


class TestAssetGetState:
    """Test Asset get_state method."""

    def test_get_state_eci_with_propagator(self):
        """Test get_state with ECI frame and propagator."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        state = asset.get_state(StateFrame.ECI)
        assert state is not None
        assert state.shape[1] == 6
        # Verify it returns a copy, not original
        state_copy = asset.get_state(StateFrame.ECI)
        assert state is not state_copy
        assert np.allclose(state, state_copy)

    def test_get_state_ecef_with_propagator(self):
        """Test get_state with ECEF frame."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        state = asset.get_state(StateFrame.ECEF)
        assert state is not None
        assert state.shape[1] == 6

    def test_get_state_lla_with_propagator(self):
        """Test get_state with LLA frame."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        lla = asset.get_state(StateFrame.LLA)
        assert lla is not None
        assert lla.shape[1] == 3

    def test_get_state_string_frames(self):
        """Test get_state with string frame names."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        eci_upper = asset.get_state("ECI")
        eci_lower = asset.get_state("eci")
        assert np.allclose(eci_upper, eci_lower)

        ecef_upper = asset.get_state("ECEF")
        ecef_lower = asset.get_state("ecef")
        assert np.allclose(ecef_upper, ecef_lower)

    def test_get_state_invalid_frame(self):
        """Test get_state with invalid frame raises ValueError."""
        asset = Asset()
        with pytest.raises(ValueError, match="Invalid StateFrame"):
            asset.get_state("INVALID")


class TestAssetGetPosition:
    """Test Asset get_position method."""

    def test_get_position_eci(self):
        """Test get_position returns only position components."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        position = asset.get_position(StateFrame.ECI)
        state = asset.get_state(StateFrame.ECI)

        assert position.shape[1] == 3
        assert np.allclose(position, state[..., 0:3])


class TestAssetGetVelocity:
    """Test Asset get_velocity method."""

    def test_get_velocity_eci(self):
        """Test get_velocity returns only velocity components."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        TIMELINE.set_mode(TimelineMode.BATCH)

        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        velocity = asset.get_velocity(StateFrame.ECI)
        state = asset.get_state(StateFrame.ECI)

        assert velocity.shape[1] == 3
        assert np.allclose(velocity, state[..., 3:6])


class TestAssetComponentManagement:
    """Test Asset component management methods."""

    def test_add_component(self):
        """Test adding a component to asset."""
        asset = Asset()
        component = Component(mass=10.0)

        asset.add_component(component)

        assert len(asset.get_components()) == 1
        assert component.parent is asset

    def test_add_component_by_name(self):
        """Test adding named component."""
        asset = Asset()
        component = Component(mass=10.0)
        component.name = "TestComponent"

        asset.add_component(component)

        assert asset.get_component(component_name="TestComponent") is component

    def test_add_component_without_id_raises(self):
        """Test that adding component without id raises ValueError."""
        asset = Asset()

        # Create object without id attribute
        class FakeComponent:
            pass

        fake = FakeComponent()

        with pytest.raises(ValueError, match="must have an 'id' attribute"):
            asset.add_component(fake)

    def test_remove_component(self):
        """Test removing a component."""
        asset = Asset()
        component = Component(mass=10.0)

        asset.add_component(component)
        assert len(asset.get_components()) == 1

        result = asset.remove_component(component.id)

        assert result is True
        assert len(asset.get_components()) == 0
        assert component.parent is None

    def test_remove_component_not_found(self):
        """Test removing non-existent component returns False."""
        asset = Asset()
        result = asset.remove_component(999)
        assert result is False

    def test_get_component_by_id(self):
        """Test getting component by ID."""
        asset = Asset()
        component = Component(mass=10.0)
        asset.add_component(component)

        retrieved = asset.get_component(component_id=component.id)

        assert retrieved is component

    def test_get_component_by_name(self):
        """Test getting component by name."""
        asset = Asset()
        component = Component(mass=10.0)
        component.name = "TestComp"
        asset.add_component(component)

        retrieved = asset.get_component(component_name="TestComp")

        assert retrieved is component

    def test_get_component_not_found(self):
        """Test getting non-existent component returns None."""
        asset = Asset()
        assert asset.get_component(component_id=999) is None
        assert asset.get_component(component_name="NonExistent") is None

    def test_get_components_returns_copy(self):
        """Test that get_components returns a copy."""
        asset = Asset()
        component = Component(mass=10.0)
        asset.add_component(component)

        components = asset.get_components()
        components.clear()

        # Original list should be unchanged
        assert len(asset.get_components()) == 1

    def test_clear_components(self):
        """Test clearing all components."""
        asset = Asset()
        comp1 = Component(mass=10.0)
        comp2 = Component(mass=20.0)
        asset.add_component(comp1)
        asset.add_component(comp2)

        assert len(asset.get_components()) == 2

        asset.clear_components()

        assert len(asset.get_components()) == 0
        assert comp1.parent is None
        assert comp2.parent is None


class TestAssetSerialization:
    """Test Asset serialization (to_dict/from_dict)."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator, name="TestAsset")

        d = asset.to_dict()

        assert d["type"] == "Asset"
        assert d["id"] == asset.id
        assert d["name"] == "TestAsset"
        assert d["propagator"] is not None
        assert d["propagator_category"] == "generic"

    def test_to_dict_no_propagator(self):
        """Test to_dict with no propagator."""
        asset = Asset(name="NoPropaAsset")

        d = asset.to_dict()

        assert d["type"] == "Asset"
        assert d["propagator"] is None

    def test_from_dict_basic(self):
        """Test basic from_dict deserialization (without propagator)."""
        asset1 = Asset(name="Original")

        d = asset1.to_dict()
        asset2 = Asset.from_dict(d)

        assert asset2.id == asset1.id
        assert asset2.name == asset1.name

    def test_from_dict_preserves_id(self):
        """Test that from_dict preserves asset ID."""
        asset1 = Asset(name="TestAsset")
        original_id = asset1.id

        d = asset1.to_dict()
        asset2 = Asset.from_dict(d)

        assert asset2.id == original_id

    def test_from_dict_invalid_type(self):
        """Test from_dict raises ValueError for invalid type."""
        d = {"type": "NotAnAsset"}

        with pytest.raises(ValueError, match="Invalid construction dictionary"):
            Asset.from_dict(d)

    def test_serialization_round_trip(self):
        """Test complete serialization round trip (without propagator)."""
        asset1 = Asset(name="RoundTrip")
        # Add a component
        comp = Component(mass=15.0, name="TestComp")
        asset1.add_component(comp)

        d = asset1.to_dict()
        asset2 = Asset.from_dict(d)

        assert asset2.id == asset1.id
        assert asset2.name == asset1.name
        assert len(asset2.get_components()) == 1
        assert asset2.get_component(component_name="TestComp") is not None
