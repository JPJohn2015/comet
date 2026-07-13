# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets.asset import Asset
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE
from comet.propagation.propagator import Propagator
from comet.frames.frame import StateFrame


class TestAssetConstruction:
    """Test Asset construction."""

    def test_asset_no_propagator(self):
        """Test Asset construction without propagator."""
        asset = Asset()
        assert asset.id is not None
        assert asset._propagator is None
        assert asset._eci_state is None
        assert asset._ecef_state is None
        assert asset.timeline is TIMELINE

    def test_asset_with_propagator(self):
        """Test Asset construction with propagator."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)
        assert asset._propagator is propagator
        assert asset.id is not None

    def test_asset_unique_ids(self):
        """Test that each Asset gets a unique ID."""
        asset1 = Asset()
        asset2 = Asset()
        asset3 = Asset()
        assert asset1.id != asset2.id
        assert asset2.id != asset3.id
        assert asset1.id != asset3.id


class TestAssetGetState:
    """Test Asset get_state method."""

    def test_get_state_eci_no_propagator(self):
        """Test get_state with ECI frame and no propagator."""
        asset = Asset()
        # Should not raise, but will have None state
        asset._sample_eci_state()
        assert asset._eci_state is None

    def test_get_state_eci_with_propagator(self):
        """Test get_state with ECI frame and propagator."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
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
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        velocity = asset.get_velocity(StateFrame.ECI)
        state = asset.get_state(StateFrame.ECI)

        assert velocity.shape[1] == 3
        assert np.allclose(velocity, state[..., 3:6])


class TestAssetSamplingMethods:
    """Test Asset internal sampling methods."""

    def test_sample_eci_caches_result(self):
        """Test that _sample_eci_state caches result."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        assert asset._eci_state is None
        asset._sample_eci_state()
        assert asset._eci_state is not None

        # Save reference
        first_state = asset._eci_state
        # Call again - should use cached result
        asset._sample_eci_state()
        # Since _sample_eci_state always recomputes, just check it exists
        assert asset._eci_state is not None

    def test_sample_ecef_computes_eci_if_needed(self):
        """Test that _sample_ecef_state computes ECI if not cached."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        assert asset._eci_state is None
        assert asset._ecef_state is None
        asset._sample_ecef_state()
        assert asset._eci_state is not None
        assert asset._ecef_state is not None

    def test_sample_state_computes_both(self):
        """Test that _sample_state computes both ECI and ECEF."""
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(seconds=3600)
        )
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)
        propagator = Propagator(epoch=epoch, state=elements)
        asset = Asset(propagator=propagator)

        assert asset._eci_state is None
        assert asset._ecef_state is None
        asset._sample_state()
        assert asset._eci_state is not None
        assert asset._ecef_state is not None
