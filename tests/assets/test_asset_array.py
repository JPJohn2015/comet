"""Comprehensive test suite for AssetArray classes.

This module tests all functionality of AssetArray, SatelliteArray, and GroundstationArray including:
- Construction and validation
- Mode-aware stacked getters
- Asset retrieval methods
- Type checking and filtering
- Concatenation with narrowing rules
- Serialization
"""

# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets.asset import Asset, AssetArray, _narrowest_common_array
from comet.assets.satellite import Satellite, SatelliteArray
from comet.assets.groundstation import Groundstation, GroundstationArray
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE, TimelineMode
from comet.propagation.propagator import Propagator, PropagatorCategory
from comet.frames.frame import StateFrame


class TestAssetArrayConstruction:
    """Test AssetArray construction and validation."""

    def test_construct_empty_array(self):
        """Test constructing empty AssetArray."""
        arr = AssetArray([])
        assert len(arr) == 0

    def test_construct_with_assets(self):
        """Test constructing AssetArray with assets."""
        assets = [Asset(), Asset(), Asset()]
        arr = AssetArray(assets)
        assert len(arr) == 3

    def test_construct_with_mixed_types(self):
        """Test constructing AssetArray with mixed asset types."""
        assets = [Asset(), Satellite(), Groundstation()]
        arr = AssetArray(assets)
        assert len(arr) == 3

    def test_construct_with_non_asset_raises(self):
        """Test that non-Asset element raises TypeError."""
        with pytest.raises(TypeError, match="not an Asset instance"):
            AssetArray([Asset(), "not an asset"])

    def test_len(self):
        """Test __len__ method."""
        arr = AssetArray([Asset(), Asset()])
        assert len(arr) == 2

    def test_getitem(self):
        """Test __getitem__ method."""
        asset1, asset2 = Asset(), Asset()
        arr = AssetArray([asset1, asset2])
        assert arr[0] is asset1
        assert arr[1] is asset2

    def test_iter(self):
        """Test __iter__ method."""
        assets = [Asset(), Asset(), Asset()]
        arr = AssetArray(assets)
        assert list(arr) == assets


class TestAssetArrayModeAwareStacking:
    """Test AssetArray mode-aware stacked getters."""

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

    def test_get_state_batch_mode(self):
        """Test get_state returns (n_asset, n_time, 6) in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)

        assets = [
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
        ]
        arr = AssetArray(assets)

        TIMELINE.set_mode(TimelineMode.BATCH)
        states = arr.get_state(StateFrame.ECI)

        assert states.ndim == 3
        assert states.shape[0] == 2  # n_asset
        assert states.shape[2] == 6  # state components
        assert states.shape[1] > 1   # n_time

    def test_get_state_stepped_mode(self):
        """Test get_state returns (n_asset, 6) in STEPPED mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)

        assets = [
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
        ]
        arr = AssetArray(assets)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        states = arr.get_state(StateFrame.ECI)

        assert states.ndim == 2
        assert states.shape[0] == 2  # n_asset
        assert states.shape[1] == 6  # state components

    def test_get_position_batch_mode(self):
        """Test get_position returns (n_asset, n_time, 3) in BATCH mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)

        assets = [
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
        ]
        arr = AssetArray(assets)

        TIMELINE.set_mode(TimelineMode.BATCH)
        positions = arr.get_position(StateFrame.ECI)

        assert positions.ndim == 3
        assert positions.shape[0] == 3  # n_asset
        assert positions.shape[2] == 3  # position components

    def test_get_velocity_stepped_mode(self):
        """Test get_velocity returns (n_asset, 3) in STEPPED mode."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)

        assets = [
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
            Asset(propagator=Propagator(epoch=epoch, state=elements)),
        ]
        arr = AssetArray(assets)

        TIMELINE.set_mode(TimelineMode.STEPPED)
        velocities = arr.get_velocity(StateFrame.ECI)

        assert velocities.ndim == 2
        assert velocities.shape[0] == 2  # n_asset
        assert velocities.shape[1] == 3  # velocity components

    def test_empty_array_getters(self):
        """Test getters on empty array return empty arrays."""
        arr = AssetArray([])

        assert len(arr.get_state()) == 0
        assert len(arr.get_position()) == 0
        assert len(arr.get_velocity()) == 0


class TestAssetArrayRetrieval:
    """Test AssetArray asset retrieval methods."""

    def test_get_asset_by_id(self):
        """Test get_asset_by_id retrieval."""
        asset1, asset2, asset3 = Asset(), Asset(), Asset()
        arr = AssetArray([asset1, asset2, asset3])

        assert arr.get_asset_by_id(asset2.id) is asset2

    def test_get_asset_by_id_not_found(self):
        """Test get_asset_by_id returns None if not found."""
        arr = AssetArray([Asset(), Asset()])
        assert arr.get_asset_by_id(999999) is None

    def test_get_asset_by_name(self):
        """Test get_asset_by_name retrieval."""
        asset1 = Asset(name="First")
        asset2 = Asset(name="Second")
        arr = AssetArray([asset1, asset2])

        assert arr.get_asset_by_name("Second") is asset2

    def test_get_asset_by_name_not_found(self):
        """Test get_asset_by_name returns None if not found."""
        arr = AssetArray([Asset(name="A"), Asset(name="B")])
        assert arr.get_asset_by_name("C") is None

    def test_get_asset_by_index(self):
        """Test get_asset_by_index retrieval."""
        assets = [Asset(), Asset(), Asset()]
        arr = AssetArray(assets)

        assert arr.get_asset_by_index(1) is assets[1]

    def test_get_assets_by_category(self):
        """Test filtering by propagator category."""
        epoch = Epoch(2024, 1, 1, 0, 0, 0)
        elements = Elements(6378 + 500, 0.01, np.deg2rad(45), 0, 0, 0)

        asset1 = Asset(propagator=Propagator(epoch=epoch, state=elements))
        asset2 = Asset(propagator=Propagator(epoch=epoch, state=elements))
        asset3 = Asset()  # No propagator

        arr = AssetArray([asset1, asset2, asset3])

        generic_assets = arr.get_assets_by_category(PropagatorCategory.GENERIC)
        assert len(generic_assets) == 2


class TestAssetArrayTypeChecking:
    """Test AssetArray type checking and filtering."""

    def test_is_homogeneous_empty_array(self):
        """Test is_homogeneous on empty array returns True."""
        arr = AssetArray([])
        assert arr.is_homogeneous() is True

    def test_is_homogeneous_all_same_type(self):
        """Test is_homogeneous with all same type."""
        arr = AssetArray([Satellite(), Satellite(), Satellite()])
        assert arr.is_homogeneous() is True

    def test_is_homogeneous_mixed_types(self):
        """Test is_homogeneous with mixed types returns False."""
        arr = AssetArray([Satellite(), Groundstation(), Satellite()])
        assert arr.is_homogeneous() is False

    def test_is_homogeneous_with_type_check_positive(self):
        """Test is_homogeneous(Satellite) when all are Satellites."""
        arr = AssetArray([Satellite(), Satellite()])
        assert arr.is_homogeneous(Satellite) is True

    def test_is_homogeneous_with_type_check_negative(self):
        """Test is_homogeneous(Satellite) when not all are Satellites."""
        arr = AssetArray([Satellite(), Groundstation()])
        assert arr.is_homogeneous(Satellite) is False

    def test_filter_by_type_satellites(self):
        """Test filter_by_type returns SatelliteArray for all Satellites."""
        arr = AssetArray([Satellite(), Groundstation(), Satellite(), Asset()])

        filtered = arr.filter_by_type(Satellite)

        assert isinstance(filtered, SatelliteArray)
        assert len(filtered) == 2

    def test_filter_by_type_groundstations(self):
        """Test filter_by_type returns GroundstationArray for all Groundstations."""
        arr = AssetArray([Satellite(), Groundstation(), Groundstation()])

        filtered = arr.filter_by_type(Groundstation)

        assert isinstance(filtered, GroundstationArray)
        assert len(filtered) == 2

    def test_filter_by_type_base_assets(self):
        """Test filter_by_type returns AssetArray for base Assets."""
        arr = AssetArray([Asset(), Satellite(), Asset()])

        # Filter for exact Asset type (not subclasses)
        filtered = arr.filter_by_type(Asset)

        # Should get all three since Satellite is isinstance of Asset
        assert len(filtered) == 3


class TestAssetArrayConcatenation:
    """Test AssetArray concatenation with narrowing rules."""

    def test_concat_satellite_arrays(self):
        """Test SatelliteArray + SatelliteArray = SatelliteArray."""
        arr1 = SatelliteArray([Satellite(), Satellite()])
        arr2 = SatelliteArray([Satellite()])

        result = arr1 + arr2

        assert isinstance(result, SatelliteArray)
        assert len(result) == 3

    def test_concat_groundstation_arrays(self):
        """Test GroundstationArray + GroundstationArray = GroundstationArray."""
        arr1 = GroundstationArray([Groundstation()])
        arr2 = GroundstationArray([Groundstation(), Groundstation()])

        result = arr1 + arr2

        assert isinstance(result, GroundstationArray)
        assert len(result) == 3

    def test_concat_mixed_returns_base(self):
        """Test SatelliteArray + GroundstationArray = AssetArray."""
        arr1 = SatelliteArray([Satellite()])
        arr2 = GroundstationArray([Groundstation()])

        result = arr1 + arr2

        assert type(result) == AssetArray
        assert not isinstance(result, (SatelliteArray, GroundstationArray))
        assert len(result) == 2

    def test_concat_asset_to_array(self):
        """Test Asset + AssetArray concatenation."""
        sat = Satellite()
        arr = SatelliteArray([Satellite(), Satellite()])

        result = arr + sat

        assert isinstance(result, SatelliteArray)
        assert len(result) == 3

    def test_concat_array_to_asset_radd(self):
        """Test __radd__: Asset + AssetArray."""
        sat = Satellite()
        arr = SatelliteArray([Satellite()])

        result = sat + arr

        assert isinstance(result, SatelliteArray)
        assert len(result) == 2

    def test_concat_base_arrays(self):
        """Test AssetArray + AssetArray concatenation."""
        arr1 = AssetArray([Asset(), Asset()])
        arr2 = AssetArray([Asset()])

        result = arr1 + arr2

        assert isinstance(result, AssetArray)
        assert len(result) == 3

    def test_concat_invalid_type_raises(self):
        """Test concatenation with invalid type raises TypeError."""
        arr = AssetArray([Asset()])

        with pytest.raises(TypeError):
            arr + "not an array"


class TestAssetArrayNarrowingHelper:
    """Test _narrowest_common_array helper function."""

    def test_empty_list_returns_base_array(self):
        """Test empty list returns AssetArray."""
        result = _narrowest_common_array([])
        assert isinstance(result, AssetArray)
        assert len(result) == 0

    def test_all_satellites_returns_satellite_array(self):
        """Test all Satellites returns SatelliteArray."""
        assets = [Satellite(), Satellite(), Satellite()]
        result = _narrowest_common_array(assets)
        assert isinstance(result, SatelliteArray)

    def test_all_groundstations_returns_groundstation_array(self):
        """Test all Groundstations returns GroundstationArray."""
        assets = [Groundstation(), Groundstation()]
        result = _narrowest_common_array(assets)
        assert isinstance(result, GroundstationArray)

    def test_mixed_returns_base_array(self):
        """Test mixed types returns AssetArray."""
        assets = [Satellite(), Groundstation()]
        result = _narrowest_common_array(assets)
        assert type(result) == AssetArray

    def test_base_assets_returns_base_array(self):
        """Test base Assets returns AssetArray."""
        assets = [Asset(), Asset()]
        result = _narrowest_common_array(assets)
        assert isinstance(result, AssetArray)


class TestSatelliteArray:
    """Test SatelliteArray typed subclass."""

    def test_construct_with_satellites(self):
        """Test constructing SatelliteArray with Satellites."""
        sats = [Satellite(), Satellite()]
        arr = SatelliteArray(sats)
        assert len(arr) == 2

    def test_construct_with_non_satellite_raises(self):
        """Test constructing with non-Satellite raises TypeError."""
        with pytest.raises(TypeError, match="not a Satellite instance"):
            SatelliteArray([Satellite(), Asset()])

    def test_copy_returns_satellite_array(self):
        """Test copy returns SatelliteArray type."""
        arr = SatelliteArray([Satellite(), Satellite()])
        copied = arr.copy()
        assert isinstance(copied, SatelliteArray)
        assert len(copied) == 2


class TestGroundstationArray:
    """Test GroundstationArray typed subclass."""

    def test_construct_with_groundstations(self):
        """Test constructing GroundstationArray with Groundstations."""
        sites = [Groundstation(), Groundstation()]
        arr = GroundstationArray(sites)
        assert len(arr) == 2

    def test_construct_with_non_groundstation_raises(self):
        """Test constructing with non-Groundstation raises TypeError."""
        with pytest.raises(TypeError, match="not a Groundstation instance"):
            GroundstationArray([Groundstation(), Satellite()])

    def test_copy_returns_groundstation_array(self):
        """Test copy returns GroundstationArray type."""
        arr = GroundstationArray([Groundstation()])
        copied = arr.copy()
        assert isinstance(copied, GroundstationArray)
        assert len(copied) == 1


class TestAssetArrayAccessStubs:
    """Test AssetArray access and range stubs."""

    def test_get_access_raises_not_implemented(self):
        """Test get_access raises NotImplementedError."""
        arr = AssetArray([Asset()])
        with pytest.raises(NotImplementedError, match="Phase 6"):
            arr.get_access()

    def test_get_range_to_raises_not_implemented(self):
        """Test get_range_to raises NotImplementedError."""
        arr = AssetArray([Asset()])
        with pytest.raises(NotImplementedError, match="Phase 6"):
            arr.get_range_to()


class TestAssetArraySerialization:
    """Test AssetArray serialization."""

    def test_to_dict_base_array(self):
        """Test to_dict for AssetArray."""
        arr = AssetArray([Asset(name="A1"), Asset(name="A2")])
        d = arr.to_dict()

        assert d["type"] == "AssetArray"
        assert len(d["assets"]) == 2

    def test_to_dict_satellite_array(self):
        """Test to_dict for SatelliteArray."""
        arr = SatelliteArray([Satellite(name="S1")])
        d = arr.to_dict()

        assert d["type"] == "SatelliteArray"
        assert len(d["assets"]) == 1

    def test_from_dict_base_array(self):
        """Test from_dict for AssetArray."""
        arr1 = AssetArray([Asset(name="Test")])
        d = arr1.to_dict()

        arr2 = AssetArray.from_dict(d)

        assert isinstance(arr2, AssetArray)
        assert len(arr2) == 1
        assert arr2[0].name == "Test"

    def test_from_dict_satellite_array(self):
        """Test from_dict reconstructs SatelliteArray."""
        arr1 = SatelliteArray([Satellite(name="Sat1"), Satellite(name="Sat2")])
        d = arr1.to_dict()

        arr2 = AssetArray.from_dict(d)

        assert isinstance(arr2, SatelliteArray)
        assert len(arr2) == 2

    def test_from_dict_groundstation_array(self):
        """Test from_dict reconstructs GroundstationArray."""
        arr1 = GroundstationArray([Groundstation(name="GS1")])
        d = arr1.to_dict()

        arr2 = AssetArray.from_dict(d)

        assert isinstance(arr2, GroundstationArray)
        assert len(arr2) == 1

    def test_serialization_round_trip_mixed(self):
        """Test serialization round trip with mixed types."""
        arr1 = AssetArray([Satellite(name="S"), Groundstation(name="G")])
        d = arr1.to_dict()

        arr2 = AssetArray.from_dict(d)

        assert type(arr2) == AssetArray
        assert len(arr2) == 2
        assert isinstance(arr2[0], Satellite)
        assert isinstance(arr2[1], Groundstation)


class TestAssetArrayEdgeCases:
    """Test AssetArray edge cases."""

    def test_single_element_array(self):
        """Test array with single element."""
        arr = AssetArray([Asset()])
        assert len(arr) == 1

    def test_large_array(self):
        """Test array with many elements."""
        assets = [Asset() for _ in range(100)]
        arr = AssetArray(assets)
        assert len(arr) == 100

    def test_copy_is_independent(self):
        """Test that copy creates independent array."""
        arr1 = AssetArray([Asset(), Asset()])
        arr2 = arr1.copy()

        # Modify internal list of arr1 should not affect arr2
        arr1._assets.append(Asset())

        assert len(arr1) == 3
        assert len(arr2) == 2
