"""Comprehensive test suite for the LLA class.

This module tests all functionality of the LLA class including:
- Construction methods
- Properties (lat, long, alt)
- Coordinate accessors
- Position conversions (ECEF, ECI)
- Operators
- Formatting
- Edge cases
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.state import LLA
from comet.time import Epoch


class TestLLAConstruction:
    """Tests for LLA construction methods."""

    def test_construction_three_components(self):
        """Test construction from 3 individual components."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.lat == 45.0
        assert lla.long == -122.0
        assert lla.alt == 0.1

    def test_construction_single_array(self):
        """Test construction from single 3-element array."""
        arr = [45.0, -122.0, 0.1]
        lla = LLA(arr)
        assert np.allclose(lla._raw, arr)

    def test_construction_numpy_array(self):
        """Test construction with numpy arrays."""
        arr = np.array([45.0, -122.0, 0.1])
        lla = LLA(arr)
        assert np.allclose(lla._raw, arr)

    def test_construction_invalid_num_args(self):
        """Test that invalid number of arguments raises error."""
        with pytest.raises(ValueError):
            LLA(1, 2)

    def test_construction_invalid_types_three_args(self):
        """Test that invalid types for 3 args raises TypeError."""
        with pytest.raises(TypeError):
            LLA("a", 0, 0)

    def test_construction_wrong_array_length(self):
        """Test that wrong array length raises TypeError."""
        with pytest.raises(TypeError):
            LLA([1, 2])  # Only 2 elements

    def test_construction_invalid_array_elements(self):
        """Test that invalid array element types raise TypeError."""
        with pytest.raises(TypeError):
            LLA([1, "a", 3])


class TestLLAProperties:
    """Tests for LLA property access methods."""

    def test_property_lat(self):
        """Test lat property."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.lat == 45.0

    def test_property_long(self):
        """Test long property."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.long == -122.0

    def test_property_alt(self):
        """Test alt property."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.alt == 0.1

    def test_get_latitude(self):
        """Test get_latitude() returns correct value."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.get_latitude() == 45.0

    def test_get_longitude(self):
        """Test get_longitude() returns correct value."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.get_longitude() == -122.0

    def test_get_altitude(self):
        """Test get_altitude() returns correct value."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.get_altitude() == 0.1


class TestLLAPositionConversions:
    """Tests for LLA position conversion methods."""

    def test_ecef_position(self):
        """Test ecef_position() returns 3-element array."""
        lla = LLA(0, 0, 0)  # Equator at sea level
        ecef = lla.ecef_position()
        assert isinstance(ecef, np.ndarray)
        assert len(ecef) == 3

    def test_ecef_position_equator(self):
        """Test ecef_position() at equator gives reasonable values."""
        lla = LLA(0, 0, 0)  # Equator at sea level
        ecef = lla.ecef_position()
        # At equator, x should be ~ Earth radius, y and z should be ~0
        assert ecef[0] > 6000  # x > 6000 km
        assert abs(ecef[1]) < 1  # y ~ 0
        assert abs(ecef[2]) < 1  # z ~ 0

    def test_ecef_position_north_pole(self):
        """Test ecef_position() at north pole gives reasonable values."""
        lla = LLA(90, 0, 0)  # North pole at sea level
        ecef = lla.ecef_position()
        # At north pole, z should be dominant, x and y should be small relative to z
        # Note: Earth is oblate, polar radius (~6357 km) < equatorial radius (~6378 km)
        assert ecef[2] > 5600  # z > 5600 km (polar radius)
        assert abs(ecef[2]) > abs(ecef[0])  # z dominant over x
        assert abs(ecef[2]) > abs(ecef[1])  # z dominant over y

    def test_eci_position_single_epoch(self):
        """Test eci_position() with single epoch."""
        lla = LLA(0, 0, 0)
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        eci = lla.eci_position(epoch)
        assert isinstance(eci, np.ndarray)
        assert len(eci) == 3

    def test_eci_position_multiple_epochs(self):
        """Test eci_position() with multiple epochs."""
        lla = LLA(0, 0, 0)
        epochs = [
            Epoch(2000, 1, 1, 12, 0, 0),
            Epoch(2000, 1, 1, 18, 0, 0),
        ]
        eci = lla.eci_position(epochs)
        assert isinstance(eci, np.ndarray)
        assert eci.shape == (2, 3)


class TestLLAUtilityMethods:
    """Tests for LLA utility methods."""

    def test_copy(self):
        """Test that copy() creates independent LLA."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla2 = lla1.copy()
        assert lla1 == lla2
        assert lla1 is not lla2
        lla2[0] = 50.0
        assert lla1.lat != lla2.lat

    def test_to_dict(self):
        """Test to_dict() serialization."""
        lla = LLA(45.0, -122.0, 0.1)
        result = lla.to_dict()
        assert result["type"] == "LLA"
        assert "coordinates" in result
        assert len(result["coordinates"]) == 3

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        data = {"type": "LLA", "coordinates": [45.0, -122.0, 0.1]}
        lla = LLA.from_dict(data)
        assert np.allclose(lla._raw, data["coordinates"])

    def test_from_dict_invalid_type(self):
        """Test that from_dict() with invalid type raises ValueError."""
        data = {"type": "NotLLA", "coordinates": [45.0, -122.0, 0.1]}
        with pytest.raises(ValueError):
            LLA.from_dict(data)

    def test_dict_roundtrip(self):
        """Test that to_dict() -> from_dict() roundtrip preserves value."""
        lla1 = LLA(45.0, -122.0, 0.1)
        data = lla1.to_dict()
        lla2 = LLA.from_dict(data)
        assert lla1 == lla2


class TestLLAOperators:
    """Tests for LLA operators."""

    def test_equality_lla(self):
        """Test equality operator with LLA."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla2 = LLA(45.0, -122.0, 0.1)
        assert lla1 == lla2

    def test_equality_array(self):
        """Test equality operator with array."""
        lla1 = LLA(45.0, -122.0, 0.1)
        arr = [45.0, -122.0, 0.1]
        assert lla1 == arr

    def test_inequality(self):
        """Test inequality operator."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla2 = LLA(46.0, -122.0, 0.1)
        assert lla1 != lla2

    def test_comparison_not_implemented(self):
        """Test that <, <=, >, >= raise NotImplementedError."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla2 = LLA(46.0, -122.0, 0.1)
        with pytest.raises(NotImplementedError):
            _ = lla1 < lla2
        with pytest.raises(NotImplementedError):
            _ = lla1 <= lla2
        with pytest.raises(NotImplementedError):
            _ = lla1 > lla2
        with pytest.raises(NotImplementedError):
            _ = lla1 >= lla2

    def test_floating_point_comparison(self):
        """Test that floating point comparison uses allclose."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla2 = LLA(45.0 + 1e-10, -122.0, 0.1)
        # Should be equal within tolerance
        assert lla1 == lla2


class TestLLAIndexing:
    """Tests for LLA indexing operations."""

    def test_getitem(self):
        """Test __getitem__ for reading components."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla[0] == 45.0
        assert lla[1] == -122.0
        assert lla[2] == 0.1
        assert lla[-1] == 0.1

    def test_setitem(self):
        """Test __setitem__ for modifying components."""
        lla = LLA(45.0, -122.0, 0.1)
        lla[0] = 50.0
        assert lla[0] == 50.0
        assert lla.lat == 50.0

    def test_len(self):
        """Test __len__ returns 3."""
        lla = LLA(45.0, -122.0, 0.1)
        assert len(lla) == 3


class TestLLAFormatting:
    """Tests for LLA string formatting."""

    def test_str_representation(self):
        """Test __str__ representation."""
        lla = LLA(45.0, -122.0, 0.1)
        result = str(lla)
        assert "45" in result
        assert "122" in result
        assert "0.1" in result
        assert "km" in result

    def test_repr_representation(self):
        """Test __repr__ representation."""
        lla = LLA(45.0, -122.0, 0.1)
        result = repr(lla)
        assert "LLA(" in result


class TestLLAHashability:
    """Tests for LLA hashability."""

    def test_hashable(self):
        """Test that LLA is hashable."""
        lla = LLA(45.0, -122.0, 0.1)
        hash_value = hash(lla)
        assert isinstance(hash_value, int)

    def test_equal_lla_same_hash(self):
        """Test that equal LLA have same hash."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla2 = LLA(45.0, -122.0, 0.1)
        assert hash(lla1) == hash(lla2)

    def test_can_use_in_set(self):
        """Test that LLA can be used in a set."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla2 = LLA(46.0, -122.0, 0.1)
        lla_set = {lla1, lla2}
        assert len(lla_set) == 2

    def test_can_use_as_dict_key(self):
        """Test that LLA can be used as dictionary key."""
        lla1 = LLA(45.0, -122.0, 0.1)
        lla_dict = {lla1: "location1"}
        assert lla_dict[lla1] == "location1"


class TestLLAEdgeCases:
    """Tests for LLA edge cases."""

    def test_zero_coordinates(self):
        """Test LLA at origin (0, 0, 0)."""
        lla = LLA(0, 0, 0)
        assert lla.lat == 0
        assert lla.long == 0
        assert lla.alt == 0

    def test_negative_latitude(self):
        """Test LLA with negative latitude (southern hemisphere)."""
        lla = LLA(-45.0, 122.0, 0.1)
        assert lla.lat == -45.0

    def test_negative_longitude(self):
        """Test LLA with negative longitude (western hemisphere)."""
        lla = LLA(45.0, -122.0, 0.1)
        assert lla.long == -122.0

    def test_negative_altitude(self):
        """Test LLA with negative altitude (below sea level)."""
        lla = LLA(45.0, -122.0, -0.5)
        assert lla.alt == -0.5

    def test_north_pole(self):
        """Test LLA at north pole."""
        lla = LLA(90, 0, 0)
        assert lla.lat == 90

    def test_south_pole(self):
        """Test LLA at south pole."""
        lla = LLA(-90, 0, 0)
        assert lla.lat == -90

    def test_antimeridian_positive(self):
        """Test LLA at positive antimeridian."""
        lla = LLA(0, 180, 0)
        assert lla.long == 180

    def test_antimeridian_negative(self):
        """Test LLA at negative antimeridian."""
        lla = LLA(0, -180, 0)
        assert lla.long == -180

    def test_high_altitude(self):
        """Test LLA with very high altitude."""
        lla = LLA(0, 0, 1000)  # 1000 km altitude
        assert lla.alt == 1000

    def test_very_small_values(self):
        """Test LLA with very small values."""
        lla = LLA(1e-6, 1e-6, 1e-6)
        assert np.isclose(lla.lat, 1e-6)
        assert np.isclose(lla.long, 1e-6)
        assert np.isclose(lla.alt, 1e-6)

    def test_ecef_roundtrip_equator(self):
        """Test that ECEF conversion is consistent."""
        lla = LLA(0, 0, 0)
        ecef = lla.ecef_position()
        # ECEF should have 3 components
        assert len(ecef) == 3
        # Position should be at approximately Earth radius
        radius = np.linalg.norm(ecef)
        assert 6300 < radius < 6400  # Earth radius in km

    def test_property_immutability_via_index(self):
        """Test that properties reflect changes made via indexing."""
        lla = LLA(45.0, -122.0, 0.1)
        lla[0] = 50.0
        assert lla.lat == 50.0
        lla[1] = -120.0
        assert lla.long == -120.0
        lla[2] = 0.2
        assert lla.alt == 0.2
