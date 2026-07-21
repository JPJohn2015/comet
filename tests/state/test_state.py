"""Comprehensive test suite for the State class.

This module tests all functionality of the State class including:
- Construction methods
- Properties (position, velocity, orbital elements)
- Operators
- Formatting
- Edge cases
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.state import State
from comet.utilities.constants import Constants as c


class TestStateConstruction:
    """Tests for State construction methods."""

    def test_construction_six_components(self):
        """Test construction from 6 individual components."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        assert state.x == 7000
        assert state.y == 0
        assert state.z == 0
        assert state.vx == 0
        assert state.vy == 7.5
        assert state.vz == 0

    def test_construction_position_velocity_arrays(self):
        """Test construction from position and velocity arrays."""
        pos = [7000, 0, 0]
        vel = [0, 7.5, 0]
        state = State(pos, vel)
        assert np.allclose(state.position(), pos)
        assert np.allclose(state.velocity(), vel)

    def test_construction_single_array(self):
        """Test construction from single 6-element array."""
        arr = [7000, 0, 0, 0, 7.5, 0]
        state = State(arr)
        assert np.allclose(state._raw, arr)

    def test_construction_numpy_array(self):
        """Test construction with numpy arrays."""
        arr = np.array([7000, 0, 0, 0, 7.5, 0])
        state = State(arr)
        assert np.allclose(state._raw, arr)

    def test_construction_invalid_num_args(self):
        """Test that invalid number of arguments raises error."""
        with pytest.raises(ValueError):
            State(1, 2, 3)

    def test_construction_invalid_types_six_args(self):
        """Test that invalid types for 6 args raises TypeError."""
        with pytest.raises(TypeError):
            State("a", 0, 0, 0, 0, 0)

    def test_construction_invalid_types_two_args(self):
        """Test that invalid types for 2 args raises TypeError."""
        with pytest.raises(TypeError):
            State(7000, 0)  # Not arrays

    def test_construction_wrong_array_length(self):
        """Test that wrong array length raises TypeError."""
        with pytest.raises(TypeError):
            State([1, 2, 3, 4, 5])  # Only 5 elements

    def test_construction_invalid_array_elements(self):
        """Test that invalid array element types raise TypeError."""
        with pytest.raises(TypeError):
            State([1, 2, 3, "a", 5, 6])


class TestStateProperties:
    """Tests for State property access methods."""

    def test_position(self):
        """Test position() returns correct vector."""
        state = State(7000, 1000, 500, 0, 7.5, 1.0)
        pos = state.position()
        assert np.allclose(pos, [7000, 1000, 500])

    def test_velocity(self):
        """Test velocity() returns correct vector."""
        state = State(7000, 1000, 500, 0, 7.5, 1.0)
        vel = state.velocity()
        assert np.allclose(vel, [0, 7.5, 1.0])

    def test_individual_components(self):
        """Test individual component properties."""
        state = State(1, 2, 3, 4, 5, 6)
        assert state.x == 1
        assert state.y == 2
        assert state.z == 3
        assert state.vx == 4
        assert state.vy == 5
        assert state.vz == 6

    def test_position_magnitude(self):
        """Test position_magnitude()."""
        state = State(3, 4, 0, 0, 0, 0)
        assert np.isclose(state.position_magnitude(), 5.0)

    def test_velocity_magnitude(self):
        """Test velocity_magnitude()."""
        state = State(0, 0, 0, 3, 4, 0)
        assert np.isclose(state.velocity_magnitude(), 5.0)

    def test_angular_momentum(self):
        """Test angular_momentum() returns correct vector."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        h = state.angular_momentum()
        # r x v = (7000, 0, 0) x (0, 7.5, 0) = (0, 0, 52500)
        assert np.allclose(h, [0, 0, 52500])

    def test_angular_momentum_magnitude(self):
        """Test angular_momentum_magnitude()."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        h_mag = state.angular_momentum_magnitude()
        assert np.isclose(h_mag, 52500.0)


class TestOrbitalElements:
    """Tests for orbital element calculations."""

    def test_semimajor_axis_circular(self):
        """Test semimajor_axis() for circular orbit."""
        # Circular orbit at ~7000 km
        v_circular = np.sqrt(c.MU_EARTH / 7000)
        state = State(7000, 0, 0, 0, v_circular, 0)
        a = state.semimajor_axis()
        assert np.isclose(a, 7000, rtol=1e-3)

    def test_semimajor_axis_elliptical(self):
        """Test semimajor_axis() for elliptical orbit."""
        # Known elliptical orbit
        state = State(7000, 0, 0, 0, 7.5, 0)
        a = state.semimajor_axis()
        assert a > 0  # Elliptical orbit has positive a

    def test_eccentricity_circular(self):
        """Test eccentricity() for circular orbit."""
        v_circular = np.sqrt(c.MU_EARTH / 7000)
        state = State(7000, 0, 0, 0, v_circular, 0)
        e = state.eccentricity()
        assert np.isclose(e, 0.0, atol=1e-6)

    def test_eccentricity_elliptical(self):
        """Test eccentricity() for elliptical orbit."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        e = state.eccentricity()
        assert 0 < e < 1

    def test_inclination_equatorial(self):
        """Test inclination() for equatorial orbit."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        inc = state.inclination()
        assert np.isclose(inc, 0.0, atol=1e-6)

    def test_inclination_polar(self):
        """Test inclination() for polar orbit."""
        state = State(7000, 0, 0, 0, 0, 7.5)
        inc = state.inclination()
        assert np.isclose(inc, np.pi / 2, rtol=1e-3)

    def test_perigee(self):
        """Test perigee() calculation."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        a = state.semimajor_axis()
        e = state.eccentricity()
        perigee = state.perigee()
        expected = a * (1 - e)
        assert np.isclose(perigee, expected)

    def test_apogee(self):
        """Test apogee() calculation."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        a = state.semimajor_axis()
        e = state.eccentricity()
        apogee = state.apogee()
        expected = a * (1 + e)
        assert np.isclose(apogee, expected)

    def test_period(self):
        """Test period() calculation."""
        v_circular = np.sqrt(c.MU_EARTH / 7000)
        state = State(7000, 0, 0, 0, v_circular, 0)
        period = state.period()
        # Period for circular orbit at 7000 km
        expected = 2 * np.pi * np.sqrt(7000**3 / c.MU_EARTH)
        assert np.isclose(period, expected, rtol=1e-3)

    def test_mean_motion(self):
        """Test mean_motion() calculation."""
        v_circular = np.sqrt(c.MU_EARTH / 7000)
        state = State(7000, 0, 0, 0, v_circular, 0)
        n = state.mean_motion()
        expected = np.sqrt(c.MU_EARTH / 7000**3)
        assert np.isclose(n, expected, rtol=1e-3)

    def test_specific_energy(self):
        """Test specific_energy() calculation."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        energy = state.specific_energy()
        a = state.semimajor_axis()
        expected = -c.MU_EARTH / (2 * a)
        assert np.isclose(energy, expected)

    def test_orbit_type_circular(self):
        """Test orbit_type() for circular orbit."""
        v_circular = np.sqrt(c.MU_EARTH / 7000)
        state = State(7000, 0, 0, 0, v_circular, 0)
        assert state.orbit_type() == 'circular'

    def test_orbit_type_elliptical(self):
        """Test orbit_type() for elliptical orbit."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        assert state.orbit_type() == 'elliptical'

    def test_orbit_type_hyperbolic(self):
        """Test orbit_type() for hyperbolic orbit."""
        # High velocity for escape trajectory
        state = State(7000, 0, 0, 0, 15.0, 0)
        assert state.orbit_type() == 'hyperbolic'


class TestStateUtilityMethods:
    """Tests for State utility methods."""

    def test_copy(self):
        """Test that copy() creates independent State."""
        s1 = State(7000, 0, 0, 0, 7.5, 0)
        s2 = s1.copy()
        assert s1 == s2
        assert s1 is not s2
        s2[0] = 8000
        assert s1.x != s2.x

    def test_inside_earth_true(self):
        """Test inside_earth() returns True when inside."""
        state = State(6000, 0, 0, 0, 0, 0)  # Less than Earth radius
        assert state.inside_earth() == True

    def test_inside_earth_false(self):
        """Test inside_earth() returns False when outside."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        assert state.inside_earth() == False

    def test_to_dict(self):
        """Test to_dict() serialization."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        result = state.to_dict()
        assert result["type"] == "State"
        assert "vector" in result
        assert len(result["vector"]) == 6

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        data = {"type": "State", "vector": [7000, 0, 0, 0, 7.5, 0]}
        state = State.from_dict(data)
        assert np.allclose(state._raw, data["vector"])

    def test_from_dict_invalid_type(self):
        """Test that from_dict() with invalid type raises ValueError."""
        data = {"type": "NotState", "vector": [7000, 0, 0, 0, 7.5, 0]}
        with pytest.raises(ValueError):
            State.from_dict(data)

    def test_dict_roundtrip(self):
        """Test that to_dict() -> from_dict() roundtrip preserves value."""
        s1 = State(7000, 1000, 500, 0, 7.5, 1.0)
        data = s1.to_dict()
        s2 = State.from_dict(data)
        assert s1 == s2


class TestStateOperators:
    """Tests for State operators."""

    def test_equality_state(self):
        """Test equality operator with State."""
        s1 = State(7000, 0, 0, 0, 7.5, 0)
        s2 = State(7000, 0, 0, 0, 7.5, 0)
        assert s1 == s2

    def test_equality_array(self):
        """Test equality operator with array."""
        s1 = State(7000, 0, 0, 0, 7.5, 0)
        arr = [7000, 0, 0, 0, 7.5, 0]
        assert s1 == arr

    def test_inequality(self):
        """Test inequality operator."""
        s1 = State(7000, 0, 0, 0, 7.5, 0)
        s2 = State(8000, 0, 0, 0, 7.5, 0)
        assert s1 != s2

    def test_add_state(self):
        """Test addition with State."""
        s1 = State(1, 2, 3, 4, 5, 6)
        s2 = State(10, 20, 30, 40, 50, 60)
        s3 = s1 + s2
        assert np.allclose(s3._raw, [11, 22, 33, 44, 55, 66])

    def test_add_array(self):
        """Test addition with array."""
        s1 = State(1, 2, 3, 4, 5, 6)
        arr = [10, 20, 30, 40, 50, 60]
        s2 = s1 + arr
        assert np.allclose(s2._raw, [11, 22, 33, 44, 55, 66])

    def test_radd(self):
        """Test reverse addition."""
        s1 = State(1, 2, 3, 4, 5, 6)
        arr = [10, 20, 30, 40, 50, 60]
        s2 = arr + s1
        assert np.allclose(s2._raw, [11, 22, 33, 44, 55, 66])

    def test_iadd(self):
        """Test in-place addition."""
        s1 = State(1, 2, 3, 4, 5, 6)
        s2 = State(10, 20, 30, 40, 50, 60)
        s1 += s2
        assert np.allclose(s1._raw, [11, 22, 33, 44, 55, 66])

    def test_subtract_state(self):
        """Test subtraction with State."""
        s1 = State(10, 20, 30, 40, 50, 60)
        s2 = State(1, 2, 3, 4, 5, 6)
        s3 = s1 - s2
        assert np.allclose(s3._raw, [9, 18, 27, 36, 45, 54])

    def test_subtract_array(self):
        """Test subtraction with array."""
        s1 = State(10, 20, 30, 40, 50, 60)
        arr = [1, 2, 3, 4, 5, 6]
        s2 = s1 - arr
        assert np.allclose(s2._raw, [9, 18, 27, 36, 45, 54])

    def test_rsub(self):
        """Test reverse subtraction."""
        s1 = State(1, 2, 3, 4, 5, 6)
        arr = [10, 20, 30, 40, 50, 60]
        s2 = arr - s1
        assert np.allclose(s2._raw, [9, 18, 27, 36, 45, 54])

    def test_isub(self):
        """Test in-place subtraction."""
        s1 = State(10, 20, 30, 40, 50, 60)
        s2 = State(1, 2, 3, 4, 5, 6)
        s1 -= s2
        assert np.allclose(s1._raw, [9, 18, 27, 36, 45, 54])

    def test_multiply_scalar(self):
        """Test multiplication by scalar."""
        s1 = State(1, 2, 3, 4, 5, 6)
        s2 = s1 * 2
        assert np.allclose(s2._raw, [2, 4, 6, 8, 10, 12])

    def test_rmul_scalar(self):
        """Test reverse multiplication."""
        s1 = State(1, 2, 3, 4, 5, 6)
        s2 = 2 * s1
        assert np.allclose(s2._raw, [2, 4, 6, 8, 10, 12])

    def test_imul_scalar(self):
        """Test in-place multiplication."""
        s1 = State(1, 2, 3, 4, 5, 6)
        s1 *= 2
        assert np.allclose(s1._raw, [2, 4, 6, 8, 10, 12])

    def test_divide_scalar(self):
        """Test division by scalar."""
        s1 = State(10, 20, 30, 40, 50, 60)
        s2 = s1 / 2
        assert np.allclose(s2._raw, [5, 10, 15, 20, 25, 30])

    def test_add_invalid_type(self):
        """Test that adding invalid type raises NotImplementedError."""
        s1 = State(1, 2, 3, 4, 5, 6)
        with pytest.raises(NotImplementedError):
            _ = s1 + 5

    def test_multiply_invalid_type(self):
        """Test that multiplying by invalid type raises NotImplementedError."""
        s1 = State(1, 2, 3, 4, 5, 6)
        with pytest.raises(NotImplementedError):
            _ = s1 * "invalid"

    def test_comparison_not_implemented(self):
        """Test that <, <=, >, >= raise NotImplementedError."""
        s1 = State(1, 2, 3, 4, 5, 6)
        s2 = State(2, 3, 4, 5, 6, 7)
        with pytest.raises(NotImplementedError):
            _ = s1 < s2
        with pytest.raises(NotImplementedError):
            _ = s1 <= s2
        with pytest.raises(NotImplementedError):
            _ = s1 > s2
        with pytest.raises(NotImplementedError):
            _ = s1 >= s2

    def test_rtruediv_not_implemented(self):
        """Test that reverse division raises NotImplementedError."""
        s1 = State(1, 2, 3, 4, 5, 6)
        with pytest.raises(NotImplementedError):
            _ = 5 / s1


class TestStateIndexing:
    """Tests for State indexing operations."""

    def test_getitem(self):
        """Test __getitem__ for reading components."""
        state = State(1, 2, 3, 4, 5, 6)
        assert state[0] == 1
        assert state[3] == 4
        assert state[-1] == 6

    def test_setitem(self):
        """Test __setitem__ for modifying components."""
        state = State(1, 2, 3, 4, 5, 6)
        state[0] = 10
        assert state[0] == 10
        assert state.x == 10

    def test_len(self):
        """Test __len__ returns 6."""
        state = State(1, 2, 3, 4, 5, 6)
        assert len(state) == 6


class TestStateFormatting:
    """Tests for State string formatting."""

    def test_str_representation(self):
        """Test __str__ representation."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        result = str(state)
        assert "7000" in result
        assert "7.5" in result or "7.500" in result
        assert "km" in result
        assert "km/s" in result

    def test_repr_representation(self):
        """Test __repr__ representation."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        result = repr(state)
        assert "State(" in result


class TestStateHashability:
    """Tests for State hashability."""

    def test_hashable(self):
        """Test that State is hashable."""
        state = State(7000, 0, 0, 0, 7.5, 0)
        hash_value = hash(state)
        assert isinstance(hash_value, int)

    def test_equal_states_same_hash(self):
        """Test that equal States have same hash."""
        s1 = State(7000, 0, 0, 0, 7.5, 0)
        s2 = State(7000, 0, 0, 0, 7.5, 0)
        assert hash(s1) == hash(s2)

    def test_can_use_in_set(self):
        """Test that State can be used in a set."""
        s1 = State(7000, 0, 0, 0, 7.5, 0)
        s2 = State(8000, 0, 0, 0, 7.5, 0)
        state_set = {s1, s2}
        assert len(state_set) == 2

    def test_can_use_as_dict_key(self):
        """Test that State can be used as dictionary key."""
        s1 = State(7000, 0, 0, 0, 7.5, 0)
        state_dict = {s1: "orbit1"}
        assert state_dict[s1] == "orbit1"


class TestStateEdgeCases:
    """Tests for State edge cases."""

    def test_zero_state(self):
        """Test State with all zeros."""
        state = State(0, 0, 0, 0, 0, 0)
        assert np.allclose(state._raw, [0, 0, 0, 0, 0, 0])

    def test_negative_components(self):
        """Test State with negative components."""
        state = State(-7000, -1000, -500, -1, -2, -3)
        assert state.x == -7000
        assert state.vx == -1

    def test_very_large_values(self):
        """Test State with very large values."""
        state = State(1e6, 1e6, 1e6, 1e3, 1e3, 1e3)
        assert state.x == 1e6

    def test_very_small_values(self):
        """Test State with very small values."""
        state = State(1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9)
        assert np.isclose(state.x, 1e-6)

    def test_floating_point_comparison(self):
        """Test that floating point comparison uses allclose."""
        s1 = State(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0)
        s2 = State(7000.0 + 1e-10, 0.0, 0.0, 0.0, 7.5, 0.0)
        # Should be equal within tolerance
        assert s1 == s2
