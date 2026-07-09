"""Comprehensive test suite for the Elements class.

This module tests all functionality of the Elements class including:
- Construction methods
- Properties (a, e, i, O, w, v)
- Orbital element accessors
- Derived orbital properties
- Operators
- Formatting
- Edge cases
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.state.elements import Elements
from comet.state.state import State
from comet.utilities.constants import Constants as c


class TestElementsConstruction:
    """Tests for Elements construction methods."""

    def test_construction_six_components(self):
        """Test construction from 6 individual components."""
        elements = Elements(7000, 0.001, np.deg2rad(45), np.deg2rad(30), np.deg2rad(60), np.deg2rad(90))
        assert elements.sma == 7000
        assert elements.ecc == 0.001
        assert np.isclose(elements.inc, np.deg2rad(45))
        assert np.isclose(elements.raan, np.deg2rad(30))
        assert np.isclose(elements.ap, np.deg2rad(60))
        assert np.isclose(elements.ta, np.deg2rad(90))

    def test_construction_single_array(self):
        """Test construction from single 6-element array."""
        arr = [7000, 0.001, 0.1, 0.2, 0.3, 0.4]
        elements = Elements(arr)
        assert np.allclose(elements._raw, arr)

    def test_construction_numpy_array(self):
        """Test construction with numpy arrays."""
        arr = np.array([7000, 0.001, 0.1, 0.2, 0.3, 0.4])
        elements = Elements(arr)
        assert np.allclose(elements._raw, arr)

    def test_construction_invalid_num_args(self):
        """Test that invalid number of arguments raises error."""
        with pytest.raises(ValueError):
            Elements(1, 2, 3)

    def test_construction_invalid_types_six_args(self):
        """Test that invalid types for 6 args raises TypeError."""
        with pytest.raises(TypeError):
            Elements("a", 0, 0, 0, 0, 0)

    def test_construction_wrong_array_length(self):
        """Test that wrong array length raises TypeError."""
        with pytest.raises(TypeError):
            Elements([1, 2, 3, 4, 5])  # Only 5 elements

    def test_construction_invalid_array_elements(self):
        """Test that invalid array element types raise TypeError."""
        with pytest.raises(TypeError):
            Elements([1, 2, 3, "a", 5, 6])


class TestElementsProperties:
    """Tests for Elements property access methods."""

    def test_property_sma(self):
        """Test sma property."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert elements.sma == 7000

    def test_property_ecc(self):
        """Test ecc property."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert elements.ecc == 0.001

    def test_property_inc(self):
        """Test inc property."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.inc, 0.1)

    def test_property_raan(self):
        """Test raan property."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.raan, 0.2)

    def test_property_ap(self):
        """Test ap property."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.ap, 0.3)

    def test_property_ta(self):
        """Test ta property."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.ta, 0.4)

    def test_property_ma(self):
        """Test ma property."""
        elements = Elements(7000, 0.1, 0, 0, 0, np.deg2rad(45))
        ma = elements.ma
        # Mean anomaly should be less than true anomaly for e > 0
        assert 0 <= ma < np.deg2rad(45)

    def test_semi_major_axis(self):
        """Test semi_major_axis() returns correct value."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert elements.semi_major_axis() == 7000

    def test_eccentricity(self):
        """Test eccentricity() returns correct value."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert elements.eccentricity() == 0.001

    def test_inclination(self):
        """Test inclination() returns correct value."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.inclination(), 0.1)

    def test_right_ascension(self):
        """Test right_ascension() returns correct value."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.right_ascension(), 0.2)

    def test_argument_perigee(self):
        """Test argument_perigee() returns correct value."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.argument_perigee(), 0.3)

    def test_true_anomaly(self):
        """Test true_anomaly() returns correct value."""
        elements = Elements(7000, 0.001, 0.1, 0.2, 0.3, 0.4)
        assert np.isclose(elements.true_anomaly(), 0.4)


class TestElementsDerivedProperties:
    """Tests for derived orbital properties."""

    def test_mean_anomaly(self):
        """Test mean_anomaly() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, np.deg2rad(45))
        ma = elements.mean_anomaly()
        # Mean anomaly should be less than true anomaly for e > 0
        assert 0 <= ma < np.deg2rad(45)

    def test_eccentric_anomaly(self):
        """Test eccentric_anomaly() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, np.deg2rad(45))
        ea = elements.eccentric_anomaly()
        # Eccentric anomaly should be between mean and true anomaly
        ma = elements.mean_anomaly()
        assert ma < ea < np.deg2rad(45)

    def test_flight_path_angle_at_periapsis(self):
        """Test flight_path_angle() at periapsis."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)  # v=0 at periapsis
        fpa = elements.flight_path_angle()
        assert np.isclose(fpa, 0.0, atol=1e-6)

    def test_flight_path_angle_at_apoapsis(self):
        """Test flight_path_angle() at apoapsis."""
        elements = Elements(7000, 0.1, 0, 0, 0, np.pi)  # v=pi at apoapsis
        fpa = elements.flight_path_angle()
        assert np.isclose(fpa, 0.0, atol=1e-6)

    def test_longitude_of_periapsis(self):
        """Test longitude_of_periapsis() calculation."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        lp = elements.longitude_of_periapsis()
        expected = 0.2 + 0.3  # O + w
        assert np.isclose(lp, expected)

    def test_true_longitude(self):
        """Test true_longitude() calculation."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        tl = elements.true_longitude()
        expected = 0.2 + 0.3 + 0.4  # O + w + v
        assert np.isclose(tl, expected)


class TestElementsOrbitalCharacteristics:
    """Tests for orbital characteristic calculations."""

    def test_apogee(self):
        """Test apogee() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)
        apogee = elements.apogee()
        expected = 7000 * (1 + 0.1)
        assert np.isclose(apogee, expected)

    def test_perigee(self):
        """Test perigee() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)
        perigee = elements.perigee()
        expected = 7000 * (1 - 0.1)
        assert np.isclose(perigee, expected)

    def test_semi_parameter(self):
        """Test semi_parameter() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)
        p = elements.semi_parameter()
        expected = 7000 * (1 - 0.1**2)
        assert np.isclose(p, expected)

    def test_specific_energy(self):
        """Test specific_energy() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)
        energy = elements.specific_energy()
        expected = -c.MU_EARTH / (2 * 7000)
        assert np.isclose(energy, expected)

    def test_radius_at_periapsis(self):
        """Test radius() at periapsis."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)  # v=0
        r = elements.radius()
        expected = elements.perigee()
        assert np.isclose(r, expected)

    def test_radius_at_apoapsis(self):
        """Test radius() at apoapsis."""
        elements = Elements(7000, 0.1, 0, 0, 0, np.pi)  # v=pi
        r = elements.radius()
        expected = elements.apogee()
        assert np.isclose(r, expected)

    def test_mean_motion(self):
        """Test mean_motion() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)
        n = elements.mean_motion()
        expected = np.sqrt(c.MU_EARTH / 7000**3)
        assert np.isclose(n, expected)

    def test_period(self):
        """Test period() calculation."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)
        period = elements.period()
        expected = 2 * np.pi * np.sqrt(7000**3 / c.MU_EARTH)
        assert np.isclose(period, expected)


class TestElementsUtilityMethods:
    """Tests for Elements utility methods."""

    def test_copy(self):
        """Test that copy() creates independent Elements."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        e2 = e1.copy()
        assert e1 == e2
        assert e1 is not e2
        e2[0] = 8000
        assert e1.sma != e2.sma

    def test_inside_earth_true(self):
        """Test inside_earth() returns True when inside."""
        # Small orbit at 6000 km with high eccentricity at periapsis
        elements = Elements(6000, 0.01, 0, 0, 0, 0)
        assert elements.inside_earth() == True

    def test_inside_earth_false(self):
        """Test inside_earth() returns False when outside."""
        # Use a=8000, e=0.1 -> perigee = 7200 km > Earth radius
        elements = Elements(8000, 0.1, 0, 0, 0, 0)
        assert elements.inside_earth() == False

    def test_orbit_type_circular(self):
        """Test orbit_type() for circular orbit."""
        elements = Elements(7000, 0.001, 0, 0, 0, 0)
        assert elements.orbit_type() == 'circular'

    def test_orbit_type_elliptical(self):
        """Test orbit_type() for elliptical orbit."""
        elements = Elements(7000, 0.5, 0, 0, 0, 0)
        assert elements.orbit_type() == 'elliptical'

    def test_orbit_type_parabolic(self):
        """Test orbit_type() for parabolic orbit."""
        elements = Elements(7000, 1.0, 0, 0, 0, 0)
        assert elements.orbit_type() == 'parabolic'

    def test_orbit_type_hyperbolic(self):
        """Test orbit_type() for hyperbolic orbit."""
        elements = Elements(7000, 1.5, 0, 0, 0, 0)
        assert elements.orbit_type() == 'hyperbolic'

    def test_to_state(self):
        """Test to_state() conversion."""
        elements = Elements(7000, 0.1, 0, 0, 0, 0)
        state = elements.to_state()
        assert isinstance(state, State)
        # Check that state has position and velocity
        assert len(state) == 6

    def test_to_dict(self):
        """Test to_dict() serialization."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        result = elements.to_dict()
        assert result["type"] == "Elements"
        assert "vector" in result
        assert len(result["vector"]) == 6

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        data = {"type": "Elements", "vector": [7000, 0.1, 0.1, 0.2, 0.3, 0.4]}
        elements = Elements.from_dict(data)
        assert np.allclose(elements._raw, data["vector"])

    def test_from_dict_invalid_type(self):
        """Test that from_dict() with invalid type raises ValueError."""
        data = {"type": "NotElements", "vector": [7000, 0.1, 0.1, 0.2, 0.3, 0.4]}
        with pytest.raises(ValueError):
            Elements.from_dict(data)

    def test_dict_roundtrip(self):
        """Test that to_dict() -> from_dict() roundtrip preserves value."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        data = e1.to_dict()
        e2 = Elements.from_dict(data)
        assert e1 == e2


class TestElementsOperators:
    """Tests for Elements operators."""

    def test_equality_elements(self):
        """Test equality operator with Elements."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        e2 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        assert e1 == e2

    def test_equality_array(self):
        """Test equality operator with array."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        arr = [7000, 0.1, 0.1, 0.2, 0.3, 0.4]
        assert e1 == arr

    def test_inequality(self):
        """Test inequality operator."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        e2 = Elements(8000, 0.1, 0.1, 0.2, 0.3, 0.4)
        assert e1 != e2

    def test_comparison_not_implemented(self):
        """Test that <, <=, >, >= raise NotImplementedError."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        e2 = Elements(8000, 0.1, 0.1, 0.2, 0.3, 0.4)
        with pytest.raises(NotImplementedError):
            _ = e1 < e2
        with pytest.raises(NotImplementedError):
            _ = e1 <= e2
        with pytest.raises(NotImplementedError):
            _ = e1 > e2
        with pytest.raises(NotImplementedError):
            _ = e1 >= e2

    def test_floating_point_comparison(self):
        """Test that floating point comparison uses allclose."""
        e1 = Elements(7000.0, 0.1, 0.1, 0.2, 0.3, 0.4)
        e2 = Elements(7000.0 + 1e-10, 0.1, 0.1, 0.2, 0.3, 0.4)
        # Should be equal within tolerance
        assert e1 == e2


class TestElementsIndexing:
    """Tests for Elements indexing operations."""

    def test_getitem(self):
        """Test __getitem__ for reading components."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        assert elements[0] == 7000
        assert elements[1] == 0.1
        assert elements[-1] == 0.4

    def test_setitem(self):
        """Test __setitem__ for modifying components."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        elements[0] = 8000
        assert elements[0] == 8000
        assert elements.sma == 8000

    def test_len(self):
        """Test __len__ returns 6."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        assert len(elements) == 6


class TestElementsFormatting:
    """Tests for Elements string formatting."""

    def test_str_representation(self):
        """Test __str__ representation."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        result = str(elements)
        assert "7000" in result
        assert "0.1" in result or "0.100000" in result

    def test_repr_representation(self):
        """Test __repr__ representation."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        result = repr(elements)
        assert "Elements(" in result


class TestElementsHashability:
    """Tests for Elements hashability."""

    def test_hashable(self):
        """Test that Elements is hashable."""
        elements = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        hash_value = hash(elements)
        assert isinstance(hash_value, int)

    def test_equal_elements_same_hash(self):
        """Test that equal Elements have same hash."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        e2 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        assert hash(e1) == hash(e2)

    def test_can_use_in_set(self):
        """Test that Elements can be used in a set."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        e2 = Elements(8000, 0.1, 0.1, 0.2, 0.3, 0.4)
        elements_set = {e1, e2}
        assert len(elements_set) == 2

    def test_can_use_as_dict_key(self):
        """Test that Elements can be used as dictionary key."""
        e1 = Elements(7000, 0.1, 0.1, 0.2, 0.3, 0.4)
        elements_dict = {e1: "orbit1"}
        assert elements_dict[e1] == "orbit1"


class TestElementsEdgeCases:
    """Tests for Elements edge cases."""

    def test_zero_eccentricity(self):
        """Test Elements with zero eccentricity (circular)."""
        elements = Elements(7000, 0, 0, 0, 0, 0)
        assert elements.ecc == 0
        assert elements.orbit_type() == 'circular'

    def test_negative_semi_major_axis(self):
        """Test Elements with negative semi-major axis (hyperbolic)."""
        elements = Elements(-7000, 1.5, 0, 0, 0, 0)
        assert elements.sma < 0
        assert elements.orbit_type() == 'hyperbolic'

    def test_very_large_semi_major_axis(self):
        """Test Elements with very large semi-major axis."""
        elements = Elements(1e6, 0.1, 0, 0, 0, 0)
        assert elements.sma == 1e6
        # Period should be very long
        assert elements.period() > 1e6

    def test_high_eccentricity_near_parabolic(self):
        """Test Elements with eccentricity very close to 1."""
        elements = Elements(7000, 0.9999, 0, 0, 0, 0)
        assert elements.orbit_type() == 'elliptical'

    def test_angles_in_radians(self):
        """Test that angles are stored in radians."""
        i_deg = 45
        O_deg = 90
        w_deg = 180
        v_deg = 270
        elements = Elements(
            7000, 0.1,
            np.deg2rad(i_deg),
            np.deg2rad(O_deg),
            np.deg2rad(w_deg),
            np.deg2rad(v_deg)
        )
        assert np.isclose(elements.inc, np.deg2rad(i_deg))
        assert np.isclose(elements.raan, np.deg2rad(O_deg))
        assert np.isclose(elements.ap, np.deg2rad(w_deg))
        assert np.isclose(elements.ta, np.deg2rad(v_deg))

    def test_equatorial_orbit(self):
        """Test Elements with zero inclination (equatorial)."""
        elements = Elements(7000, 0.1, 0, 0.2, 0.3, 0.4)
        assert elements.inc == 0

    def test_polar_orbit(self):
        """Test Elements with 90 degree inclination (polar)."""
        elements = Elements(7000, 0.1, np.pi/2, 0.2, 0.3, 0.4)
        assert np.isclose(elements.inc, np.pi/2)

    def test_state_conversion_roundtrip(self):
        """Test Elements -> State -> Elements roundtrip."""
        e1 = Elements(7000, 0.1, np.deg2rad(30), np.deg2rad(45), np.deg2rad(60), np.deg2rad(90))
        state = e1.to_state()
        e2 = state.to_elements()
        # Check major orbital elements are preserved (within tolerance)
        assert np.isclose(e1.sma, e2.sma, rtol=1e-3)
        assert np.isclose(e1.ecc, e2.ecc, atol=1e-6)
        assert np.isclose(e1.inc, e2.inc, atol=1e-6)
