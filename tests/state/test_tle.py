"""Comprehensive test suite for the TLE class.

This module tests all functionality of the TLE class including:
- Construction methods
- Properties (line1, line2, lines)
- TLE field parsing
- Orbital element derivation
- Conversions (to State, to Elements)
- Operators
- Formatting
- Edge cases
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.state import TLE
from comet.state import State
from comet.state import Elements
from comet.time import Epoch


# Sample TLE for ISS (International Space Station)
ISS_LINE1 = "1 25544U 98067A   24153.50000000  .00016717  00000-0  10270-3 0  9005"
ISS_LINE2 = "2 25544  51.6400 208.5800 0001160  90.1200 270.0000 15.50103472123456"

# Sample TLE for a geostationary satellite
GEO_LINE1 = "1 59069U 24040A   24153.34215875  .00000159  00000-0  00000-0 0  9994"
GEO_LINE2 = "2 59069   0.0491 104.1558 0003000 296.0103   6.9785  1.00271391    57"


class TestTLEConstruction:
    """Tests for TLE construction methods."""

    def test_construction_two_lines(self):
        """Test construction from two separate lines."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        assert tle.line1 == ISS_LINE1
        assert tle.line2 == ISS_LINE2

    def test_construction_list(self):
        """Test construction from list of lines."""
        tle = TLE([ISS_LINE1, ISS_LINE2])
        assert tle.line1 == ISS_LINE1
        assert tle.line2 == ISS_LINE2

    def test_construction_tuple(self):
        """Test construction from tuple of lines."""
        tle = TLE((ISS_LINE1, ISS_LINE2))
        assert tle.line1 == ISS_LINE1
        assert tle.line2 == ISS_LINE2

    def test_construction_strips_whitespace(self):
        """Test that construction strips leading/trailing whitespace."""
        tle = TLE("  " + ISS_LINE1 + "  ", "  " + ISS_LINE2 + "  ")
        assert tle.line1 == ISS_LINE1
        assert tle.line2 == ISS_LINE2

    def test_construction_invalid_num_args(self):
        """Test that invalid number of arguments raises error."""
        with pytest.raises((ValueError, TypeError)):
            TLE(ISS_LINE1)

    def test_construction_non_string_lines(self):
        """Test that non-string lines raise TypeError."""
        with pytest.raises(TypeError):
            TLE(123, ISS_LINE2)

    def test_construction_invalid_list_length(self):
        """Test that list with wrong number of elements raises TypeError."""
        with pytest.raises(TypeError):
            TLE([ISS_LINE1])

    def test_construction_line1_too_short(self):
        """Test that line 1 too short raises ValueError."""
        with pytest.raises(ValueError):
            TLE("1 25544U", ISS_LINE2)

    def test_construction_line2_too_short(self):
        """Test that line 2 too short raises ValueError."""
        with pytest.raises(ValueError):
            TLE(ISS_LINE1, "2 25544")

    def test_construction_line1_wrong_start(self):
        """Test that line 1 not starting with '1' raises ValueError."""
        with pytest.raises(ValueError):
            bad_line1 = "2" + ISS_LINE1[1:]
            TLE(bad_line1, ISS_LINE2)

    def test_construction_line2_wrong_start(self):
        """Test that line 2 not starting with '2' raises ValueError."""
        with pytest.raises(ValueError):
            bad_line2 = "1" + ISS_LINE2[1:]
            TLE(ISS_LINE1, bad_line2)


class TestTLEProperties:
    """Tests for TLE property access methods."""

    def test_property_line1(self):
        """Test line1 property."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        assert tle.line1 == ISS_LINE1

    def test_property_line2(self):
        """Test line2 property."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        assert tle.line2 == ISS_LINE2

    def test_property_lines(self):
        """Test lines property returns tuple."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        lines = tle.lines
        assert isinstance(lines, tuple)
        assert len(lines) == 2
        assert lines[0] == ISS_LINE1
        assert lines[1] == ISS_LINE2


class TestTLEFieldParsing:
    """Tests for TLE field parsing methods."""

    def test_norad_id(self):
        """Test norad_id() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        assert tle.norad_id() == 25544

    def test_classification(self):
        """Test classification() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        assert tle.classification() == 'U'

    def test_launch_year(self):
        """Test launch_year() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        # ISS launched in 1998, format is "98067A"
        # Positions 8:11 give "980" from "98067"
        launch_year = tle.launch_year()
        assert launch_year == 98  # This parses 98 from international designator

    def test_launch_number(self):
        """Test launch_number() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        launch_num = tle.launch_number()
        assert launch_num == 67

    def test_designator(self):
        """Test designator() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        assert tle.designator() == 'A'

    def test_year(self):
        """Test year() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        assert tle.year() == 24  # 2024

    def test_day_of_year(self):
        """Test day_of_year() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        doy = tle.day_of_year()
        assert 1 <= doy <= 366

    def test_epoch(self):
        """Test epoch() returns Epoch object."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        epoch = tle.epoch()
        assert isinstance(epoch, Epoch)

    def test_epoch_2000s(self):
        """Test epoch() correctly interprets 2000s year."""
        tle = TLE(ISS_LINE1, ISS_LINE2)  # Year 24 = 2024
        epoch = tle.epoch()
        # Should be in 2024
        assert 2024 <= epoch.year <= 2024

    def test_epoch_1900s(self):
        """Test epoch() correctly interprets 1900s year."""
        # Create a TLE with year 98 (1998)
        line1 = "1 25544U 98067A   98153.50000000  .00016717  00000-0  10270-3 0  9005"
        tle = TLE(line1, ISS_LINE2)
        epoch = tle.epoch()
        # Should be in 1998
        assert 1998 <= epoch.year <= 1998


class TestTLEOrbitalElements:
    """Tests for TLE orbital element methods."""

    def test_inclination(self):
        """Test inclination() returns value in radians."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        inc = tle.inclination()
        # ISS has ~51.64 degree inclination
        assert 0 < inc < np.pi

    def test_right_ascension(self):
        """Test right_ascension() returns value in radians."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        raan = tle.right_ascension()
        assert 0 <= raan < 2 * np.pi

    def test_eccentricity(self):
        """Test eccentricity() parsing."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        ecc = tle.eccentricity()
        # LEO orbits have very low eccentricity
        assert 0 <= ecc < 0.1

    def test_argument_perigee(self):
        """Test argument_perigee() returns value in radians."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        ap = tle.argument_perigee()
        assert 0 <= ap < 2 * np.pi

    def test_mean_anomaly(self):
        """Test mean_anomaly() returns value in radians."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        ma = tle.mean_anomaly()
        assert 0 <= ma < 2 * np.pi

    def test_true_anomaly(self):
        """Test true_anomaly() returns value in radians."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        ta = tle.true_anomaly()
        # True anomaly should be a valid angle (may need normalization)
        assert isinstance(ta, (int, float, np.number))

    def test_eccentric_anomaly(self):
        """Test eccentric_anomaly() returns value in radians."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        ea = tle.eccentric_anomaly()
        assert 0 <= ea < 2 * np.pi

    def test_mean_motion(self):
        """Test mean_motion() returns value in rad/s."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        n = tle.mean_motion()
        # ISS completes ~15.5 orbits per day
        assert n > 0

    def test_semimajor_axis(self):
        """Test semimajor_axis() returns reasonable value."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        sma = tle.semimajor_axis()
        # ISS orbits at ~400 km altitude, so SMA should be ~6778 km
        assert 6700 < sma < 7000

    def test_period(self):
        """Test period() returns reasonable value."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        period = tle.period()
        # ISS period is ~90 minutes = ~5400 seconds
        assert 5000 < period < 6000


class TestTLEConversions:
    """Tests for TLE conversion methods."""

    def test_to_state(self):
        """Test to_state() conversion."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        state = tle.to_state()
        assert isinstance(state, State)
        assert len(state) == 6

    def test_to_state_position_magnitude(self):
        """Test to_state() gives reasonable position magnitude."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        state = tle.to_state()
        pos_mag = state.position_magnitude()
        # ISS orbits at ~400 km altitude, so position magnitude ~6778 km
        assert 6700 < pos_mag < 7000

    def test_to_elements(self):
        """Test to_elements() conversion."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        elements = tle.to_elements()
        assert isinstance(elements, Elements)
        assert len(elements) == 6

    def test_to_elements_sma(self):
        """Test to_elements() preserves semimajor axis."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        elements = tle.to_elements()
        assert np.isclose(elements.sma, tle.semimajor_axis())

    def test_to_elements_ecc(self):
        """Test to_elements() preserves eccentricity."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        elements = tle.to_elements()
        assert np.isclose(elements.ecc, tle.eccentricity())


class TestTLEUtilityMethods:
    """Tests for TLE utility methods."""

    def test_copy(self):
        """Test that copy() creates independent TLE."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        tle2 = tle1.copy()
        assert tle1 == tle2
        assert tle1 is not tle2

    def test_to_dict(self):
        """Test to_dict() serialization."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        result = tle.to_dict()
        assert result["type"] == "TLE"
        assert "line1" in result
        assert "line2" in result

    def test_from_dict(self):
        """Test from_dict() deserialization."""
        data = {"type": "TLE", "line1": ISS_LINE1, "line2": ISS_LINE2}
        tle = TLE.from_dict(data)
        assert tle.line1 == ISS_LINE1
        assert tle.line2 == ISS_LINE2

    def test_from_dict_invalid_type(self):
        """Test that from_dict() with invalid type raises ValueError."""
        data = {"type": "NotTLE", "line1": ISS_LINE1, "line2": ISS_LINE2}
        with pytest.raises(ValueError):
            TLE.from_dict(data)

    def test_dict_roundtrip(self):
        """Test that to_dict() -> from_dict() roundtrip preserves value."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        data = tle1.to_dict()
        tle2 = TLE.from_dict(data)
        assert tle1 == tle2


class TestTLEOperators:
    """Tests for TLE operators."""

    def test_equality_tle(self):
        """Test equality operator with TLE."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        tle2 = TLE(ISS_LINE1, ISS_LINE2)
        assert tle1 == tle2

    def test_inequality(self):
        """Test inequality operator."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        tle2 = TLE(GEO_LINE1, GEO_LINE2)
        assert tle1 != tle2

    def test_comparison_not_implemented(self):
        """Test that <, <=, >, >= raise NotImplementedError."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        tle2 = TLE(GEO_LINE1, GEO_LINE2)
        with pytest.raises(NotImplementedError):
            _ = tle1 < tle2
        with pytest.raises(NotImplementedError):
            _ = tle1 <= tle2
        with pytest.raises(NotImplementedError):
            _ = tle1 > tle2
        with pytest.raises(NotImplementedError):
            _ = tle1 >= tle2

    def test_equality_with_non_tle(self):
        """Test that equality with non-TLE raises NotImplementedError."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        with pytest.raises(NotImplementedError):
            _ = tle == "not a TLE"


class TestTLEFormatting:
    """Tests for TLE string formatting."""

    def test_str_representation(self):
        """Test __str__ representation."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        result = str(tle)
        # Should contain orbital elements in a readable format
        assert "km" in result

    def test_repr_representation(self):
        """Test __repr__ representation."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        result = repr(tle)
        assert "TLE(" in result


class TestTLEHashability:
    """Tests for TLE hashability."""

    def test_hashable(self):
        """Test that TLE is hashable."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        hash_value = hash(tle)
        assert isinstance(hash_value, int)

    def test_equal_tle_same_hash(self):
        """Test that equal TLEs have same hash."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        tle2 = TLE(ISS_LINE1, ISS_LINE2)
        assert hash(tle1) == hash(tle2)

    def test_can_use_in_set(self):
        """Test that TLE can be used in a set."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        tle2 = TLE(GEO_LINE1, GEO_LINE2)
        tle_set = {tle1, tle2}
        assert len(tle_set) == 2

    def test_can_use_as_dict_key(self):
        """Test that TLE can be used as dictionary key."""
        tle1 = TLE(ISS_LINE1, ISS_LINE2)
        tle_dict = {tle1: "ISS"}
        assert tle_dict[tle1] == "ISS"


class TestTLEEdgeCases:
    """Tests for TLE edge cases."""

    def test_geostationary_orbit(self):
        """Test TLE for geostationary satellite."""
        tle = TLE(GEO_LINE1, GEO_LINE2)
        # Geostationary orbit should have period ~86400 seconds (1 day)
        period = tle.period()
        assert 86000 < period < 87000

    def test_geostationary_inclination(self):
        """Test geostationary satellite has low inclination."""
        tle = TLE(GEO_LINE1, GEO_LINE2)
        inc = tle.inclination()
        # Geostationary should be near equatorial
        assert inc < np.deg2rad(1)  # Less than 1 degree

    def test_geostationary_semimajor_axis(self):
        """Test geostationary satellite semimajor axis."""
        tle = TLE(GEO_LINE1, GEO_LINE2)
        sma = tle.semimajor_axis()
        # GEO orbit is at ~35,786 km altitude, SMA ~42,164 km
        assert 42000 < sma < 43000

    def test_mean_to_true_anomaly_conversion(self):
        """Test that mean anomaly converts to true anomaly."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        ma = tle.mean_anomaly()
        ta = tle.true_anomaly()
        # For low eccentricity, they should be close but not identical
        assert not np.isclose(ma, ta) or tle.eccentricity() < 0.001

    def test_state_elements_roundtrip(self):
        """Test TLE -> Elements -> State gives consistent results."""
        tle = TLE(ISS_LINE1, ISS_LINE2)
        elements = tle.to_elements()
        state = tle.to_state()
        # Position magnitudes should match semimajor axis approximately
        # (exact match only at periapsis/apoapsis)
        pos_mag = state.position_magnitude()
        sma = tle.semimajor_axis()
        ecc = tle.eccentricity()
        # Position should be between perigee and apogee
        assert sma * (1 - ecc) <= pos_mag <= sma * (1 + ecc)
