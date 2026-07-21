"""Comprehensive test suite for the Duration class.

This module tests all functionality of the Duration class including:
- Construction methods
- Time conversions
- Operators
- Properties
- Formatting
- Edge cases
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.time import Duration
from comet.utilities.constants import Constants as c


class TestDurationConstruction:
    """Tests for Duration construction methods."""

    def test_default_construction(self):
        """Test default construction creates zero duration."""
        duration = Duration()
        assert duration.total_seconds() == 0.0
        assert duration.days == 0
        assert duration.hours == 0
        assert duration.minutes == 0
        assert duration.seconds == 0.0

    def test_construction_days_only(self):
        """Test construction with days only."""
        duration = Duration(days=5)
        assert duration.days == 5
        assert duration.hours == 0
        assert duration.minutes == 0
        assert duration.seconds == 0.0
        assert duration.total_seconds() == 5 * c.DAY

    def test_construction_hours_only(self):
        """Test construction with hours only."""
        duration = Duration(hours=12)
        assert duration.days == 0
        assert duration.hours == 12
        assert duration.minutes == 0
        assert duration.seconds == 0.0
        assert duration.total_seconds() == 12 * c.HOUR

    def test_construction_minutes_only(self):
        """Test construction with minutes only."""
        duration = Duration(minutes=30)
        assert duration.days == 0
        assert duration.hours == 0
        assert duration.minutes == 30
        assert duration.seconds == 0.0
        assert duration.total_seconds() == 30 * c.MINUTE

    def test_construction_seconds_only(self):
        """Test construction with seconds only."""
        duration = Duration(seconds=45)
        assert duration.days == 0
        assert duration.hours == 0
        assert duration.minutes == 0
        assert duration.seconds == 45.0
        assert duration.total_seconds() == 45.0

    def test_construction_all_components(self):
        """Test construction with all components."""
        duration = Duration(days=2, hours=3, minutes=15, seconds=30.5)
        assert duration.days == 2
        assert duration.hours == 3
        assert duration.minutes == 15
        assert duration.seconds == 30.5
        expected = 2 * c.DAY + 3 * c.HOUR + 15 * c.MINUTE + 30.5
        assert abs(duration.total_seconds() - expected) < 1e-10

    def test_construction_with_fractional_seconds(self):
        """Test construction with fractional seconds."""
        duration = Duration(seconds=12.345)
        assert abs(duration.seconds - 12.345) < 1e-3

    def test_construction_from_string(self):
        """Test construction from string format."""
        duration = Duration("5T12:30:45.500")
        assert duration.days == 5
        assert duration.hours == 12
        assert duration.minutes == 30
        assert abs(duration.seconds - 45.5) < 1e-3

    def test_construction_from_string_zero_days(self):
        """Test construction from string with zero days."""
        duration = Duration("0T01:02:03.000")
        assert duration.days == 0
        assert duration.hours == 1
        assert duration.minutes == 2
        assert duration.seconds == 3.0

    def test_construction_invalid_string_format(self):
        """Test that invalid string format raises ValueError."""
        with pytest.raises(ValueError):
            Duration("invalid_format")

    def test_construction_invalid_type_days(self):
        """Test that invalid days type raises TypeError."""
        with pytest.raises(TypeError):
            Duration(days=3.5, hours=0, minutes=0, seconds=0)

    def test_construction_invalid_type_hours(self):
        """Test that invalid hours type raises TypeError."""
        with pytest.raises(TypeError):
            Duration(days=0, hours=2.5, minutes=0, seconds=0)

    def test_construction_invalid_type_minutes(self):
        """Test that invalid minutes type raises TypeError."""
        with pytest.raises(TypeError):
            Duration(days=0, hours=0, minutes=1.5, seconds=0)

    def test_construction_seconds_accepts_float(self):
        """Test that seconds accepts float values."""
        duration = Duration(seconds=30.5)
        assert abs(duration.seconds - 30.5) < 1e-3


class TestDurationConversions:
    """Tests for Duration time conversion methods."""

    def test_total_days(self):
        """Test total_days conversion."""
        duration = Duration(days=2, hours=12)
        assert abs(duration.total_days() - 2.5) < 1e-10

    def test_total_hours(self):
        """Test total_hours conversion."""
        duration = Duration(hours=3, minutes=30)
        assert abs(duration.total_hours() - 3.5) < 1e-10

    def test_total_minutes(self):
        """Test total_minutes conversion."""
        duration = Duration(minutes=45, seconds=30)
        assert abs(duration.total_minutes() - 45.5) < 1e-10

    def test_total_seconds(self):
        """Test total_seconds conversion."""
        duration = Duration(days=1, hours=2, minutes=3, seconds=4.5)
        expected = c.DAY + 2 * c.HOUR + 3 * c.MINUTE + 4.5
        assert abs(duration.total_seconds() - expected) < 1e-10

    def test_overflow_seconds_to_minutes(self):
        """Test that 60 seconds overflows to minutes."""
        duration = Duration(seconds=90)
        assert duration.minutes == 1
        assert duration.seconds == 30.0

    def test_overflow_minutes_to_hours(self):
        """Test that 60 minutes overflows to hours."""
        duration = Duration(minutes=90)
        assert duration.hours == 1
        assert duration.minutes == 30

    def test_overflow_hours_to_days(self):
        """Test that 24 hours overflows to days."""
        duration = Duration(hours=30)
        assert duration.days == 1
        assert duration.hours == 6


class TestDurationProperties:
    """Tests for Duration properties."""

    def test_is_negative_false(self):
        """Test is_negative is False for positive duration."""
        duration = Duration(days=1, hours=2)
        assert duration.is_negative == False

    def test_is_negative_true(self):
        """Test is_negative is True for negative duration."""
        duration = Duration(seconds=-100)
        assert duration.is_negative == True

    def test_is_negative_zero(self):
        """Test is_negative is False for zero duration."""
        duration = Duration()
        assert duration.is_negative == False

    def test_negative_duration_components(self):
        """Test that negative duration has positive component values."""
        duration = Duration(seconds=-90)
        # Components are stored as absolute values
        assert duration.minutes == 1
        assert duration.seconds == 30.0
        assert duration.is_negative == True


class TestDurationFormatting:
    """Tests for Duration string formatting."""

    def test_str_representation(self):
        """Test string representation."""
        duration = Duration(days=2, hours=3, minutes=15, seconds=30)
        result = str(duration)
        assert "2d" in result
        assert "3h" in result
        assert "15m" in result
        assert "30.000s" in result

    def test_str_zero_duration(self):
        """Test string representation of zero duration."""
        duration = Duration()
        result = str(duration)
        assert "0d 0h 0m 00.000s" == result

    def test_str_negative_duration(self):
        """Test string representation of negative duration."""
        duration = Duration(seconds=-100)
        result = str(duration)
        assert result.startswith("-")

    def test_repr_representation(self):
        """Test repr representation."""
        duration = Duration(days=1, hours=2, minutes=3, seconds=4.5)
        result = repr(duration)
        assert "Duration(" in result
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_repr_negative_duration(self):
        """Test repr representation of negative duration."""
        duration = Duration(seconds=-100)
        result = repr(duration)
        assert "Duration(-" in result


class TestDurationOperators:
    """Tests for Duration operators."""

    def test_equality(self):
        """Test equality operator."""
        d1 = Duration(hours=2)
        d2 = Duration(minutes=120)
        assert d1 == d2

    def test_inequality(self):
        """Test inequality operator."""
        d1 = Duration(hours=2)
        d2 = Duration(hours=3)
        assert d1 != d2

    def test_less_than(self):
        """Test less than operator."""
        d1 = Duration(hours=1)
        d2 = Duration(hours=2)
        assert d1 < d2

    def test_less_than_or_equal(self):
        """Test less than or equal operator."""
        d1 = Duration(hours=2)
        d2 = Duration(minutes=120)
        d3 = Duration(hours=3)
        assert d1 <= d2
        assert d1 <= d3

    def test_greater_than(self):
        """Test greater than operator."""
        d1 = Duration(hours=3)
        d2 = Duration(hours=2)
        assert d1 > d2

    def test_greater_than_or_equal(self):
        """Test greater than or equal operator."""
        d1 = Duration(hours=2)
        d2 = Duration(minutes=120)
        d3 = Duration(hours=1)
        assert d1 >= d2
        assert d1 >= d3

    def test_add_duration(self):
        """Test addition of two durations."""
        d1 = Duration(hours=2)
        d2 = Duration(minutes=30)
        d3 = d1 + d2
        assert abs(d3.total_hours() - 2.5) < 1e-10

    def test_iadd_duration(self):
        """Test in-place addition."""
        d1 = Duration(hours=2)
        d2 = Duration(minutes=30)
        d1 += d2
        assert abs(d1.total_hours() - 2.5) < 1e-10

    def test_subtract_duration(self):
        """Test subtraction of durations."""
        d1 = Duration(hours=3)
        d2 = Duration(minutes=30)
        d3 = d1 - d2
        assert abs(d3.total_hours() - 2.5) < 1e-10

    def test_isub_duration(self):
        """Test in-place subtraction."""
        d1 = Duration(hours=3)
        d2 = Duration(minutes=30)
        d1 -= d2
        assert abs(d1.total_hours() - 2.5) < 1e-10

    def test_multiply_by_scalar(self):
        """Test multiplication by scalar."""
        d1 = Duration(hours=2)
        d2 = d1 * 3
        assert d2.total_hours() == 6.0

    def test_multiply_by_float(self):
        """Test multiplication by float."""
        d1 = Duration(hours=2)
        d2 = d1 * 2.5
        assert d2.total_hours() == 5.0

    def test_rmul_by_scalar(self):
        """Test reverse multiplication."""
        d1 = Duration(hours=2)
        d2 = 3 * d1
        assert d2.total_hours() == 6.0

    def test_imul_by_scalar(self):
        """Test in-place multiplication."""
        d1 = Duration(hours=2)
        d1 *= 3
        assert d1.total_hours() == 6.0

    def test_multiply_duration_by_duration(self):
        """Test multiplication of duration by duration."""
        d1 = Duration(seconds=2)
        d2 = Duration(seconds=3)
        d3 = d1 * d2
        assert d3.total_seconds() == 6.0

    def test_divide_by_scalar(self):
        """Test division by scalar."""
        d1 = Duration(hours=6)
        d2 = d1 / 2
        assert d2.total_hours() == 3.0

    def test_divide_by_float(self):
        """Test division by float."""
        d1 = Duration(hours=5)
        d2 = d1 / 2.5
        assert d2.total_hours() == 2.0

    def test_divide_duration_by_duration(self):
        """Test division of duration by duration returns scalar."""
        d1 = Duration(hours=6)
        d2 = Duration(hours=2)
        result = d1 / d2
        assert result == 3.0

    def test_rtruediv_duration(self):
        """Test reverse division with duration."""
        d1 = Duration(seconds=2)
        d2 = Duration(seconds=6)
        # d2.__rtruediv__(d1) computes d1 / d2
        result = d2.__rtruediv__(d1)
        assert abs(result - (2.0 / 6.0)) < 1e-10

    def test_rtruediv_scalar(self):
        """Test reverse division with scalar."""
        d1 = Duration(seconds=2)
        result = 10 / d1
        assert result.total_seconds() == 5.0

    def test_add_invalid_type(self):
        """Test that adding invalid type raises NotImplementedError."""
        d1 = Duration(hours=1)
        with pytest.raises(NotImplementedError):
            _ = d1 + 5

    def test_subtract_invalid_type(self):
        """Test that subtracting invalid type raises NotImplementedError."""
        d1 = Duration(hours=1)
        with pytest.raises(NotImplementedError):
            _ = d1 - 5

    def test_comparison_invalid_type(self):
        """Test that comparing with invalid type raises NotImplementedError."""
        d1 = Duration(hours=1)
        with pytest.raises(NotImplementedError):
            _ = d1 == 5

    def test_multiply_invalid_type(self):
        """Test that multiplying by invalid type raises NotImplementedError."""
        d1 = Duration(hours=1)
        with pytest.raises(NotImplementedError):
            _ = d1 * "invalid"

    def test_divide_invalid_type(self):
        """Test that dividing by invalid type raises NotImplementedError."""
        d1 = Duration(hours=1)
        with pytest.raises(NotImplementedError):
            _ = d1 / "invalid"


class TestDurationUtilityMethods:
    """Tests for Duration utility methods."""

    def test_copy(self):
        """Test that copy creates independent Duration."""
        d1 = Duration(days=1, hours=2, minutes=3, seconds=4.5)
        d2 = d1.copy()
        assert d1 == d2
        assert d1 is not d2
        d2 += Duration(hours=1)
        assert d1 != d2

    def test_to_dict(self):
        """Test to_dict serialization."""
        duration = Duration(days=1, hours=2, minutes=3, seconds=4.5)
        result = duration.to_dict()
        assert result["type"] == "Duration"
        assert "total_seconds" in result
        assert isinstance(result["total_seconds"], float)

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {"type": "Duration", "total_seconds": 3661.5}
        duration = Duration.from_dict(data)
        assert abs(duration.total_seconds() - 3661.5) < 1e-10

    def test_from_dict_invalid_type(self):
        """Test that from_dict with invalid type raises ValueError."""
        data = {"type": "NotDuration", "total_seconds": 3600.0}
        with pytest.raises(ValueError):
            Duration.from_dict(data)

    def test_dict_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip preserves value."""
        d1 = Duration(days=2, hours=3, minutes=15, seconds=30.5)
        data = d1.to_dict()
        d2 = Duration.from_dict(data)
        assert abs(d1.total_seconds() - d2.total_seconds()) < 1e-10


class TestDurationEdgeCases:
    """Tests for Duration edge cases."""

    def test_zero_duration(self):
        """Test zero duration."""
        duration = Duration()
        assert duration.total_seconds() == 0.0
        assert duration.days == 0
        assert duration.hours == 0
        assert duration.minutes == 0
        assert duration.seconds == 0.0
        assert duration.is_negative == False

    def test_very_large_duration(self):
        """Test very large duration."""
        duration = Duration(days=1000)
        assert duration.days == 1000
        assert duration.total_seconds() == 1000 * c.DAY

    def test_negative_seconds(self):
        """Test negative duration from negative seconds."""
        duration = Duration(seconds=-3661.5)
        assert duration.is_negative == True
        assert duration.hours == 1
        assert duration.minutes == 1
        assert abs(duration.seconds - 1.5) < 1e-3

    def test_negative_large_value(self):
        """Test large negative duration."""
        duration = Duration(seconds=-c.DAY * 2)
        assert duration.is_negative == True
        assert duration.days == 2

    def test_high_precision_seconds(self):
        """Test high precision fractional seconds."""
        duration = Duration(seconds=1.23456789)
        # Duration rounds to 3 decimals
        assert abs(duration.seconds - 1.235) < 1e-3

    def test_seconds_overflow_boundary(self):
        """Test seconds at overflow boundary (59.999)."""
        duration = Duration(seconds=59.999)
        # Should overflow to next minute
        assert duration.minutes == 1
        assert duration.seconds == 0.0

    def test_minutes_overflow_boundary(self):
        """Test minutes at overflow boundary."""
        duration = Duration(minutes=60)
        assert duration.hours == 1
        assert duration.minutes == 0

    def test_hours_overflow_boundary(self):
        """Test hours at overflow boundary."""
        duration = Duration(hours=24)
        assert duration.days == 1
        assert duration.hours == 0

    def test_multiple_overflows(self):
        """Test multiple simultaneous overflows."""
        duration = Duration(hours=25, minutes=65, seconds=75)
        assert duration.days == 1
        assert duration.hours == 2
        assert duration.minutes == 6
        assert duration.seconds == 15.0

    def test_subtraction_resulting_negative(self):
        """Test subtraction that results in negative duration."""
        d1 = Duration(hours=1)
        d2 = Duration(hours=2)
        d3 = d1 - d2
        assert d3.is_negative == True
        assert d3.total_seconds() == -c.HOUR
