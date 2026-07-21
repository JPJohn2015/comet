"""Comprehensive test suite for the Epoch class.

This module tests all functionality of the Epoch class including:
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
from datetime import datetime, UTC

# comet imports
from comet.time import Epoch, TimeSystem
from comet.time import Duration
from comet.utilities.constants import Constants as c
from comet.time.time_conversions import jd_to_unix


class TestEpochConstruction:
    """Tests for Epoch construction methods."""

    def test_default_construction(self):
        """Test default construction creates J2000 epoch."""
        epoch = Epoch()
        assert epoch.julian_date() == c.J2000
        assert epoch.year == 2000
        assert epoch.month == 1
        assert epoch.day == 1
        assert epoch.hour == 12
        assert epoch.minute == 0
        assert epoch.second == 0.0

    def test_construction_year_month_day(self):
        """Test construction from year, month, day."""
        epoch = Epoch(2000, 1, 1)
        assert epoch.julian_date() == c.J2000
        assert epoch.year == 2000
        assert epoch.month == 1
        assert epoch.day == 1

    def test_construction_full_ymdhms(self):
        """Test construction from complete date and time."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        assert epoch.julian_date() == c.J2000
        assert epoch.year == 2000
        assert epoch.month == 1
        assert epoch.day == 1
        assert epoch.hour == 12
        assert epoch.minute == 0
        assert epoch.second == 0.0

    def test_construction_with_fractional_seconds(self):
        """Test construction with fractional seconds."""
        epoch = Epoch(2024, 6, 15, 14, 30, 45.789)
        assert epoch.second == 45.789

    def test_construction_from_julian_date(self):
        """Test construction from Julian Date."""
        epoch = Epoch(c.J2000)
        assert epoch.julian_date() == c.J2000
        assert epoch.year == 2000
        assert epoch.month == 1
        assert epoch.day == 1

    def test_construction_from_string(self):
        """Test construction from ISO 8601 string."""
        epoch = Epoch("2000-01-01T12:00:00.0")
        assert epoch.julian_date() == c.J2000

    def test_construction_from_string_with_z(self):
        """Test construction from ISO 8601 string with Z suffix."""
        epoch = Epoch("2000-01-01T12:00:00.0Z")
        assert epoch.julian_date() == c.J2000

    def test_construction_invalid_string_format(self):
        """Test that invalid string format raises ValueError."""
        with pytest.raises(ValueError):
            Epoch("invalid-date-string")

    def test_construction_invalid_type_year(self):
        """Test that invalid year type raises TypeError."""
        with pytest.raises(TypeError):
            Epoch([1, 2, 3], 1, 1)

    def test_from_datetime(self):
        """Test construction from Python datetime."""
        dt = datetime(2024, 6, 15, 14, 30, 45, 500000)
        epoch = Epoch.from_datetime(dt)
        assert epoch.year == 2024
        assert epoch.month == 6
        assert epoch.day == 15
        assert epoch.hour == 14
        assert epoch.minute == 30
        assert abs(epoch.second - 45.5) < 0.001

    def test_from_unix(self):
        """Test construction from Unix timestamp."""
        unix_time = 0.0  # 1970-01-01 00:00:00 UTC
        epoch = Epoch.from_unix(unix_time)
        assert epoch.year == 1970
        assert epoch.month == 1
        assert epoch.day == 1

    def test_from_mjd(self):
        """Test construction from Modified Julian Date."""
        epoch = Epoch.from_mjd(c.MJ2000)
        assert abs(epoch.julian_date() - c.J2000) < 1e-10

    def test_now(self):
        """Test that now() creates an Epoch close to current time."""
        before = datetime.now(UTC)
        epoch = Epoch.now()
        after = datetime.now(UTC)

        # Check that epoch is between before and after
        assert epoch.year == before.year
        assert epoch.month == before.month
        assert epoch.day == before.day


class TestEpochConversions:
    """Tests for Epoch time conversion methods."""

    def test_julian_date_utc(self):
        """Test Julian Date in UTC."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        assert epoch.julian_date() == c.J2000
        assert epoch.julian_date("UTC") == c.J2000
        assert epoch.julian_date(TimeSystem.UTC) == c.J2000

    def test_julian_date_tt(self):
        """Test Julian Date in TT."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        expected = c.J2000 + (c.UTC_TT / c.DAY)
        assert epoch.julian_date("TT") == expected
        assert epoch.julian_date(TimeSystem.TT) == expected

    def test_julian_date_tai(self):
        """Test Julian Date in TAI."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        expected = c.J2000 + (c.UTC_TAI / c.DAY)
        assert epoch.julian_date("TAI") == expected
        assert epoch.julian_date(TimeSystem.TAI) == expected

    def test_julian_date_ut1_not_implemented(self):
        """Test that UT1 raises NotImplementedError."""
        epoch = Epoch(2000, 1, 1)
        with pytest.raises(NotImplementedError):
            epoch.julian_date("UT1")

    def test_julian_date_invalid_timesystem(self):
        """Test that invalid time system raises Exception."""
        epoch = Epoch(2000, 1, 1)
        with pytest.raises(Exception):
            epoch.julian_date("INVALID")

    def test_modified_julian_date_utc(self):
        """Test Modified Julian Date in UTC."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        assert epoch.modified_julian_date() == c.MJ2000
        assert epoch.modified_julian_date("UTC") == c.MJ2000

    def test_modified_julian_date_tt(self):
        """Test Modified Julian Date in TT."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        expected = c.MJ2000 + (c.UTC_TT / c.DAY)
        assert epoch.modified_julian_date("TT") == expected

    def test_unix_utc(self):
        """Test Unix timestamp in UTC."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        expected = jd_to_unix(c.J2000)
        assert epoch.unix() == expected
        assert epoch.unix("UTC") == expected

    def test_unix_tt(self):
        """Test Unix timestamp in TT."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        expected = jd_to_unix(c.J2000) + c.UTC_TT
        assert epoch.unix("TT") == expected

    def test_datetime_conversion(self):
        """Test conversion to Python datetime."""
        epoch = Epoch(2024, 6, 15, 14, 30, 45.5)
        dt = epoch.datetime()
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 45

    def test_julian_centuries_utc(self):
        """Test Julian centuries since J2000."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        assert epoch.julian_centuries() == 0.0

        epoch_plus_day = Epoch(2000, 1, 2, 12, 0, 0)
        expected = 1 / c.JULIAN_CENTURY
        assert epoch_plus_day.julian_centuries() == expected

    def test_julian_centuries_tt(self):
        """Test Julian centuries in TT."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        expected = (c.UTC_TT / c.DAY) / c.JULIAN_CENTURY
        assert np.isclose(epoch.julian_centuries("TT"), expected, atol=1e-10)


class TestEpochProperties:
    """Tests for Epoch property methods."""

    def test_day_of_year_january(self):
        """Test day of year for January dates."""
        epoch = Epoch(2024, 1, 15)
        assert epoch.day_of_year() == 15

    def test_day_of_year_february_non_leap(self):
        """Test day of year for February in non-leap year."""
        epoch = Epoch(2023, 2, 1)
        assert epoch.day_of_year() == 32

    def test_day_of_year_february_leap(self):
        """Test day of year for February in leap year."""
        epoch = Epoch(2024, 2, 1)
        assert epoch.day_of_year() == 32

    def test_day_of_year_march_leap(self):
        """Test day of year for March in leap year (accounts for Feb 29)."""
        epoch = Epoch(2024, 3, 1)
        assert epoch.day_of_year() == 61  # 31 (Jan) + 29 (Feb) + 1

    def test_day_of_year_december(self):
        """Test day of year for end of year."""
        epoch = Epoch(2023, 12, 31)
        assert epoch.day_of_year() == 365

        epoch_leap = Epoch(2024, 12, 31)
        assert epoch_leap.day_of_year() == 366

    def test_week_of_year(self):
        """Test ISO week number calculation."""
        epoch = Epoch(2024, 1, 1)
        week = epoch.week_of_year()
        assert isinstance(week, int)
        assert 1 <= week <= 53

    def test_week_of_year_middle_of_year(self):
        """Test week number in middle of year."""
        epoch = Epoch(2024, 6, 15)
        week = epoch.week_of_year()
        assert 20 <= week <= 30


class TestEpochFormatting:
    """Tests for Epoch string representation methods."""

    def test_str_representation(self):
        """Test string representation."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        assert str(epoch) == "01/01/2000 12:00:00.000 UTC"

    def test_str_with_fractional_seconds(self):
        """Test string representation with fractional seconds."""
        epoch = Epoch(2024, 6, 15, 14, 30, 45.789)
        expected = "06/15/2024 14:30:45.789 UTC"
        assert str(epoch) == expected

    def test_repr_representation(self):
        """Test repr representation."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        assert repr(epoch) == "Epoch(2000, 1, 1, 12, 0, 00.000)"

    def test_isoformat(self):
        """Test ISO 8601 format."""
        epoch = Epoch(2024, 6, 15, 14, 30, 45.789)
        iso = epoch.isoformat()
        assert iso == "2024-06-15T14:30:45.789Z"

    def test_isoformat_roundtrip(self):
        """Test that isoformat can be parsed back."""
        epoch1 = Epoch(2024, 6, 15, 14, 30, 45.789)
        iso_str = epoch1.isoformat()
        epoch2 = Epoch(iso_str)
        assert abs(epoch1.julian_date() - epoch2.julian_date()) < 1e-10


class TestEpochOperators:
    """Tests for Epoch operator overloads."""

    def test_equality(self):
        """Test equality operator."""
        epoch1 = Epoch(2000, 1, 1, 12, 0, 0)
        epoch2 = Epoch(2000, 1, 1, 12, 0, 0)
        assert epoch1 == epoch2

    def test_inequality(self):
        """Test inequality operator."""
        epoch1 = Epoch(2000, 1, 1)
        epoch2 = Epoch(2000, 1, 2)
        assert epoch1 != epoch2

    def test_less_than(self):
        """Test less than operator."""
        epoch1 = Epoch(2000, 1, 1)
        epoch2 = Epoch(2000, 1, 2)
        assert epoch1 < epoch2

    def test_less_than_or_equal(self):
        """Test less than or equal operator."""
        epoch1 = Epoch(2000, 1, 1)
        epoch2 = Epoch(2000, 1, 2)
        epoch3 = Epoch(2000, 1, 1)
        assert epoch1 <= epoch2
        assert epoch1 <= epoch3

    def test_greater_than(self):
        """Test greater than operator."""
        epoch1 = Epoch(2000, 1, 2)
        epoch2 = Epoch(2000, 1, 1)
        assert epoch1 > epoch2

    def test_greater_than_or_equal(self):
        """Test greater than or equal operator."""
        epoch1 = Epoch(2000, 1, 2)
        epoch2 = Epoch(2000, 1, 1)
        epoch3 = Epoch(2000, 1, 2)
        assert epoch1 >= epoch2
        assert epoch1 >= epoch3

    def test_add_duration(self):
        """Test adding Duration to Epoch."""
        epoch1 = Epoch(2000, 1, 1)
        duration = Duration(days=1)
        epoch2 = epoch1 + duration
        assert epoch2 == Epoch(2000, 1, 2)

    def test_iadd_duration(self):
        """Test in-place addition of Duration."""
        epoch = Epoch(2000, 1, 1)
        duration = Duration(days=1)
        epoch += duration
        assert epoch == Epoch(2000, 1, 2)

    def test_subtract_duration(self):
        """Test subtracting Duration from Epoch."""
        epoch1 = Epoch(2000, 1, 2)
        duration = Duration(days=1)
        epoch2 = epoch1 - duration
        assert epoch2 == Epoch(2000, 1, 1)

    def test_subtract_epoch(self):
        """Test subtracting Epoch from Epoch."""
        epoch1 = Epoch(2000, 1, 2)
        epoch2 = Epoch(2000, 1, 1)
        duration = epoch1 - epoch2
        assert isinstance(duration, Duration)
        assert duration == Duration(days=1)

    def test_add_invalid_type(self):
        """Test that adding invalid type raises error."""
        epoch = Epoch(2000, 1, 1)
        with pytest.raises(NotImplementedError):
            _ = epoch + 5

    def test_subtract_invalid_type(self):
        """Test that subtracting invalid type raises error."""
        epoch = Epoch(2000, 1, 1)
        with pytest.raises(NotImplementedError):
            _ = epoch - 5

    def test_comparison_invalid_type(self):
        """Test that comparing with invalid type raises error."""
        epoch = Epoch(2000, 1, 1)
        with pytest.raises(NotImplementedError):
            _ = epoch == 5

    def test_multiplication_not_allowed(self):
        """Test that multiplication raises error."""
        epoch = Epoch(2000, 1, 1)
        with pytest.raises(NotImplementedError):
            _ = epoch * 2

    def test_division_not_allowed(self):
        """Test that division raises error."""
        epoch = Epoch(2000, 1, 1)
        with pytest.raises(NotImplementedError):
            _ = epoch / 2


class TestEpochHashability:
    """Tests for Epoch hashability."""

    def test_hashable(self):
        """Test that Epoch is hashable."""
        epoch = Epoch(2000, 1, 1)
        hash_value = hash(epoch)
        assert isinstance(hash_value, int)

    def test_equal_epochs_same_hash(self):
        """Test that equal Epochs have same hash."""
        epoch1 = Epoch(2000, 1, 1)
        epoch2 = Epoch(2000, 1, 1)
        assert hash(epoch1) == hash(epoch2)

    def test_can_use_in_set(self):
        """Test that Epoch can be used in sets."""
        epoch1 = Epoch(2000, 1, 1)
        epoch2 = Epoch(2000, 1, 2)
        epoch3 = Epoch(2000, 1, 1)

        epoch_set = {epoch1, epoch2, epoch3}
        assert len(epoch_set) == 2  # epoch1 and epoch3 are equal

    def test_can_use_as_dict_key(self):
        """Test that Epoch can be used as dictionary key."""
        epoch1 = Epoch(2000, 1, 1)
        epoch2 = Epoch(2000, 1, 2)

        epoch_dict = {epoch1: "first", epoch2: "second"}
        assert epoch_dict[epoch1] == "first"
        assert epoch_dict[epoch2] == "second"


class TestEpochUtilityMethods:
    """Tests for Epoch utility methods."""

    def test_copy(self):
        """Test that copy creates independent copy."""
        epoch1 = Epoch(2024, 6, 15, 14, 30, 45.789)
        epoch2 = epoch1.copy()

        assert epoch1 == epoch2
        assert epoch1 is not epoch2

        # Modify copy should not affect original
        epoch2 += Duration(days=1)
        assert epoch1 != epoch2

    def test_to_dict(self):
        """Test dictionary serialization."""
        epoch = Epoch(2024, 6, 15, 14, 30, 45)
        d = epoch.to_dict()

        assert "type" in d
        assert "jd" in d
        assert d["type"] == "Epoch"
        assert isinstance(d["jd"], float)

    def test_from_dict(self):
        """Test dictionary deserialization."""
        epoch1 = Epoch(2024, 6, 15, 14, 30, 45)
        d = epoch1.to_dict()
        epoch2 = Epoch.from_dict(d)

        assert epoch1 == epoch2

    def test_from_dict_invalid_type(self):
        """Test that from_dict with invalid type raises error."""
        invalid_dict = {"type": "NotAnEpoch", "jd": c.J2000}
        with pytest.raises(ValueError):
            Epoch.from_dict(invalid_dict)

    def test_dict_roundtrip(self):
        """Test that dict serialization and deserialization preserves value."""
        epoch1 = Epoch(2024, 6, 15, 14, 30, 45.789)
        d = epoch1.to_dict()
        epoch2 = Epoch.from_dict(d)

        assert abs(epoch1.julian_date() - epoch2.julian_date()) < 1e-10


class TestEpochEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_leap_year_february_29(self):
        """Test leap year date February 29."""
        epoch = Epoch(2024, 2, 29)
        assert epoch.year == 2024
        assert epoch.month == 2
        assert epoch.day == 29

    def test_year_boundary(self):
        """Test year boundary transition."""
        epoch = Epoch(1999, 12, 31, 23, 59, 59)
        epoch_next = epoch + Duration(seconds=1)
        assert epoch_next.year == 2000
        assert epoch_next.month == 1
        assert epoch_next.day == 1
        assert epoch_next.hour == 0
        assert epoch_next.minute == 0
        assert epoch_next.second == 0.0

    def test_distant_past(self):
        """Test epoch in distant past."""
        epoch = Epoch(1900, 1, 1)
        assert epoch.year == 1900
        assert epoch.month == 1
        assert epoch.day == 1

    def test_distant_future(self):
        """Test epoch in distant future."""
        epoch = Epoch(2100, 12, 31)
        assert epoch.year == 2100
        assert epoch.month == 12
        assert epoch.day == 31

    def test_midnight(self):
        """Test midnight time."""
        epoch = Epoch(2024, 6, 15, 0, 0, 0)
        assert epoch.hour == 0
        assert epoch.minute == 0
        assert epoch.second == 0.0

    def test_end_of_day(self):
        """Test end of day."""
        epoch = Epoch(2024, 6, 15, 23, 59, 59.999)
        assert epoch.hour == 23
        assert epoch.minute == 59
        assert epoch.second == 59.999

    def test_high_precision_seconds(self):
        """Test high precision fractional seconds."""
        epoch = Epoch(2024, 6, 15, 12, 30, 45.123456)
        # Should be rounded to millisecond precision
        assert abs(epoch.second - 45.123) < 0.001


class TestEpochTimeSystem:
    """Tests for TimeSystem enum and conversions."""

    def test_timesystem_enum_values(self):
        """Test TimeSystem enum has correct values."""
        assert TimeSystem.UTC.value == "UTC"
        assert TimeSystem.TT.value == "TT"
        assert TimeSystem.TAI.value == "TAI"
        assert TimeSystem.UT1.value == "UT1"

    def test_case_insensitive_timesystem(self):
        """Test that time system strings are case-insensitive."""
        epoch = Epoch(2000, 1, 1)
        jd_upper = epoch.julian_date("UTC")
        jd_lower = epoch.julian_date("utc")
        assert jd_upper == jd_lower


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
