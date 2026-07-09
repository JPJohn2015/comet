"""Comprehensive test suite for time_conversions module.

This module tests all time conversion functions including:
- Julian Date conversions
- Modified Julian Date conversions
- Unix timestamp conversions
- YMDHMS conversions
- Day of year conversions
- Sidereal time conversions
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.time.time_conversions import (
    ymdhms_to_jd,
    jd_to_ymdhms,
    date_round,
    jd_to_mjd,
    mjd_to_jd,
    unix_to_jd,
    jd_to_unix,
    unix_to_mjd,
    mjd_to_unix,
    ymdhms_to_mjd,
    mjd_to_ymdhms,
    ymdhms_to_unix,
    unix_to_ymdhms,
    ymdhms_to_dayofyear,
    dayofyear_to_monthday,
    jd_to_gmst,
    jd_to_lst,
)
from comet.utilities.constants import Constants as c


class TestJulianDateConversions:
    """Tests for Julian Date conversions."""

    def test_ymdhms_to_jd_j2000(self):
        """Test conversion to J2000 epoch."""
        jd = ymdhms_to_jd(2000, 1, 1, 12, 0, 0)
        assert abs(jd - c.J2000) < 1e-10

    def test_ymdhms_to_jd_with_time(self):
        """Test conversion with specific time."""
        jd = ymdhms_to_jd(2000, 1, 1, 0, 0, 0)
        assert abs(jd - (c.J2000 - 0.5)) < 1e-10

    def test_ymdhms_to_jd_february(self):
        """Test conversion for February date."""
        jd = ymdhms_to_jd(2000, 2, 1, 12, 0, 0)
        expected = c.J2000 + 31.0
        assert abs(jd - expected) < 1e-10

    def test_ymdhms_to_jd_leap_year(self):
        """Test conversion for leap year date."""
        jd = ymdhms_to_jd(2000, 2, 29, 12, 0, 0)
        expected = c.J2000 + 59.0
        assert abs(jd - expected) < 1e-10

    def test_ymdhms_to_jd_with_fractional_seconds(self):
        """Test conversion with fractional seconds."""
        jd1 = ymdhms_to_jd(2000, 1, 1, 12, 0, 0.0)
        jd2 = ymdhms_to_jd(2000, 1, 1, 12, 0, 1.0)
        diff = (jd2 - jd1) * c.DAY
        assert abs(diff - 1.0) < 1e-5

    def test_jd_to_ymdhms_j2000(self):
        """Test conversion from J2000 epoch."""
        y, mo, d, h, m, s = jd_to_ymdhms(c.J2000)
        assert y == 2000
        assert mo == 1
        assert d == 1
        assert h == 12
        assert m == 0
        assert abs(s) < 1e-6

    def test_jd_to_ymdhms_midnight(self):
        """Test conversion at midnight."""
        jd = c.J2000 - 0.5
        y, mo, d, h, m, s = jd_to_ymdhms(jd)
        assert y == 2000
        assert mo == 1
        assert d == 1
        assert h == 0
        assert m == 0
        assert abs(s) < 1e-6

    def test_jd_to_ymdhms_with_fractional_day(self):
        """Test conversion with fractional day."""
        jd = c.J2000 + 0.25
        y, mo, d, h, m, s = jd_to_ymdhms(jd)
        assert y == 2000
        assert mo == 1
        assert d == 1
        assert h == 18

    def test_ymdhms_jd_roundtrip(self):
        """Test round-trip conversion YMDHMS -> JD -> YMDHMS."""
        test_dates = [
            (2000, 1, 1, 12, 0, 0),
            (2024, 6, 15, 14, 30, 45),
            (1900, 1, 1, 0, 0, 0),
            (2100, 12, 31, 23, 59, 59),
            (2000, 2, 29, 12, 0, 0),  # Leap year
        ]
        for y1, mo1, d1, h1, m1, s1 in test_dates:
            jd = ymdhms_to_jd(y1, mo1, d1, h1, m1, s1)
            y2, mo2, d2, h2, m2, s2 = jd_to_ymdhms(jd)
            assert y1 == y2
            assert mo1 == mo2
            assert d1 == d2
            assert h1 == h2
            assert m1 == m2
            assert abs(s1 - s2) < 1e-3


class TestModifiedJulianDateConversions:
    """Tests for Modified Julian Date conversions."""

    def test_jd_to_mjd_j2000(self):
        """Test JD to MJD for J2000."""
        mjd = jd_to_mjd(c.J2000)
        assert abs(mjd - c.MJ2000) < 1e-10

    def test_mjd_to_jd_j2000(self):
        """Test MJD to JD for J2000."""
        jd = mjd_to_jd(c.MJ2000)
        assert abs(jd - c.J2000) < 1e-10

    def test_jd_mjd_roundtrip(self):
        """Test round-trip JD -> MJD -> JD."""
        test_jds = [c.J2000, c.J2000 + 100, c.J2000 - 100, 2451545.0]
        for jd1 in test_jds:
            mjd = jd_to_mjd(jd1)
            jd2 = mjd_to_jd(mjd)
            assert abs(jd1 - jd2) < 1e-10

    def test_jd_to_mjd_array(self):
        """Test JD to MJD with array input."""
        jds = np.array([c.J2000, c.J2000 + 1, c.J2000 + 2])
        mjds = jd_to_mjd(jds)
        expected = np.array([c.MJ2000, c.MJ2000 + 1, c.MJ2000 + 2])
        assert np.allclose(mjds, expected)

    def test_ymdhms_to_mjd(self):
        """Test direct YMDHMS to MJD conversion."""
        mjd = ymdhms_to_mjd(2000, 1, 1, 12, 0, 0)
        assert abs(mjd - c.MJ2000) < 1e-10

    def test_mjd_to_ymdhms(self):
        """Test direct MJD to YMDHMS conversion."""
        y, mo, d, h, m, s = mjd_to_ymdhms(c.MJ2000)
        assert y == 2000
        assert mo == 1
        assert d == 1
        assert h == 12
        assert m == 0
        assert abs(s) < 1e-6

    def test_ymdhms_mjd_roundtrip(self):
        """Test round-trip YMDHMS -> MJD -> YMDHMS."""
        y1, mo1, d1, h1, m1, s1 = 2024, 7, 9, 15, 30, 45.5
        mjd = ymdhms_to_mjd(y1, mo1, d1, h1, m1, s1)
        y2, mo2, d2, h2, m2, s2 = mjd_to_ymdhms(mjd)
        assert y1 == y2
        assert mo1 == mo2
        assert d1 == d2
        assert h1 == h2
        assert m1 == m2
        assert abs(s1 - s2) < 1e-3


class TestUnixTimestampConversions:
    """Tests for Unix timestamp conversions."""

    def test_unix_to_jd_epoch(self):
        """Test Unix epoch (0) to JD."""
        jd = unix_to_jd(0)
        assert abs(jd - c.UNIX0) < 1e-10

    def test_jd_to_unix_epoch(self):
        """Test JD to Unix for Unix epoch."""
        unix = jd_to_unix(c.UNIX0)
        assert abs(unix) < 1e-10

    def test_unix_jd_roundtrip(self):
        """Test round-trip Unix -> JD -> Unix."""
        test_unix = [0, 946728000, 1609459200, -86400]  # Various timestamps
        for u1 in test_unix:
            jd = unix_to_jd(u1)
            u2 = jd_to_unix(jd)
            assert abs(u1 - u2) < 1e-6

    def test_unix_to_jd_j2000(self):
        """Test Unix timestamp for J2000."""
        unix_j2000 = jd_to_unix(c.J2000)
        jd = unix_to_jd(unix_j2000)
        assert abs(jd - c.J2000) < 1e-10

    def test_unix_to_mjd(self):
        """Test Unix to MJD conversion."""
        unix = 0
        mjd = unix_to_mjd(unix)
        expected_jd = c.UNIX0
        expected_mjd = jd_to_mjd(expected_jd)
        assert abs(mjd - expected_mjd) < 1e-10

    def test_mjd_to_unix(self):
        """Test MJD to Unix conversion."""
        mjd = c.MJ2000
        unix = mjd_to_unix(mjd)
        expected_jd = mjd_to_jd(mjd)
        expected_unix = jd_to_unix(expected_jd)
        assert abs(unix - expected_unix) < 1e-6

    def test_unix_mjd_roundtrip(self):
        """Test round-trip Unix -> MJD -> Unix."""
        test_unix = [0, 946728000, 1609459200]
        for u1 in test_unix:
            mjd = unix_to_mjd(u1)
            u2 = mjd_to_unix(mjd)
            assert abs(u1 - u2) < 1e-6

    def test_ymdhms_to_unix_epoch(self):
        """Test YMDHMS to Unix for Unix epoch."""
        unix = ymdhms_to_unix(1970, 1, 1, 0, 0, 0)
        assert abs(unix) < 1e-6

    def test_ymdhms_to_unix_j2000(self):
        """Test YMDHMS to Unix for J2000."""
        unix = ymdhms_to_unix(2000, 1, 1, 12, 0, 0)
        expected = jd_to_unix(c.J2000)
        assert abs(unix - expected) < 1e-6

    def test_unix_to_ymdhms_epoch(self):
        """Test Unix to YMDHMS for Unix epoch."""
        y, mo, d, h, m, s = unix_to_ymdhms(0)
        assert y == 1970
        assert mo == 1
        assert d == 1
        assert h == 0
        assert m == 0
        assert abs(s) < 1e-6

    def test_ymdhms_unix_roundtrip(self):
        """Test round-trip YMDHMS -> Unix -> YMDHMS."""
        y1, mo1, d1, h1, m1, s1 = 2024, 7, 9, 15, 30, 45.5
        unix = ymdhms_to_unix(y1, mo1, d1, h1, m1, s1)
        y2, mo2, d2, h2, m2, s2 = unix_to_ymdhms(unix)
        assert y1 == y2
        assert mo1 == mo2
        assert d1 == d2
        assert h1 == h2
        assert m1 == m2
        assert abs(s1 - s2) < 1e-3


class TestDayOfYearConversions:
    """Tests for day of year conversions."""

    def test_ymdhms_to_dayofyear_jan1(self):
        """Test January 1st is day 1."""
        doy = ymdhms_to_dayofyear(2024, 1, 1)
        assert doy == 1

    def test_ymdhms_to_dayofyear_feb1(self):
        """Test February 1st is day 32."""
        doy = ymdhms_to_dayofyear(2024, 2, 1)
        assert doy == 32

    def test_ymdhms_to_dayofyear_dec31_nonleap(self):
        """Test December 31st is day 365 in non-leap year."""
        doy = ymdhms_to_dayofyear(2023, 12, 31)
        assert doy == 365

    def test_ymdhms_to_dayofyear_dec31_leap(self):
        """Test December 31st is day 366 in leap year."""
        doy = ymdhms_to_dayofyear(2024, 12, 31)
        assert doy == 366

    def test_ymdhms_to_dayofyear_leap_day(self):
        """Test February 29th is day 60 in leap year."""
        doy = ymdhms_to_dayofyear(2024, 2, 29)
        assert doy == 60

    def test_ymdhms_to_dayofyear_after_leap_day(self):
        """Test March 1st is day 61 in leap year."""
        doy = ymdhms_to_dayofyear(2024, 3, 1)
        assert doy == 61

    def test_ymdhms_to_dayofyear_nonleap_march(self):
        """Test March 1st is day 60 in non-leap year."""
        doy = ymdhms_to_dayofyear(2023, 3, 1)
        assert doy == 60

    def test_dayofyear_to_monthday_day1(self):
        """Test day 1 is January 1st."""
        mo, d = dayofyear_to_monthday(2024, 1)
        assert mo == 1
        assert d == 1

    def test_dayofyear_to_monthday_day32(self):
        """Test day 32 is February 1st."""
        mo, d = dayofyear_to_monthday(2024, 32)
        assert mo == 2
        assert d == 1

    def test_dayofyear_to_monthday_day365_nonleap(self):
        """Test day 365 is December 31st in non-leap year."""
        mo, d = dayofyear_to_monthday(2023, 365)
        assert mo == 12
        assert d == 31

    def test_dayofyear_to_monthday_day366_leap(self):
        """Test day 366 is December 31st in leap year."""
        mo, d = dayofyear_to_monthday(2024, 366)
        assert mo == 12
        assert d == 31

    def test_dayofyear_to_monthday_leap_day(self):
        """Test day 60 is February 29th in leap year."""
        mo, d = dayofyear_to_monthday(2024, 60)
        assert mo == 2
        assert d == 29

    def test_dayofyear_roundtrip_leap(self):
        """Test round-trip for all days in leap year."""
        year = 2024
        for month in range(1, 13):
            for day in range(1, 29):  # Safe for all months
                doy = ymdhms_to_dayofyear(year, month, day)
                mo2, d2 = dayofyear_to_monthday(year, doy)
                assert month == mo2
                assert day == d2

    def test_dayofyear_roundtrip_nonleap(self):
        """Test round-trip for all days in non-leap year."""
        year = 2023
        for month in range(1, 13):
            for day in range(1, 29):  # Safe for all months
                doy = ymdhms_to_dayofyear(year, month, day)
                mo2, d2 = dayofyear_to_monthday(year, doy)
                assert month == mo2
                assert day == d2


class TestSiderealTimeConversions:
    """Tests for sidereal time conversions."""

    def test_jd_to_gmst_j2000(self):
        """Test GMST at J2000 epoch."""
        # At J2000 (2000-01-01 12:00:00 TT), GMST should be approximately 280.46 degrees
        gmst = jd_to_gmst(c.J2000)
        # Allow reasonable tolerance for GMST calculation
        assert 0 <= gmst < 360

    def test_jd_to_gmst_24h_difference(self):
        """Test GMST advances ~361 degrees per day."""
        gmst1 = jd_to_gmst(c.J2000)
        gmst2 = jd_to_gmst(c.J2000 + 1)
        # GMST advances about 361.0 degrees per solar day (360 + ~1 from Earth orbit)
        diff = (gmst2 - gmst1) % 360
        assert 0.9 < diff < 1.1  # Should be close to 1 degree

    def test_jd_to_gmst_range(self):
        """Test GMST is always in range [0, 360)."""
        test_jds = [c.J2000, c.J2000 + 100, c.J2000 + 365.25, c.J2000 - 100]
        for jd in test_jds:
            gmst = jd_to_gmst(jd)
            assert 0 <= gmst < 360

    def test_jd_to_gmst_array(self):
        """Test GMST with array input."""
        jds = np.array([c.J2000, c.J2000 + 1, c.J2000 + 2])
        gmsts = jd_to_gmst(jds)
        assert len(gmsts) == 3
        assert np.all((gmsts >= 0) & (gmsts < 360))

    def test_jd_to_lst_zero_longitude(self):
        """Test LST equals GMST at zero longitude."""
        lst = jd_to_lst(c.J2000, 0)
        gmst = jd_to_gmst(c.J2000)
        assert abs(lst - gmst) < 1e-10

    def test_jd_to_lst_positive_longitude(self):
        """Test LST is GMST + longitude."""
        lon = 45.0  # degrees
        lst = jd_to_lst(c.J2000, lon)
        gmst = jd_to_gmst(c.J2000)
        expected = (gmst + lon) % 360
        assert abs(lst - expected) < 1e-10

    def test_jd_to_lst_negative_longitude(self):
        """Test LST with negative longitude."""
        lon = -75.0  # degrees
        lst = jd_to_lst(c.J2000, lon)
        gmst = jd_to_gmst(c.J2000)
        expected = (gmst + lon) % 360
        assert abs(lst - expected) < 1e-10

    def test_jd_to_lst_wraparound(self):
        """Test LST wraps around at 360 degrees."""
        lon = 180.0
        lst = jd_to_lst(c.J2000, lon)
        assert 0 <= lst < 360

    def test_jd_to_lst_array(self):
        """Test LST with array inputs."""
        jds = np.array([c.J2000, c.J2000 + 1])
        lons = np.array([0, 45])
        lsts = jd_to_lst(jds, lons)
        assert len(lsts) == 2
        assert np.all((lsts >= 0) & (lsts < 360))


class TestDateRound:
    """Tests for date_round utility function."""

    def test_date_round_no_overflow(self):
        """Test date_round with no overflow."""
        ymdhms = [2024, 7, 9, 15, 30, 45.5]
        result = date_round(ymdhms)
        assert result[0] == 2024
        assert result[1] == 7
        assert result[2] == 9
        assert result[3] == 15
        assert result[4] == 30
        assert result[5] == 45.5

    def test_date_round_seconds_overflow(self):
        """Test date_round with seconds >= 60."""
        ymdhms = [2024, 7, 9, 15, 30, 59.999]
        result = date_round(ymdhms)
        assert result[4] == 31
        assert result[5] == 0.0

    def test_date_round_minutes_overflow(self):
        """Test date_round with minutes == 60."""
        ymdhms = [2024, 7, 9, 15, 60, 0.0]
        result = date_round(ymdhms)
        assert result[3] == 16
        assert result[4] == 0

    def test_date_round_hours_overflow(self):
        """Test date_round with hours == 24."""
        ymdhms = [2024, 7, 9, 24, 0, 0.0]
        result = date_round(ymdhms)
        assert result[2] == 10
        assert result[3] == 0

    def test_date_round_day_overflow_31day_month(self):
        """Test date_round with day overflow in 31-day month."""
        ymdhms = [2024, 7, 32, 0, 0, 0.0]
        result = date_round(ymdhms)
        assert result[1] == 8
        assert result[2] == 1

    def test_date_round_day_overflow_30day_month(self):
        """Test date_round with day overflow in 30-day month."""
        ymdhms = [2024, 6, 31, 0, 0, 0.0]
        result = date_round(ymdhms)
        assert result[1] == 7
        assert result[2] == 1

    def test_date_round_february_nonleap_overflow(self):
        """Test date_round with Feb 29 in non-leap year."""
        ymdhms = [2023, 2, 29, 0, 0, 0.0]
        result = date_round(ymdhms)
        assert result[1] == 3
        assert result[2] == 1

    def test_date_round_february_leap_overflow(self):
        """Test date_round with Feb 30 in leap year."""
        ymdhms = [2024, 2, 30, 0, 0, 0.0]
        result = date_round(ymdhms)
        assert result[1] == 3
        assert result[2] == 1

    def test_date_round_month_overflow(self):
        """Test date_round with month == 13."""
        ymdhms = [2024, 13, 1, 0, 0, 0.0]
        result = date_round(ymdhms)
        assert result[0] == 2025
        assert result[1] == 1

    def test_date_round_multiple_overflows(self):
        """Test date_round with multiple simultaneous overflows."""
        ymdhms = [2024, 12, 31, 23, 59, 59.999]
        result = date_round(ymdhms)
        assert result[0] == 2025
        assert result[1] == 1
        assert result[2] == 1
        assert result[3] == 0
        assert result[4] == 0
        assert result[5] == 0.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_negative_unix_timestamp(self):
        """Test Unix timestamp before 1970."""
        unix = -86400  # One day before epoch
        jd = unix_to_jd(unix)
        unix2 = jd_to_unix(jd)
        assert abs(unix - unix2) < 1e-6

    def test_very_old_date(self):
        """Test conversion for old date (within algorithm's reliable range)."""
        y1, mo1, d1 = 1950, 1, 1
        jd = ymdhms_to_jd(y1, mo1, d1, 12, 0, 0)
        y2, mo2, d2, h2, m2, s2 = jd_to_ymdhms(jd)
        assert y1 == y2
        assert mo1 == mo2
        assert d1 == d2

    def test_far_future_date(self):
        """Test conversion for future date (within algorithm's reliable range)."""
        y1, mo1, d1 = 2050, 12, 31
        jd = ymdhms_to_jd(y1, mo1, d1, 12, 0, 0)
        y2, mo2, d2, h2, m2, s2 = jd_to_ymdhms(jd)
        assert y1 == y2
        assert mo1 == mo2
        assert d1 == d2

    def test_high_precision_seconds(self):
        """Test high precision fractional seconds."""
        s1 = 45.123456789
        jd = ymdhms_to_jd(2024, 7, 9, 12, 0, s1)
        y, mo, d, h, m, s2 = jd_to_ymdhms(jd)
        # Expect some loss of precision due to algorithm precision
        assert abs(s1 - s2) < 1e-4

    def test_array_operations(self):
        """Test that array operations work correctly."""
        jds = np.array([c.J2000, c.J2000 + 1, c.J2000 + 2])

        # Test JD to MJD
        mjds = jd_to_mjd(jds)
        assert isinstance(mjds, np.ndarray)
        assert len(mjds) == 3

        # Test MJD to JD
        jds2 = mjd_to_jd(mjds)
        assert np.allclose(jds, jds2)

        # Test JD to Unix
        unix = jd_to_unix(jds)
        assert isinstance(unix, np.ndarray)
        assert len(unix) == 3

        # Test Unix to JD
        jds3 = unix_to_jd(unix)
        assert np.allclose(jds, jds3)
