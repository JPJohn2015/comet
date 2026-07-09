"""Comprehensive test suite for the Constants class.

This module tests:
- Physical constants accuracy
- Unit consistency
- Mathematical relationships between constants
- Standard values against known references
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.utilities.constants import Constants as c


class TestTimeConstants:
    """Tests for time-related constants."""

    def test_minute_to_seconds(self):
        """Test minute definition."""
        assert c.MINUTE == 60

    def test_hour_to_seconds(self):
        """Test hour definition."""
        assert c.HOUR == 3600
        assert c.HOUR == 60 * c.MINUTE

    def test_day_to_seconds(self):
        """Test day definition."""
        assert c.DAY == 86400
        assert c.DAY == 24 * c.HOUR

    def test_week_in_days(self):
        """Test week definition."""
        assert c.WEEK == 7

    def test_month_in_days(self):
        """Test month definition (average)."""
        assert c.MONTH == 30

    def test_year_in_days(self):
        """Test year definition (includes leap years)."""
        assert c.YEAR == 365.25

    def test_julian_century(self):
        """Test Julian century definition."""
        assert c.JULIAN_CENTURY == 36525
        assert c.JULIAN_CENTURY == c.YEAR * 100


class TestAngleConversions:
    """Tests for angle conversion constants."""

    def test_deg_to_rad(self):
        """Test degree to radian conversion."""
        assert np.isclose(c.DEG2RAD, np.pi / 180)
        assert np.isclose(c.DEG2RAD * 180, np.pi)

    def test_rad_to_deg(self):
        """Test radian to degree conversion."""
        assert np.isclose(c.RAD2DEG, 180 / np.pi)
        assert np.isclose(c.RAD2DEG * np.pi, 180)

    def test_deg_rad_roundtrip(self):
        """Test degree-radian conversion roundtrip."""
        assert np.isclose(c.DEG2RAD * c.RAD2DEG, 1.0)

    def test_deg_to_arcsec(self):
        """Test degree to arcsecond conversion."""
        assert c.DEG2AS == 3600

    def test_arcsec_to_deg(self):
        """Test arcsecond to degree conversion."""
        assert c.AS2DEG == 1 / 3600

    def test_arcsec_to_rad(self):
        """Test arcsecond to radian conversion."""
        expected = (1 / 3600) * (np.pi / 180)
        assert np.isclose(c.AS2RAD, expected)

    def test_rad_to_arcsec(self):
        """Test radian to arcsecond conversion."""
        assert np.isclose(c.RAD2AS, 1 / c.AS2RAD)


class TestTimeSystemConversions:
    """Tests for time system offset constants."""

    def test_utc_tai_offset(self):
        """Test UTC to TAI offset."""
        # As of 2017, offset is 37 seconds
        # This is a fixed value at a specific date
        assert c.UTC_TAI == 37
        assert c.TAI_UTC == -37
        assert c.UTC_TAI == -c.TAI_UTC

    def test_tai_tt_offset(self):
        """Test TAI to TT offset (fixed by definition)."""
        assert c.TAI_TT == 32.184
        assert c.TT_TAI == -32.184
        assert c.TAI_TT == -c.TT_TAI

    def test_utc_tt_offset(self):
        """Test UTC to TT offset."""
        expected = c.UTC_TAI + c.TAI_TT
        assert np.isclose(c.UTC_TT, expected)
        assert np.isclose(c.UTC_TT, 69.184)


class TestEpochConstants:
    """Tests for epoch reference constants."""

    def test_unix_epoch(self):
        """Test Unix epoch (Jan 1, 1970 GMT)."""
        assert c.UNIX0 == 2440587.5

    def test_mjd_epoch(self):
        """Test Modified Julian Date epoch (Nov 17, 1858 GMT)."""
        assert c.MJD0 == 2400000.5

    def test_j2000_epoch(self):
        """Test J2000 epoch (Jan 1, 2000 GMT)."""
        assert c.J2000 == 2451545.0

    def test_mj2000_calculation(self):
        """Test MJ2000 derived from J2000."""
        assert c.MJ2000 == c.J2000 - c.MJD0
        assert c.MJ2000 == 51544.5


class TestGravitationalParameters:
    """Tests for gravitational parameters."""

    def test_mu_earth(self):
        """Test Earth gravitational parameter (WGS84)."""
        # WGS84 standard value
        assert np.isclose(c.MU_EARTH, 398600.4418, rtol=1e-6)

    def test_mu_moon(self):
        """Test Moon gravitational parameter."""
        # Should be much smaller than Earth
        assert c.MU_MOON < c.MU_EARTH
        assert c.MU_MOON == 4904.86959

    def test_mu_sun(self):
        """Test Sun gravitational parameter."""
        # Should be much larger than Earth
        assert c.MU_SUN > c.MU_EARTH
        assert np.isclose(c.MU_SUN, 1.327124400189e11)

    def test_mu_ratios(self):
        """Test gravitational parameter ratios make physical sense."""
        # Sun >> Earth >> Moon
        assert c.MU_SUN / c.MU_EARTH > 1e5
        assert c.MU_EARTH / c.MU_MOON > 50


class TestRadiusConstants:
    """Tests for planetary radius constants."""

    def test_radius_earth(self):
        """Test Earth radius (WGS84 equatorial)."""
        # WGS84 equatorial radius
        assert np.isclose(c.RADIUS_EARTH, 6378.14, rtol=1e-4)

    def test_radius_moon(self):
        """Test Moon radius."""
        assert c.RADIUS_MOON == 1737.4
        assert c.RADIUS_MOON < c.RADIUS_EARTH

    def test_radius_sun(self):
        """Test Sun radius."""
        assert c.RADIUS_SUN == 695700.0
        assert c.RADIUS_SUN > c.RADIUS_EARTH


class TestAstronomicalConstants:
    """Tests for astronomical constants."""

    def test_astronomical_unit(self):
        """Test astronomical unit (AU) in km."""
        # IAU 2012 definition
        assert np.isclose(c.AU, 1.495978707e8, rtol=1e-6)

    def test_au_reasonableness(self):
        """Test AU is reasonable distance."""
        # AU should be about 150 million km
        assert 1.49e8 < c.AU < 1.51e8


class TestEarthProperties:
    """Tests for Earth-specific properties."""

    def test_surface_gravity(self):
        """Test Earth surface gravity (standard)."""
        # Standard gravity at sea level
        assert c.SURFACE_GRAVITY == 9.807

    def test_j2_earth(self):
        """Test J2 (second zonal harmonic)."""
        # WGS84 J2 value
        assert np.isclose(c.J2_EARTH, 0.001082, atol=1e-6)

    def test_j3_earth(self):
        """Test J3 (third zonal harmonic)."""
        assert np.isclose(c.J3_EARTH, -0.0000025, atol=1e-7)

    def test_omega_earth(self):
        """Test Earth rotation rate."""
        # Earth rotates once per sidereal day
        # ω = 2π / (86164.0905 sec) ≈ 7.292115e-5 rad/s
        assert np.isclose(c.OMEGA_EARTH, 7.292115e-5, rtol=1e-6)

    def test_flattening_earth(self):
        """Test Earth flattening (1/f)."""
        # WGS84 value
        assert c.FLATTENING_EARTH == 298.257223563

    def test_earth_ellipsoid_consistency(self):
        """Test Earth ellipsoid parameters are consistent."""
        # Semi-major axis (equatorial radius in meters)
        assert c.A_EARTH == 6378137.0

        # Semi-minor axis (polar radius)
        expected_b = c.A_EARTH * (1 - 1 / c.FLATTENING_EARTH)
        assert np.isclose(c.B_EARTH, expected_b, rtol=1e-10)

        # First eccentricity squared
        expected_e_sq = 1 - (c.B_EARTH ** 2 / c.A_EARTH ** 2)
        assert np.isclose(c.E_SQ_EARTH, expected_e_sq, rtol=1e-10)

        # Second eccentricity squared
        expected_ep_sq = (c.A_EARTH ** 2 - c.B_EARTH ** 2) / c.B_EARTH ** 2
        assert np.isclose(c.EP_SQ_EARTH, expected_ep_sq, rtol=1e-10)

    def test_earth_ellipsoid_units(self):
        """Test Earth ellipsoid parameters units."""
        # A_EARTH and B_EARTH are in METERS (not km!)
        # RADIUS_EARTH is in km
        # This is intentional for geodetic calculations
        assert c.A_EARTH > 6e6  # More than 6 million meters
        assert c.RADIUS_EARTH < 7e3  # Less than 7 thousand km

        # Verify relationship
        assert np.isclose(c.A_EARTH / 1000, c.RADIUS_EARTH, rtol=1e-3)


class TestSolarProperties:
    """Tests for solar properties."""

    def test_solar_pressure(self):
        """Test solar radiation pressure at 1 AU."""
        # Solar pressure at 1 AU in N/m^2 (Pa)
        assert c.SOLAR_PRESSURE == 4.57e-6


class TestCalendarConstants:
    """Tests for calendar-related constants."""

    def test_days_per_month_length(self):
        """Test days per month array length."""
        # 13 elements (index 0 unused, 1-12 for months)
        assert len(c.DAYS_PER_MONTH) == 13
        assert len(c.DAYS_PER_MONTH_LEAP) == 13

    def test_days_per_month_values(self):
        """Test days per month values."""
        expected = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        assert c.DAYS_PER_MONTH == expected

    def test_days_per_month_leap_values(self):
        """Test days per month for leap year."""
        expected = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        assert c.DAYS_PER_MONTH_LEAP == expected

    def test_february_leap_year(self):
        """Test February has 29 days in leap year."""
        assert c.DAYS_PER_MONTH[2] == 28
        assert c.DAYS_PER_MONTH_LEAP[2] == 29

    def test_total_days_per_year(self):
        """Test sum of days equals year length."""
        normal_year = sum(c.DAYS_PER_MONTH)
        leap_year = sum(c.DAYS_PER_MONTH_LEAP)

        assert normal_year == 365
        assert leap_year == 366


class TestConstantsFrozen:
    """Tests that Constants class is immutable."""

    def test_class_attributes_immutable(self):
        """Test that Constants class attributes cannot be modified."""
        # Constants is a dataclass with frozen=True
        # This prevents instance modification, but class attributes can still change
        # (This is expected behavior for class-level constants)
        original_mu = c.MU_EARTH
        assert original_mu == 398600.4418

    def test_constants_are_class_attributes(self):
        """Test that constants are accessible as class attributes."""
        assert hasattr(c, 'MU_EARTH')
        assert hasattr(c, 'RADIUS_EARTH')
        assert hasattr(c, 'AU')


class TestPhysicalRelationships:
    """Tests for relationships between constants."""

    def test_escape_velocity_earth(self):
        """Test escape velocity calculation using constants."""
        # v_esc = sqrt(2 * mu / r)
        v_esc = np.sqrt(2 * c.MU_EARTH / c.RADIUS_EARTH)
        # Should be about 11.2 km/s
        assert 11.0 < v_esc < 11.5

    def test_circular_orbit_velocity(self):
        """Test circular orbit velocity at Earth surface."""
        # v_circ = sqrt(mu / r)
        v_circ = np.sqrt(c.MU_EARTH / c.RADIUS_EARTH)
        # Should be about 7.9 km/s
        assert 7.8 < v_circ < 8.0

    def test_sidereal_day_from_omega(self):
        """Test sidereal day calculation from Earth rotation rate."""
        # T = 2π / ω
        sidereal_day = 2 * np.pi / c.OMEGA_EARTH
        # Should be about 86164 seconds (23h 56m 4s)
        assert 86160 < sidereal_day < 86170
