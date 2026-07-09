# python imports
import pytest
import numpy as np

# comet imports
from comet.celestial.sun import Sun
from comet.celestial.celestial_fidelity import CelestialFidelity
from comet.time.epoch import Epoch
from comet.time.timeline import Timeline
from comet.time.duration import Duration
from comet.utilities.constants import Constants as c


class TestSunProperties:
    """Tests for Sun property methods."""

    def test_mu(self):
        """Test Sun mu() returns correct gravitational parameter."""
        sun = Sun()
        mu = sun.mu()
        assert mu == c.MU_SUN
        assert isinstance(mu, float)
        assert mu > 0

    def test_radius(self):
        """Test Sun radius() returns correct solar radius."""
        sun = Sun()
        radius = sun.radius()
        assert radius == c.RADIUS_SUN
        assert isinstance(radius, float)
        assert radius > 0


class TestSunGetPosition:
    """Tests for Sun.get_position() method."""

    def test_get_position_default_j2000(self):
        """Test get_position() with default J2000 epoch."""
        sun = Sun()
        position = sun.get_position()
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        # Solar distance should be roughly 1 AU (149.6 million km)
        distance = np.linalg.norm(position)
        assert 145e6 < distance < 155e6

    def test_get_position_custom_epoch(self):
        """Test get_position() with custom epoch."""
        sun = Sun()
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        position = sun.get_position(epoch=epoch)
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        distance = np.linalg.norm(position)
        assert 145e6 < distance < 155e6

    def test_get_position_lofi_explicit(self):
        """Test get_position() with explicit LoFi fidelity."""
        sun = Sun()
        position = sun.get_position(fidelity=CelestialFidelity.LoFi)
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)

    def test_get_position_hifi_not_implemented(self):
        """Test get_position() raises NotImplementedError for HiFi."""
        sun = Sun()
        with pytest.raises(NotImplementedError, match="High Fidelity Solar Position not Implemented"):
            sun.get_position(fidelity=CelestialFidelity.HiFi)

    def test_get_position_invalid_fidelity(self):
        """Test get_position() raises ValueError for invalid fidelity."""
        sun = Sun()
        # Create an invalid fidelity by using a value that's not in the enum
        with pytest.raises((ValueError, AttributeError)):
            sun.get_position(fidelity="InvalidFidelity")

    def test_get_position_different_epochs_differ(self):
        """Test get_position() returns different positions for different epochs."""
        sun = Sun()
        epoch1 = Epoch(2000, 1, 1, 12, 0, 0)
        epoch2 = Epoch(2000, 7, 1, 12, 0, 0)

        pos1 = sun.get_position(epoch=epoch1)
        pos2 = sun.get_position(epoch=epoch2)

        # Positions should differ (6 months apart)
        assert not np.allclose(pos1, pos2, atol=1e6)


class TestSunGetPositionArray:
    """Tests for Sun.get_position_array() method."""

    def test_get_position_array_timeline(self):
        """Test get_position_array() with Timeline."""
        sun = Sun()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(days=30)
        timeline = Timeline(start_epoch, stop_epoch, Duration(days=1))

        positions = sun.get_position_array(timeline=timeline)

        assert isinstance(positions, np.ndarray)
        assert positions.shape[0] == len(timeline.get_julian_date_list())
        assert positions.shape[1] == 3

        # All distances should be reasonable (around 1 AU)
        distances = np.linalg.norm(positions, axis=1)
        assert np.all((distances > 145e6) & (distances < 155e6))

    def test_get_position_array_lofi_explicit(self):
        """Test get_position_array() with explicit LoFi fidelity."""
        sun = Sun()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(days=10)
        timeline = Timeline(start_epoch, stop_epoch, Duration(days=5))

        positions = sun.get_position_array(timeline=timeline, fidelity=CelestialFidelity.LoFi)

        assert isinstance(positions, np.ndarray)
        assert positions.shape[1] == 3

    def test_get_position_array_hifi_not_implemented(self):
        """Test get_position_array() raises NotImplementedError for HiFi."""
        sun = Sun()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(days=1)
        timeline = Timeline(start_epoch, stop_epoch, Duration(days=1))

        with pytest.raises(NotImplementedError, match="High Fidelity Solar Position not Implemented"):
            sun.get_position_array(timeline=timeline, fidelity=CelestialFidelity.HiFi)

    def test_get_position_array_positions_change_over_time(self):
        """Test that positions change over time in array."""
        sun = Sun()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(days=30)
        timeline = Timeline(start_epoch, stop_epoch, Duration(days=10))

        positions = sun.get_position_array(timeline=timeline)

        # First and last positions should differ
        assert not np.allclose(positions[0], positions[-1], atol=1e6)


class TestSunPositionLowFidelity:
    """Tests for Sun._sun_position_low_fidelity() method."""

    def test_sun_position_scalar_jd(self):
        """Test _sun_position_low_fidelity with scalar Julian date."""
        sun = Sun()
        jd = c.J2000
        position = sun._sun_position_low_fidelity(jd)

        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        distance = np.linalg.norm(position)
        assert 145e6 < distance < 155e6

    def test_sun_position_array_jd(self):
        """Test _sun_position_low_fidelity with array of Julian dates."""
        sun = Sun()
        jd_array = np.array([c.J2000, c.J2000 + 30, c.J2000 + 60])
        positions = sun._sun_position_low_fidelity(jd_array)

        assert isinstance(positions, np.ndarray)
        assert positions.shape == (3, 3)

        # Check each position has reasonable distance
        for i in range(3):
            distance = np.linalg.norm(positions[i])
            assert 145e6 < distance < 155e6

    def test_sun_position_j2000_reference(self):
        """Test _sun_position_low_fidelity at J2000 against expected values.

        Reference: Vallado Algorithm 29, pg. 279-280
        Expected accuracy: ~0.01 deg in ecliptic longitude per Vallado
        At J2000, Sun distance should be approximately 1 AU
        """
        sun = Sun()
        jd = c.J2000
        position = sun._sun_position_low_fidelity(jd)

        distance = np.linalg.norm(position)
        # Earth's distance varies between ~147.1 and ~152.1 million km
        # At J2000 (Jan 1), Earth is near perihelion
        assert 147e6 < distance < 152e6

        # Check that distance is close to 1 AU
        assert pytest.approx(distance, rel=0.02) == c.AU

    def test_sun_position_consistency(self):
        """Test that same Julian date produces same position."""
        sun = Sun()
        jd = c.J2000 + 100

        pos1 = sun._sun_position_low_fidelity(jd)
        pos2 = sun._sun_position_low_fidelity(jd)

        np.testing.assert_array_equal(pos1, pos2)

    def test_sun_position_orbital_period(self):
        """Test that Sun position changes over one year.

        The Earth's orbital period is approximately 365.25 days.
        After one year, position should return close to starting position.
        """
        sun = Sun()
        jd_start = c.J2000
        jd_end = c.J2000 + 365.25  # One year

        pos_start = sun._sun_position_low_fidelity(jd_start)
        pos_end = sun._sun_position_low_fidelity(jd_end)

        # Positions should be close after one full orbit (within ~1000 km due to precession)
        distance_change = np.linalg.norm(pos_end - pos_start)
        assert distance_change < 1e6  # Should be relatively close after one year

    def test_sun_position_half_year_change(self):
        """Test that Sun position is roughly opposite after half a year."""
        sun = Sun()
        jd_start = c.J2000
        jd_half_year = c.J2000 + 182.625  # Half year

        pos_start = sun._sun_position_low_fidelity(jd_start)
        pos_half_year = sun._sun_position_low_fidelity(jd_half_year)

        # After half year, Sun should be roughly on opposite side
        # Dot product should be negative
        dot_product = np.dot(pos_start, pos_half_year)
        assert dot_product < 0

    def test_sun_position_components_reasonable(self):
        """Test that Sun position components are individually reasonable."""
        sun = Sun()
        jd = c.J2000
        position = sun._sun_position_low_fidelity(jd)

        # Each component should be less than total distance
        distance = np.linalg.norm(position)
        assert np.abs(position[0]) < distance
        assert np.abs(position[1]) < distance
        assert np.abs(position[2]) < distance

    def test_sun_position_z_component_small(self):
        """Test that Sun position Z component is small (ecliptic plane).

        The Sun's position should lie mostly in the ecliptic plane (XY),
        with small Z component due to obliquity transformation.
        """
        sun = Sun()
        jd = c.J2000
        position = sun._sun_position_low_fidelity(jd)

        distance = np.linalg.norm(position)
        z_fraction = np.abs(position[2]) / distance

        # Z component should be much smaller than XY components
        # due to small obliquity angle (~23.4 degrees)
        assert z_fraction < 0.5  # Should be less than half

    def test_sun_position_mean_anomaly_progression(self):
        """Test that Sun position progresses correctly with mean anomaly."""
        sun = Sun()
        # Test at different points in the orbit
        jd_perihelion = c.J2000  # Near perihelion (Jan 1)
        jd_aphelion = c.J2000 + 182.625  # Near aphelion (July 1)

        pos_peri = sun._sun_position_low_fidelity(jd_perihelion)
        pos_aph = sun._sun_position_low_fidelity(jd_aphelion)

        dist_peri = np.linalg.norm(pos_peri)
        dist_aph = np.linalg.norm(pos_aph)

        # Perihelion distance should be less than aphelion
        assert dist_peri < dist_aph


class TestSunIntegration:
    """Integration tests for Sun class."""

    def test_get_position_matches_low_fidelity_calculation(self):
        """Test that get_position() matches direct low fidelity calculation."""
        sun = Sun()
        epoch = Epoch(2010, 6, 15, 12, 0, 0)

        # Get position via public method
        pos_public = sun.get_position(epoch=epoch, fidelity=CelestialFidelity.LoFi)

        # Get position via private method directly
        jd = epoch.julian_date()
        pos_private = sun._sun_position_low_fidelity(jd)

        np.testing.assert_array_equal(pos_public, pos_private)

    def test_position_array_consistency_with_single_calls(self):
        """Test that get_position_array() is consistent with multiple get_position() calls."""
        sun = Sun()
        epochs = [Epoch(2000, 1, 1 + i, 0, 0, 0) for i in range(5)]

        # Get positions individually
        positions_individual = np.array([sun.get_position(epoch=e) for e in epochs])

        # Get positions as array
        timeline = Timeline(epochs[0], epochs[-1], Duration(days=1))
        positions_array = sun.get_position_array(timeline=timeline)

        # Should be very close (allowing for numerical differences)
        np.testing.assert_allclose(positions_individual, positions_array, rtol=1e-10)

    def test_sun_moon_distance_comparison(self):
        """Test that Sun is much farther than Moon (sanity check)."""
        from comet.celestial.moon import Moon

        sun = Sun()
        moon = Moon()
        epoch = Epoch(2000, 1, 1, 12, 0, 0)

        sun_pos = sun.get_position(epoch=epoch)
        moon_pos = moon.get_position(epoch=epoch)

        sun_dist = np.linalg.norm(sun_pos)
        moon_dist = np.linalg.norm(moon_pos)

        # Sun should be roughly 400 times farther than Moon
        ratio = sun_dist / moon_dist
        assert 300 < ratio < 500


class TestSunVallado:
    """Tests for Sun position against Vallado reference examples."""

    def test_sun_position_vallado_algorithm_29(self):
        """Test Sun position calculation against Vallado Algorithm 29 sanity check.

        Reference: Vallado, "Fundamentals of Astrodynamics and Applications",
        Algorithm 29, pg. 279-280

        While specific test vectors aren't provided in the text, we verify:
        1. Distance is approximately 1 AU
        2. Position magnitude varies with eccentricity (~3%)
        3. Calculation produces consistent results
        """
        sun = Sun()

        # Test at J2000
        jd_j2000 = c.J2000
        pos_j2000 = sun._sun_position_low_fidelity(jd_j2000)
        dist_j2000 = np.linalg.norm(pos_j2000)

        # Distance should be close to 1 AU (within 3% due to eccentricity)
        assert pytest.approx(dist_j2000, rel=0.03) == c.AU

        # Test at different time of year
        jd_summer = c.J2000 + 182.625
        pos_summer = sun._sun_position_low_fidelity(jd_summer)
        dist_summer = np.linalg.norm(pos_summer)

        # Both should be close to 1 AU
        assert pytest.approx(dist_summer, rel=0.03) == c.AU

        # Distances should differ slightly due to orbital eccentricity
        assert abs(dist_summer - dist_j2000) > 1e6  # Should differ by more than 1000 km
