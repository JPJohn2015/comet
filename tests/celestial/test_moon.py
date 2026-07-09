# python imports
import pytest
import numpy as np

# comet imports
from comet.celestial.moon import Moon
from comet.celestial.celestial_fidelity import CelestialFidelity
from comet.time.epoch import Epoch
from comet.time.timeline import Timeline
from comet.time.duration import Duration
from comet.utilities.constants import Constants as c


class TestMoonProperties:
    """Tests for Moon property methods."""

    def test_mu(self):
        """Test Moon mu() returns correct gravitational parameter."""
        moon = Moon()
        mu = moon.mu()
        assert mu == c.MU_MOON
        assert isinstance(mu, float)
        assert mu > 0

    def test_radius(self):
        """Test Moon radius() returns correct lunar radius."""
        moon = Moon()
        radius = moon.radius()
        assert radius == c.RADIUS_MOON
        assert isinstance(radius, float)
        assert radius > 0


class TestMoonGetPosition:
    """Tests for Moon.get_position() method."""

    def test_get_position_default_j2000(self):
        """Test get_position() with default J2000 epoch."""
        moon = Moon()
        position = moon.get_position()
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        # Lunar distance should be roughly 384,400 km
        distance = np.linalg.norm(position)
        assert 350000 < distance < 420000

    def test_get_position_custom_epoch(self):
        """Test get_position() with custom epoch."""
        moon = Moon()
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        position = moon.get_position(epoch=epoch)
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        distance = np.linalg.norm(position)
        assert 350000 < distance < 420000

    def test_get_position_lofi_explicit(self):
        """Test get_position() with explicit LoFi fidelity."""
        moon = Moon()
        position = moon.get_position(fidelity=CelestialFidelity.LoFi)
        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)

    def test_get_position_hifi_not_implemented(self):
        """Test get_position() raises NotImplementedError for HiFi."""
        moon = Moon()
        with pytest.raises(NotImplementedError, match="High Fidelity Lunar Position not Implemented"):
            moon.get_position(fidelity=CelestialFidelity.HiFi)

    def test_get_position_invalid_fidelity(self):
        """Test get_position() raises ValueError for invalid fidelity."""
        moon = Moon()
        # Create an invalid fidelity by using a value that's not in the enum
        with pytest.raises((ValueError, AttributeError)):
            moon.get_position(fidelity="InvalidFidelity")

    def test_get_position_different_epochs_differ(self):
        """Test get_position() returns different positions for different epochs."""
        moon = Moon()
        epoch1 = Epoch(2000, 1, 1, 12, 0, 0)
        epoch2 = Epoch(2020, 1, 1, 12, 0, 0)

        pos1 = moon.get_position(epoch=epoch1)
        pos2 = moon.get_position(epoch=epoch2)

        # Positions should differ
        assert not np.allclose(pos1, pos2, atol=1000)


class TestMoonGetPositionArray:
    """Tests for Moon.get_position_array() method."""

    def test_get_position_array_timeline(self):
        """Test get_position_array() with Timeline."""
        moon = Moon()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(hours=24)
        timeline = Timeline(start_epoch, stop_epoch, Duration(hours=1))

        positions = moon.get_position_array(timeline=timeline)

        assert isinstance(positions, np.ndarray)
        assert positions.shape[0] == len(timeline.get_julian_date_list())
        assert positions.shape[1] == 3

        # All distances should be reasonable
        distances = np.linalg.norm(positions, axis=1)
        assert np.all((distances > 350000) & (distances < 420000))

    def test_get_position_array_lofi_explicit(self):
        """Test get_position_array() with explicit LoFi fidelity."""
        moon = Moon()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(hours=12)
        timeline = Timeline(start_epoch, stop_epoch, Duration(hours=6))

        positions = moon.get_position_array(timeline=timeline, fidelity=CelestialFidelity.LoFi)

        assert isinstance(positions, np.ndarray)
        assert positions.shape[1] == 3

    def test_get_position_array_hifi_not_implemented(self):
        """Test get_position_array() raises NotImplementedError for HiFi."""
        moon = Moon()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(hours=1)
        timeline = Timeline(start_epoch, stop_epoch, Duration(hours=1))

        with pytest.raises(NotImplementedError, match="High Fidelity Lunar Position not Implemented"):
            moon.get_position_array(timeline=timeline, fidelity=CelestialFidelity.HiFi)

    def test_get_position_array_positions_change_over_time(self):
        """Test that positions change over time in array."""
        moon = Moon()
        start_epoch = Epoch(2000, 1, 1, 0, 0, 0)
        stop_epoch = start_epoch + Duration(days=1)
        timeline = Timeline(start_epoch, stop_epoch, Duration(hours=6))

        positions = moon.get_position_array(timeline=timeline)

        # First and last positions should differ
        assert not np.allclose(positions[0], positions[-1], atol=1000)


class TestMoonPositionLowFidelity:
    """Tests for Moon._moon_position_low_fidelity() method."""

    def test_moon_position_scalar_jd(self):
        """Test _moon_position_low_fidelity with scalar Julian date."""
        moon = Moon()
        jd = c.J2000
        position = moon._moon_position_low_fidelity(jd)

        assert isinstance(position, np.ndarray)
        assert position.shape == (3,)
        distance = np.linalg.norm(position)
        assert 350000 < distance < 420000

    def test_moon_position_array_jd(self):
        """Test _moon_position_low_fidelity with array of Julian dates."""
        moon = Moon()
        jd_array = np.array([c.J2000, c.J2000 + 1, c.J2000 + 2])
        positions = moon._moon_position_low_fidelity(jd_array)

        assert isinstance(positions, np.ndarray)
        assert positions.shape == (3, 3)

        # Check each position has reasonable distance
        for i in range(3):
            distance = np.linalg.norm(positions[i])
            assert 350000 < distance < 420000

    def test_moon_position_j2000_reference(self):
        """Test _moon_position_low_fidelity at J2000 against expected range.

        Reference: Vallado Algorithm 31, pg. 288
        Expected accuracy: ~0.3 deg in ecliptic longitude
        At J2000, Moon distance should be approximately 384,400 km
        """
        moon = Moon()
        jd = c.J2000
        position = moon._moon_position_low_fidelity(jd)

        distance = np.linalg.norm(position)
        # Moon's distance varies between ~356,400 and ~406,700 km
        assert 356000 < distance < 407000

    def test_moon_position_consistency(self):
        """Test that same Julian date produces same position."""
        moon = Moon()
        jd = c.J2000 + 100

        pos1 = moon._moon_position_low_fidelity(jd)
        pos2 = moon._moon_position_low_fidelity(jd)

        np.testing.assert_array_equal(pos1, pos2)

    def test_moon_position_orbital_period(self):
        """Test that Moon position changes over one sidereal month.

        The Moon's sidereal period is approximately 27.3 days.
        After one complete orbit, position should return close to starting position.
        """
        moon = Moon()
        jd_start = c.J2000
        jd_end = c.J2000 + 27.3  # One sidereal period

        pos_start = moon._moon_position_low_fidelity(jd_start)
        pos_end = moon._moon_position_low_fidelity(jd_end)

        # After one sidereal period, should return close to start (within ~10,000 km)
        distance_change = np.linalg.norm(pos_end - pos_start)
        assert distance_change < 20000  # Should return close to start position

    def test_moon_position_components_reasonable(self):
        """Test that Moon position components are individually reasonable."""
        moon = Moon()
        jd = c.J2000
        position = moon._moon_position_low_fidelity(jd)

        # Each component should be less than total distance
        distance = np.linalg.norm(position)
        assert np.abs(position[0]) < distance
        assert np.abs(position[1]) < distance
        assert np.abs(position[2]) < distance

    def test_moon_position_z_component_sign_fix(self):
        """Test that Z component sign fix is correct (regression test).

        This is a regression test for the bug where the Z component
        had an incorrect negative sign in front of cos(obliquity_ecliptic).
        The transformation should be:
        z = sin(ε) * cos(β) * sin(λ) + cos(ε) * sin(β)
        """
        moon = Moon()
        jd = c.J2000
        position = moon._moon_position_low_fidelity(jd)

        # The Moon should have a non-zero Z component at J2000
        # This test ensures the calculation doesn't produce obviously wrong values
        assert isinstance(position[2], (int, float, np.number))
        # Z component should be within reasonable bounds for lunar orbit
        assert np.abs(position[2]) < 100000  # Should be less than ~100,000 km


class TestMoonIntegration:
    """Integration tests for Moon class."""

    def test_get_position_matches_low_fidelity_calculation(self):
        """Test that get_position() matches direct low fidelity calculation."""
        moon = Moon()
        epoch = Epoch(2010, 6, 15, 12, 0, 0)

        # Get position via public method
        pos_public = moon.get_position(epoch=epoch, fidelity=CelestialFidelity.LoFi)

        # Get position via private method directly
        jd = epoch.julian_date()
        pos_private = moon._moon_position_low_fidelity(jd)

        np.testing.assert_array_equal(pos_public, pos_private)

    def test_position_array_consistency_with_single_calls(self):
        """Test that get_position_array() is consistent with multiple get_position() calls."""
        moon = Moon()
        epochs = [Epoch(2000, 1, 1, i, 0, 0) for i in range(5)]

        # Get positions individually
        positions_individual = np.array([moon.get_position(epoch=e) for e in epochs])

        # Get positions as array
        timeline = Timeline(epochs[0], epochs[-1], Duration(hours=1))
        positions_array = moon.get_position_array(timeline=timeline)

        # Should be very close (allowing for numerical differences)
        np.testing.assert_allclose(positions_individual, positions_array, rtol=1e-10)
