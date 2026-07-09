"""Comprehensive test suite for the astro.py utility functions.

This module tests:
- Semi-major axis and mean motion conversions
- Period conversions
- Anomaly conversions (mean, eccentric, true)
- Time-to-apse calculations
- Edge cases and error handling
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.utilities.astro import (
    semimajor_axis_to_mean_motion,
    mean_motion_to_semimajor_axis,
    semimajoraxis_to_period,
    period_to_semimajor_axis,
    eccentric_to_mean_anomaly,
    mean_to_eccentric_anomaly,
    true_to_eccentric_anomaly,
    eccentric_to_true_anomaly,
    true_to_mean_anomaly,
    mean_to_true_anomaly,
    time_to_perigee,
    time_since_perigee,
    time_to_apogee,
    time_since_apogee,
)
from comet.utilities.constants import Constants as c


class TestSemiMajorAxisMeanMotion:
    """Tests for semi-major axis and mean motion conversions."""

    def test_semimajor_axis_to_mean_motion(self):
        """Test conversion from semi-major axis to mean motion."""
        # ISS-like orbit at ~7000 km
        a = 7000  # km
        n = semimajor_axis_to_mean_motion(a)

        # Mean motion should be positive
        assert n > 0

        # Check against expected value (n = sqrt(mu/a^3))
        expected = np.sqrt(c.MU_EARTH / a**3)
        assert np.isclose(n, expected)

    def test_mean_motion_to_semimajor_axis(self):
        """Test conversion from mean motion to semi-major axis."""
        # ISS completes ~15.5 orbits per day
        n = 15.5 * (2*np.pi) / c.DAY  # rad/s
        a = mean_motion_to_semimajor_axis(n)

        # Semi-major axis should be reasonable for LEO
        assert 6500 < a < 7500

    def test_semimajor_axis_mean_motion_roundtrip(self):
        """Test roundtrip conversion."""
        a_orig = 7000
        n = semimajor_axis_to_mean_motion(a_orig)
        a_back = mean_motion_to_semimajor_axis(n)

        assert np.isclose(a_orig, a_back)

    def test_geostationary_orbit(self):
        """Test mean motion for geostationary orbit."""
        # GEO orbit period = 1 sidereal day
        n_geo = 2*np.pi / (23.9345 * 3600)  # rad/s
        a = mean_motion_to_semimajor_axis(n_geo)

        # GEO altitude is ~35,786 km, so SMA ~42,164 km
        assert 42000 < a < 42500


class TestPeriodConversions:
    """Tests for period conversions."""

    def test_semimajoraxis_to_period(self):
        """Test conversion from semi-major axis to period."""
        a = 7000
        T = semimajoraxis_to_period(a)

        # ISS period is ~90 minutes
        assert 5000 < T < 6000  # seconds

    def test_period_to_semimajor_axis(self):
        """Test conversion from period to semi-major axis."""
        T = 5400  # ~90 minutes in seconds
        a = period_to_semimajor_axis(T)

        # Should give LEO altitude
        assert 6500 < a < 7500

    def test_period_roundtrip(self):
        """Test roundtrip period conversion."""
        a_orig = 8000
        T = semimajoraxis_to_period(a_orig)
        a_back = period_to_semimajor_axis(T)

        assert np.isclose(a_orig, a_back)

    def test_geostationary_period(self):
        """Test period calculation for geostationary orbit."""
        # GEO altitude ~35,786 km, SMA ~42,164 km
        a_geo = 42164
        T = semimajoraxis_to_period(a_geo)

        # Should be ~1 sidereal day (86164 seconds)
        assert 86000 < T < 87000


class TestAnomalyConversions:
    """Tests for anomaly conversions."""

    def test_eccentric_to_mean_anomaly(self):
        """Test conversion from eccentric to mean anomaly."""
        E = np.pi / 2  # 90 degrees
        e = 0.1
        M = eccentric_to_mean_anomaly(E, e)

        # Mean anomaly should be less than eccentric for e > 0
        assert M < E
        assert M > 0

    def test_mean_to_eccentric_anomaly_circular(self):
        """Test conversion for circular orbit (e=0)."""
        M = np.pi / 3
        e = 0.0
        E = mean_to_eccentric_anomaly(M, e)

        # For circular orbit, E = M
        assert np.isclose(E, M)

    def test_mean_to_eccentric_anomaly_elliptical(self):
        """Test conversion for elliptical orbit."""
        M = np.pi / 4
        e = 0.3
        E = mean_to_eccentric_anomaly(M, e)

        # Check it's a valid angle
        assert 0 <= E <= 2*np.pi

        # Verify by converting back
        M_back = eccentric_to_mean_anomaly(E, e)
        assert np.isclose(M, M_back, atol=1e-10)

    def test_mean_to_eccentric_anomaly_high_ecc(self):
        """Test conversion for high eccentricity orbit."""
        M = np.pi / 2
        e = 0.9
        E = mean_to_eccentric_anomaly(M, e)

        # Should converge even for high eccentricity
        assert 0 <= E <= 2*np.pi

        # Verify
        M_back = eccentric_to_mean_anomaly(E, e)
        assert np.isclose(M, M_back, atol=1e-10)

    def test_mean_to_eccentric_anomaly_invalid_ecc(self):
        """Test that e >= 1 raises ValueError."""
        M = np.pi / 2
        e = 1.0  # Parabolic
        with pytest.raises(ValueError):
            mean_to_eccentric_anomaly(M, e)

        e = 1.5  # Hyperbolic
        with pytest.raises(ValueError):
            mean_to_eccentric_anomaly(M, e)

    def test_mean_to_eccentric_anomaly_max_iter(self):
        """Test that max_iter parameter works."""
        M = np.pi / 2
        e = 0.5

        # Should work with default max_iter
        E = mean_to_eccentric_anomaly(M, e)
        assert 0 <= E <= 2*np.pi

        # Should work with custom max_iter
        E = mean_to_eccentric_anomaly(M, e, max_iter=50)
        assert 0 <= E <= 2*np.pi

    def test_true_to_eccentric_anomaly(self):
        """Test conversion from true to eccentric anomaly."""
        nu = np.pi / 3
        e = 0.2
        E = true_to_eccentric_anomaly(nu, e)

        # Check it's a valid angle
        assert 0 <= E <= 2*np.pi

        # Verify by converting back
        nu_back = eccentric_to_true_anomaly(E, e)
        assert np.isclose(nu, nu_back, atol=1e-10)

    def test_eccentric_to_true_anomaly(self):
        """Test conversion from eccentric to true anomaly."""
        E = np.pi / 4
        e = 0.3
        nu = eccentric_to_true_anomaly(E, e)

        # Check it's a valid angle
        assert 0 <= nu <= 2*np.pi

        # Verify by converting back
        E_back = true_to_eccentric_anomaly(nu, e)
        assert np.isclose(E, E_back, atol=1e-10)

    def test_true_to_mean_anomaly(self):
        """Test conversion from true to mean anomaly."""
        nu = np.pi / 2
        e = 0.15
        M = true_to_mean_anomaly(nu, e)

        # Check it's a valid angle
        assert 0 <= M <= 2*np.pi

        # Verify by converting back
        nu_back = mean_to_true_anomaly(M, e)
        assert np.isclose(nu, nu_back, atol=1e-10)

    def test_mean_to_true_anomaly(self):
        """Test conversion from mean to true anomaly."""
        M = np.pi / 3
        e = 0.25
        nu = mean_to_true_anomaly(M, e)

        # Check it's a valid angle
        assert 0 <= nu <= 2*np.pi

        # Verify by converting back
        M_back = true_to_mean_anomaly(nu, e)
        assert np.isclose(M, M_back, atol=1e-10)

    def test_anomaly_at_periapsis(self):
        """Test anomalies at periapsis (all should be 0)."""
        e = 0.3

        # At periapsis
        M = 0
        E = mean_to_eccentric_anomaly(M, e)
        nu = eccentric_to_true_anomaly(E, e)

        assert np.isclose(M, 0)
        assert np.isclose(E, 0, atol=1e-10)
        assert np.isclose(nu, 0, atol=1e-10)

    def test_anomaly_at_apoapsis(self):
        """Test anomalies at apoapsis (all should be π)."""
        e = 0.3

        # At apoapsis
        M = np.pi
        E = mean_to_eccentric_anomaly(M, e)
        nu = eccentric_to_true_anomaly(E, e)

        assert np.isclose(M, np.pi)
        assert np.isclose(E, np.pi, atol=1e-10)
        assert np.isclose(nu, np.pi, atol=1e-10)

    def test_anomaly_vectorization(self):
        """Test that anomaly functions work with arrays."""
        M_array = np.array([0, np.pi/4, np.pi/2, np.pi])
        e = 0.2

        E_array = mean_to_eccentric_anomaly(M_array, e)
        assert E_array.shape == M_array.shape

        nu_array = eccentric_to_true_anomaly(E_array, e)
        assert nu_array.shape == M_array.shape


class TestTimeToApse:
    """Tests for time-to-apse calculations."""

    def test_time_to_perigee(self):
        """Test time to perigee calculation."""
        nu = np.pi / 2  # 90 degrees from periapsis
        a = 7000
        e = 0.1

        dt = time_to_perigee(nu, a, e)

        # Should be positive
        assert dt > 0

        # Should be less than one orbital period
        T = semimajoraxis_to_period(a)
        assert dt < T

    def test_time_since_perigee(self):
        """Test time since perigee calculation."""
        nu = np.pi / 2
        a = 7000
        e = 0.1

        dt = time_since_perigee(nu, a, e)

        # Should be positive
        assert dt > 0

        # Should be less than one orbital period
        T = semimajoraxis_to_period(a)
        assert dt < T

    def test_time_to_apogee(self):
        """Test time to apogee calculation."""
        nu = np.pi / 4  # 45 degrees from periapsis
        a = 7000
        e = 0.1

        dt = time_to_apogee(nu, a, e)

        # Should be positive
        assert dt > 0

        # Should be less than one orbital period
        T = semimajoraxis_to_period(a)
        assert dt < T

    def test_time_since_apogee(self):
        """Test time since apogee calculation."""
        nu = 3*np.pi / 2  # 270 degrees (past apogee)
        a = 7000
        e = 0.1

        dt = time_since_apogee(nu, a, e)

        # Should be positive
        assert dt > 0

        # Should be less than half period
        T = semimajoraxis_to_period(a)
        assert dt < T / 2

    def test_time_at_periapsis(self):
        """Test time calculations at periapsis."""
        nu = 0
        a = 7000
        e = 0.1

        # At periapsis
        dt_since = time_since_perigee(nu, a, e)
        assert np.isclose(dt_since, 0, atol=1e-6)

        # Time to next periapsis should be full period
        dt_to = time_to_perigee(nu, a, e)
        T = semimajoraxis_to_period(a)
        assert np.isclose(dt_to, T, rtol=1e-3)

    def test_time_at_apoapsis(self):
        """Test time calculations at apoapsis."""
        nu = np.pi
        a = 7000
        e = 0.1

        # At apoapsis, time to next apoapsis is full period
        dt_to = time_to_apogee(nu, a, e)
        T = semimajoraxis_to_period(a)
        assert np.isclose(dt_to, T, rtol=1e-3)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_circular_orbit_anomalies(self):
        """Test anomaly conversions for circular orbit (e=0)."""
        e = 0.0
        angles = np.array([0, np.pi/4, np.pi/2, np.pi])

        for angle in angles:
            # For circular orbit, all anomalies are equal
            E = mean_to_eccentric_anomaly(angle, e)
            assert np.isclose(E, angle)

            nu = eccentric_to_true_anomaly(angle, e)
            assert np.isclose(nu, angle, atol=1e-10)

    def test_very_small_eccentricity(self):
        """Test with very small but non-zero eccentricity."""
        e = 1e-10
        M = np.pi / 3

        E = mean_to_eccentric_anomaly(M, e)
        nu = eccentric_to_true_anomaly(E, e)

        # Should be nearly equal to M
        assert np.isclose(E, M, atol=1e-8)
        assert np.isclose(nu, M, atol=1e-8)

    def test_near_parabolic_orbit(self):
        """Test with eccentricity very close to 1."""
        e = 0.9999
        M = np.pi / 6

        # Should still converge
        E = mean_to_eccentric_anomaly(M, e)
        assert 0 <= E <= 2*np.pi

        # Verify
        M_back = eccentric_to_mean_anomaly(E, e)
        assert np.isclose(M, M_back, atol=1e-8)

    def test_full_orbit_sweep(self):
        """Test anomaly conversions across full orbit."""
        e = 0.3
        M_values = np.linspace(0, 2*np.pi, 100)

        for M in M_values:
            E = mean_to_eccentric_anomaly(M, e)
            nu = eccentric_to_true_anomaly(E, e)

            # All should be valid angles
            assert 0 <= E <= 2*np.pi
            assert 0 <= nu <= 2*np.pi

            # Verify roundtrip
            M_back = true_to_mean_anomaly(nu, e)
            assert np.isclose(M, M_back, atol=1e-9)


class TestTimeToApseVectorization:
    """Tests for vectorized time-to-apse calculations."""

    def test_time_to_perigee_vectorized(self):
        """Test vectorized time_to_perigee calculation."""
        # Multiple orbits with different true anomalies
        nu = np.array([0, np.pi/4, np.pi/2, np.pi])
        a = 7000
        e = 0.1

        dt = time_to_perigee(nu, a, e)

        # Should return array of same shape
        assert dt.shape == nu.shape

        # All should be positive
        assert np.all(dt > 0)

        # At periapsis (nu=0), should return full period
        T = semimajoraxis_to_period(a)
        assert np.isclose(dt[0], T, rtol=1e-3)

    def test_time_since_perigee_vectorized(self):
        """Test vectorized time_since_perigee calculation."""
        nu = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        a = 7000
        e = 0.1

        dt = time_since_perigee(nu, a, e)

        # Should return array of same shape
        assert dt.shape == nu.shape

        # All should be non-negative
        assert np.all(dt >= 0)

        # At periapsis (nu=0), should be zero
        assert np.isclose(dt[0], 0, atol=1e-6)

    def test_time_to_apogee_vectorized(self):
        """Test vectorized time_to_apogee calculation."""
        nu = np.array([0, np.pi/4, np.pi/2, np.pi])
        a = 7000
        e = 0.1

        dt = time_to_apogee(nu, a, e)

        # Should return array of same shape
        assert dt.shape == nu.shape

        # All should be positive
        assert np.all(dt > 0)

        # At apogee (nu=π), should return full period
        T = semimajoraxis_to_period(a)
        assert np.isclose(dt[3], T, rtol=1e-3)

    def test_time_since_apogee_vectorized(self):
        """Test vectorized time_since_apogee calculation."""
        nu = np.array([np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        a = 7000
        e = 0.1

        dt = time_since_apogee(nu, a, e)

        # Should return array of same shape
        assert dt.shape == nu.shape

        # All should be non-negative
        assert np.all(dt >= 0)

    def test_time_to_apse_batch_orbits(self):
        """Test time-to-apse with batch of different orbits."""
        # Different orbits with different parameters
        nu = np.array([np.pi/4, np.pi/2, 3*np.pi/4])
        a = np.array([7000, 8000, 9000])
        e = np.array([0.1, 0.2, 0.05])

        # All functions should handle broadcasting
        dt_to_peri = time_to_perigee(nu, a, e)
        dt_since_peri = time_since_perigee(nu, a, e)
        dt_to_apo = time_to_apogee(nu, a, e)
        dt_since_apo = time_since_apogee(nu, a, e)

        # All should return arrays of same shape
        assert dt_to_peri.shape == nu.shape
        assert dt_since_peri.shape == nu.shape
        assert dt_to_apo.shape == nu.shape
        assert dt_since_apo.shape == nu.shape

        # All should be positive/non-negative
        assert np.all(dt_to_peri > 0)
        assert np.all(dt_since_peri >= 0)
        assert np.all(dt_to_apo > 0)
        assert np.all(dt_since_apo >= 0)


class TestQuadrantHandling:
    """Tests for proper quadrant handling in anomaly conversions."""

    def test_all_quadrants_true_to_eccentric(self):
        """Test true to eccentric conversion in all quadrants."""
        e = 0.2

        # First quadrant
        nu_q1 = np.pi / 4
        E_q1 = true_to_eccentric_anomaly(nu_q1, e)
        assert 0 < E_q1 < np.pi / 2

        # Second quadrant
        nu_q2 = 3*np.pi / 4
        E_q2 = true_to_eccentric_anomaly(nu_q2, e)
        assert np.pi / 2 < E_q2 < np.pi

        # Third quadrant
        nu_q3 = 5*np.pi / 4
        E_q3 = true_to_eccentric_anomaly(nu_q3, e)
        assert np.pi < E_q3 < 3*np.pi / 2

        # Fourth quadrant
        nu_q4 = 7*np.pi / 4
        E_q4 = true_to_eccentric_anomaly(nu_q4, e)
        assert 3*np.pi / 2 < E_q4 < 2*np.pi

    def test_all_quadrants_eccentric_to_true(self):
        """Test eccentric to true conversion in all quadrants."""
        e = 0.2

        # Test all four quadrants
        E_values = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]

        for E in E_values:
            nu = eccentric_to_true_anomaly(E, e)
            assert 0 <= nu <= 2*np.pi

            # Verify roundtrip
            E_back = true_to_eccentric_anomaly(nu, e)
            assert np.isclose(E, E_back, atol=1e-10)
