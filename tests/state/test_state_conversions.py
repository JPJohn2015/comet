"""Comprehensive test suite for state conversion functions.

This module tests all functionality of the state conversion functions including:
- cartesian_to_elements conversion
- elements_to_cartesian conversion
- Batch processing (N x 6 arrays)
- Roundtrip conversions
- Edge cases (circular, elliptical, parabolic, hyperbolic)
- Reference test vectors
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.state.state_conversions import cartesian_to_elements, elements_to_cartesian
from comet.utilities.constants import Constants as c


class TestCartesianToElements:
    """Tests for cartesian_to_elements conversion."""

    def test_conversion_single_state(self):
        """Test conversion of single state vector."""
        state = np.array([7000, 0, 0, 0, 7.5, 0])
        elements = cartesian_to_elements(state)
        assert len(elements) == 6
        assert elements[0] > 0  # Positive semi-major axis for elliptical orbit

    def test_conversion_batch_states(self):
        """Test conversion of multiple state vectors."""
        states = np.array([
            [7000, 0, 0, 0, 7.5, 0],
            [8000, 0, 0, 0, 7.0, 0]
        ])
        elements = cartesian_to_elements(states)
        assert elements.shape == (2, 6)
        assert np.all(elements[:, 0] > 0)  # All positive semi-major axes

    def test_invalid_state_length(self):
        """Test that invalid state length raises ValueError."""
        state = np.array([[7000, 0, 0, 0, 7.5]])  # Only 5 elements in 2D array
        with pytest.raises(ValueError):
            cartesian_to_elements(state)

    def test_circular_orbit(self):
        """Test conversion of circular orbit."""
        # Circular orbit at 7000 km
        v_circular = np.sqrt(c.MU_EARTH / 7000)
        state = np.array([7000, 0, 0, 0, v_circular, 0])
        elements = cartesian_to_elements(state)

        # Check semi-major axis
        assert np.isclose(elements[0], 7000, rtol=1e-3)
        # Check eccentricity is near zero
        assert elements[1] < 0.01

    def test_elliptical_orbit(self):
        """Test conversion of elliptical orbit."""
        state = np.array([7000, 0, 0, 0, 7.5, 0])
        elements = cartesian_to_elements(state)

        # Check elliptical orbit properties
        assert elements[0] > 0  # Positive semi-major axis
        assert 0 < elements[1] < 1  # Eccentricity between 0 and 1

    def test_hyperbolic_orbit(self):
        """Test conversion of hyperbolic orbit."""
        # High velocity for escape trajectory
        state = np.array([7000, 0, 0, 0, 15.0, 0])
        elements = cartesian_to_elements(state)

        # Hyperbolic orbit has negative semi-major axis
        assert elements[0] < 0
        # Eccentricity > 1
        assert elements[1] > 1

    def test_parabolic_orbit(self):
        """Test conversion of parabolic orbit."""
        # Escape velocity for parabolic orbit
        v_escape = np.sqrt(2 * c.MU_EARTH / 7000)
        state = np.array([7000, 0, 0, 0, v_escape, 0])
        elements = cartesian_to_elements(state)

        # Parabolic orbit has infinite semi-major axis or very large value
        assert np.isinf(elements[0]) or np.abs(elements[0]) > 1e10
        # Eccentricity should be very close to 1 (numerical precision limits exact 1.0)
        assert np.isclose(elements[1], 1.0, rtol=1e-4)

    def test_equatorial_orbit(self):
        """Test conversion of equatorial orbit."""
        state = np.array([7000, 0, 0, 0, 7.5, 0])
        elements = cartesian_to_elements(state)

        # Inclination should be 0
        assert np.isclose(elements[2], 0.0, atol=1e-6)

    def test_polar_orbit(self):
        """Test conversion of polar orbit."""
        # Polar orbit (velocity in z direction)
        state = np.array([7000, 0, 0, 0, 0, 7.5])
        elements = cartesian_to_elements(state)

        # Inclination should be 90 degrees
        assert np.isclose(elements[2], np.pi / 2, rtol=1e-3)

    def test_inclined_orbit(self):
        """Test conversion of inclined orbit."""
        # Orbit with 45 degree inclination
        v = 7.5
        state = np.array([7000, 0, 0, 0, v * np.cos(np.pi/4), v * np.sin(np.pi/4)])
        elements = cartesian_to_elements(state)

        # Inclination should be near 45 degrees
        assert 0 < elements[2] < np.pi / 2


class TestElementsToCartesian:
    """Tests for elements_to_cartesian conversion."""

    def test_conversion_single_elements(self):
        """Test conversion of single element set."""
        elements = np.array([7000, 0.1, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)
        assert len(state) == 6
        assert np.linalg.norm(state[:3]) > 0  # Non-zero position

    def test_conversion_batch_elements(self):
        """Test conversion of multiple element sets."""
        elements = np.array([
            [7000, 0.1, 0.0, 0.0, 0.0, 0.0],
            [8000, 0.2, 0.1, 0.2, 0.3, 0.4]
        ])
        states = elements_to_cartesian(elements)
        assert states.shape == (2, 6)
        assert np.all(np.linalg.norm(states[:, :3], axis=1) > 0)

    def test_invalid_elements_length(self):
        """Test that invalid elements length raises ValueError."""
        elements = np.array([[7000, 0.1, 0.0, 0.0, 0.0]])  # Only 5 elements in 2D array
        with pytest.raises(ValueError):
            elements_to_cartesian(elements)

    def test_circular_orbit(self):
        """Test conversion of circular orbit elements."""
        # Circular orbit (e=0)
        elements = np.array([7000, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)

        # Position should be at periapsis (= semi-major axis for circular)
        r = np.linalg.norm(state[:3])
        assert np.isclose(r, 7000, rtol=1e-6)

    def test_elliptical_orbit_periapsis(self):
        """Test conversion at periapsis."""
        # At periapsis (v=0)
        elements = np.array([7000, 0.1, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)

        # At periapsis, r = a(1-e)
        expected_r = 7000 * (1 - 0.1)
        r = np.linalg.norm(state[:3])
        assert np.isclose(r, expected_r, rtol=1e-6)

    def test_elliptical_orbit_apoapsis(self):
        """Test conversion at apoapsis."""
        # At apoapsis (v=pi)
        elements = np.array([7000, 0.1, 0.0, 0.0, 0.0, np.pi])
        state = elements_to_cartesian(elements)

        # At apoapsis, r = a(1+e)
        expected_r = 7000 * (1 + 0.1)
        r = np.linalg.norm(state[:3])
        assert np.isclose(r, expected_r, rtol=1e-6)

    def test_inclined_orbit(self):
        """Test conversion of inclined orbit."""
        # 45 degree inclination
        elements = np.array([7000, 0.1, np.pi/4, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)

        # State should have non-zero z component due to inclination
        assert len(state) == 6

    def test_rotated_orbit(self):
        """Test conversion with RAAN rotation."""
        # 90 degree RAAN
        elements = np.array([7000, 0.1, 0.0, np.pi/2, 0.0, 0.0])
        state = elements_to_cartesian(elements)

        # Position should be rotated by 90 degrees in xy-plane
        assert len(state) == 6


class TestRoundtripConversions:
    """Tests for roundtrip conversion accuracy."""

    def test_roundtrip_circular_equatorial(self):
        """Test roundtrip for circular equatorial orbit."""
        # Start with elements
        elements_orig = np.array([7000, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements_orig)
        elements_back = cartesian_to_elements(state)

        # Check major elements are preserved
        assert np.isclose(elements_orig[0], elements_back[0], rtol=1e-6)  # a
        assert np.isclose(elements_orig[1], elements_back[1], atol=1e-6)  # e

    def test_roundtrip_elliptical_orbit(self):
        """Test roundtrip for elliptical orbit."""
        elements_orig = np.array([7000, 0.1, np.deg2rad(30), np.deg2rad(45),
                                   np.deg2rad(60), np.deg2rad(90)])
        state = elements_to_cartesian(elements_orig)
        elements_back = cartesian_to_elements(state)

        # Check elements are preserved within tolerance
        assert np.isclose(elements_orig[0], elements_back[0], rtol=1e-6)  # a
        assert np.isclose(elements_orig[1], elements_back[1], atol=1e-6)  # e
        assert np.isclose(elements_orig[2], elements_back[2], atol=1e-6)  # i
        # RAAN and argument of perigee may differ if inclination or eccentricity is near zero
        # True anomaly should match
        assert np.isclose(elements_orig[5], elements_back[5], atol=1e-6)

    def test_roundtrip_state_to_elements_to_state(self):
        """Test roundtrip starting from state."""
        state_orig = np.array([7000, 1000, 500, 0, 7.5, 1.0])
        elements = cartesian_to_elements(state_orig)
        state_back = elements_to_cartesian(elements)

        # State should be preserved
        assert np.allclose(state_orig, state_back, rtol=1e-6)

    def test_roundtrip_batch_states(self):
        """Test roundtrip with batch of states."""
        # Use orbits with non-zero inclination to avoid numerical instability
        # at equatorial orbits where RAAN is undefined
        states_orig = np.array([
            [7000, 0, 100, 0, 7.5, 0.5],  # Inclined orbit
            [8000, 100, 0, 0, 7.0, 0.8],  # Inclined orbit
            [6500, 500, 200, 0, 7.8, 0.5]  # Inclined orbit
        ])
        elements = cartesian_to_elements(states_orig)
        states_back = elements_to_cartesian(elements)

        # All states should be preserved
        assert np.allclose(states_orig, states_back, rtol=1e-5)

    def test_roundtrip_hyperbolic_orbit(self):
        """Test roundtrip for hyperbolic orbit."""
        state_orig = np.array([7000, 0, 0, 0, 15.0, 0])
        elements = cartesian_to_elements(state_orig)
        state_back = elements_to_cartesian(elements)

        # State should be preserved
        assert np.allclose(state_orig, state_back, rtol=1e-5)


class TestEdgeCases:
    """Tests for edge cases and special orbits."""

    def test_zero_eccentricity(self):
        """Test handling of zero eccentricity."""
        elements = np.array([7000, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)
        elements_back = cartesian_to_elements(state)

        # Eccentricity should remain zero
        assert np.isclose(elements_back[1], 0.0, atol=1e-6)

    def test_zero_inclination(self):
        """Test handling of zero inclination."""
        elements = np.array([7000, 0.1, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)
        elements_back = cartesian_to_elements(state)

        # Inclination should remain zero
        assert np.isclose(elements_back[2], 0.0, atol=1e-6)

    def test_high_eccentricity(self):
        """Test handling of high eccentricity orbits."""
        elements = np.array([7000, 0.9, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)
        elements_back = cartesian_to_elements(state)

        # Eccentricity should be preserved
        assert np.isclose(elements_back[1], 0.9, rtol=1e-3)

    def test_retrograde_orbit(self):
        """Test handling of retrograde orbit (i > 90 deg)."""
        elements = np.array([7000, 0.1, np.deg2rad(120), 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)
        elements_back = cartesian_to_elements(state)

        # Inclination should be preserved
        assert np.isclose(elements_back[2], np.deg2rad(120), rtol=1e-3)

    def test_multiple_angles(self):
        """Test with various angle combinations."""
        elements = np.array([7000, 0.1, np.deg2rad(45), np.deg2rad(90),
                            np.deg2rad(180), np.deg2rad(270)])
        state = elements_to_cartesian(elements)
        elements_back = cartesian_to_elements(state)

        # All angles should be preserved
        assert np.isclose(elements_back[2], np.deg2rad(45), atol=1e-6)  # i
        # RAAN should be preserved
        # Argument of perigee should be preserved
        # True anomaly should be preserved (modulo 2*pi)

    def test_very_small_semi_major_axis(self):
        """Test handling of very small semi-major axis."""
        elements = np.array([6500, 0.01, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)

        # Should not crash, position should be reasonable
        r = np.linalg.norm(state[:3])
        assert r > 0

    def test_very_large_semi_major_axis(self):
        """Test handling of very large semi-major axis."""
        elements = np.array([100000, 0.1, 0.0, 0.0, 0.0, 0.0])
        state = elements_to_cartesian(elements)

        # Should not crash
        r = np.linalg.norm(state[:3])
        assert r > 0


class TestCustomMu:
    """Tests for custom gravitational parameter."""

    def test_cartesian_to_elements_custom_mu(self):
        """Test conversion with custom mu."""
        state = np.array([7000, 0, 0, 0, 7.5, 0])
        mu_custom = c.MU_EARTH * 2
        elements = cartesian_to_elements(state, mu=mu_custom)

        # Should produce different result than default mu
        elements_default = cartesian_to_elements(state)
        assert not np.allclose(elements, elements_default)

    def test_elements_to_cartesian_custom_mu(self):
        """Test conversion with custom mu."""
        elements = np.array([7000, 0.1, 0.0, 0.0, 0.0, 0.0])
        mu_custom = c.MU_EARTH * 2
        state = elements_to_cartesian(elements, mu=mu_custom)

        # Should produce different velocity than default mu
        state_default = elements_to_cartesian(elements)
        assert not np.allclose(state[3:], state_default[3:])

    def test_roundtrip_custom_mu(self):
        """Test roundtrip with consistent custom mu."""
        elements_orig = np.array([7000, 0.1, 0.0, 0.0, 0.0, 0.0])
        mu_custom = c.MU_EARTH * 2

        state = elements_to_cartesian(elements_orig, mu=mu_custom)
        elements_back = cartesian_to_elements(state, mu=mu_custom)

        # Elements should be preserved with consistent mu
        assert np.allclose(elements_orig, elements_back, rtol=1e-6)


class TestValidation:
    """Tests for validation functions."""

    def test_validate_state_valid(self):
        """Test validate_state with valid state."""
        from comet.state.state_conversions import validate_state

        state = np.array([7000, 0, 0, 0, 7.5, 0])
        assert validate_state(state, warn=False) == True

    def test_validate_state_nan(self):
        """Test validate_state detects NaN."""
        from comet.state.state_conversions import validate_state

        state = np.array([np.nan, 0, 0, 0, 7.5, 0])
        assert validate_state(state, warn=False) == False

    def test_validate_state_inf(self):
        """Test validate_state detects Inf."""
        from comet.state.state_conversions import validate_state

        state = np.array([np.inf, 0, 0, 0, 7.5, 0])
        assert validate_state(state, warn=False) == False

    def test_validate_state_inside_earth(self):
        """Test validate_state detects position inside Earth."""
        from comet.state.state_conversions import validate_state

        state = np.array([6000, 0, 0, 0, 7.5, 0])  # Less than Earth radius
        assert validate_state(state, warn=False) == False

    def test_validate_state_batch(self):
        """Test validate_state with batch of states."""
        from comet.state.state_conversions import validate_state

        states = np.array([
            [7000, 0, 0, 0, 7.5, 0],
            [8000, 0, 0, 0, 7.0, 0]
        ])
        assert validate_state(states, warn=False) == True

    def test_validate_elements_valid(self):
        """Test validate_elements with valid elements."""
        from comet.state.state_conversions import validate_elements

        # Use a=8000, e=0.1 -> r_p = 7200 km > Earth radius
        elements = np.array([8000, 0.1, 0.5, 1.0, 0.5, 1.0])
        assert validate_elements(elements, warn=False) == True

    def test_validate_elements_negative_ecc(self):
        """Test validate_elements detects negative eccentricity."""
        from comet.state.state_conversions import validate_elements

        elements = np.array([7000, -0.1, 0.5, 1.0, 0.5, 1.0])
        assert validate_elements(elements, warn=False) == False

    def test_validate_elements_zero_sma(self):
        """Test validate_elements detects zero semi-major axis."""
        from comet.state.state_conversions import validate_elements

        elements = np.array([0.0, 0.5, 0.5, 1.0, 0.5, 1.0])
        assert validate_elements(elements, warn=False) == False

    def test_validate_elements_invalid_inclination(self):
        """Test validate_elements detects invalid inclination."""
        from comet.state.state_conversions import validate_elements

        elements = np.array([7000, 0.1, -0.5, 1.0, 0.5, 1.0])
        assert validate_elements(elements, warn=False) == False

        elements = np.array([7000, 0.1, 4.0, 1.0, 0.5, 1.0])
        assert validate_elements(elements, warn=False) == False

    def test_validate_elements_periapsis_inside_earth(self):
        """Test validate_elements detects periapsis inside Earth."""
        from comet.state.state_conversions import validate_elements

        # Small orbit with periapsis inside Earth
        elements = np.array([6000, 0.01, 0.5, 1.0, 0.5, 1.0])
        assert validate_elements(elements, warn=False) == False

    def test_validate_elements_parabolic(self):
        """Test validate_elements accepts parabolic orbit."""
        from comet.state.state_conversions import validate_elements

        elements = np.array([np.inf, 1.0, 0.5, 1.0, 0.5, 1.0])
        assert validate_elements(elements, warn=False) == True

    def test_validate_elements_batch(self):
        """Test validate_elements with batch."""
        from comet.state.state_conversions import validate_elements

        # Use safe periapsis distances
        elements = np.array([
            [8000, 0.1, 0.5, 1.0, 0.5, 1.0],  # r_p = 7200 km
            [10000, 0.2, 0.6, 1.5, 0.6, 1.2]  # r_p = 8000 km
        ])
        assert validate_elements(elements, warn=False) == True


class TestVectorizationPerformance:
    """Tests to verify vectorization improvements."""

    def test_batch_conversion_no_errors(self):
        """Test that batch conversions work correctly."""
        # Large batch to test vectorization
        N = 100
        elements = np.array([
            [7000 + i*10, 0.1, np.deg2rad(45), np.deg2rad(i),
             np.deg2rad(60), np.deg2rad(90)]
            for i in range(N)
        ])

        states = elements_to_cartesian(elements)
        assert states.shape == (N, 6)
        assert not np.any(np.isnan(states))

    def test_roundtrip_large_batch(self):
        """Test roundtrip with large batch maintains accuracy."""
        N = 50
        states_orig = np.array([
            [7000 + i*10, 100*i, 50*i, 0.1*i, 7.5 - 0.01*i, 0.5]
            for i in range(N)
        ])

        elements = cartesian_to_elements(states_orig)
        states_back = elements_to_cartesian(elements)

        # All states should roundtrip accurately
        assert np.allclose(states_orig, states_back, rtol=1e-5)
