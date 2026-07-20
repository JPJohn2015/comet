"""Unit tests for access computation functions.

Tests all vectorized access functions against known geometric configurations:
- Equatorial line-of-sight scenarios
- Straight-up elevation cases
- Colinear phase angles
- Canonical range/rate geometries

Each test validates physical correctness and proper broadcasting behavior.
"""

# python imports
import pytest
import numpy as np

# COMET imports
from comet.access import functions
from comet.utilities.constants import Constants as c


class TestMatchDims:
    """Test the match_dims broadcasting helper."""

    def test_match_dims_3d_arrays(self):
        """Test broadcasting (N, T, 3) and (K, T, 3) arrays."""
        source = np.random.randn(3, 10, 3)  # 3 sources, 10 times
        target = np.random.randn(5, 10, 3)  # 5 targets, 10 times

        src_bc, tgt_bc = functions.match_dims(source, target)

        assert src_bc.shape == (3, 1, 10, 3)  # Ready for (3, 5, 10, 3)
        assert tgt_bc.shape == (1, 5, 10, 3)  # Ready for (3, 5, 10, 3)

    def test_match_dims_2d_arrays(self):
        """Test broadcasting when one input is (T, 3)."""
        source = np.random.randn(10, 3)      # Single source, 10 times
        target = np.random.randn(5, 10, 3)   # 5 targets, 10 times

        src_bc, tgt_bc = functions.match_dims(source, target)

        assert src_bc.shape == (1, 1, 10, 3)
        assert tgt_bc.shape == (1, 5, 10, 3)

    def test_match_dims_broadcasts_correctly(self):
        """Test that broadcasted arrays multiply correctly."""
        source = np.ones((2, 5, 3))
        target = np.ones((3, 5, 3)) * 2

        src_bc, tgt_bc = functions.match_dims(source, target)

        # Broadcasting should allow element-wise operations
        result = src_bc + tgt_bc  # Should broadcast to (2, 3, 5, 3)
        assert result.shape == (2, 3, 5, 3)
        assert np.allclose(result, 3.0)


class TestLineOfSight:
    """Test Earth line-of-sight computations."""

    def test_los_clear_space(self):
        """Test LOS when both objects are far from Earth with clear path."""
        # Source and target in GEO, clear LOS
        source = np.array([[[42000, 0, 0]]])      # (1, 1, 3)
        target = np.array([[[0, 42000, 0]]])      # (1, 1, 3)

        los = functions.compute_los_access(source, target)

        assert los.shape == (1, 1, 1)
        assert los[0, 0, 0] == 1.0  # Clear LOS

    def test_los_earth_obstruction(self):
        """Test LOS when Earth blocks the path."""
        # Source and target on opposite sides of Earth at LEO altitude
        alt = c.A_EARTH / 1000.0 + 500  # 500 km altitude
        source = np.array([[[alt, 0, 0]]])       # (1, 1, 3)
        target = np.array([[[-alt, 0, 0]]])      # (1, 1, 3)

        los = functions.compute_los_access(source, target)

        assert los.shape == (1, 1, 1)
        assert los[0, 0, 0] == 0.0  # Earth blocks

    def test_los_tangent_case(self):
        """Test LOS for satellites that can see each other."""
        # Source at GEO, target at GEO on same belt (small angle)
        # At GEO altitude, much wider angles have clear LOS
        geo_alt = 42000  # GEO altitude in km
        source = np.array([[[geo_alt, 0, 0]]])

        # Target at 30 degrees away (not 90 - that's blocked even at GEO!)
        angle = np.deg2rad(30)
        target = np.array([[[geo_alt * np.cos(angle), geo_alt * np.sin(angle), 0]]])

        los = functions.compute_los_access(source, target)

        # Should have LOS at GEO with small angle
        assert los[0, 0, 0] == 1.0

    def test_los_multiple_sources_targets(self):
        """Test LOS with multiple sources and targets."""
        # 2 sources, 3 targets, 5 time steps
        sources = np.random.randn(2, 5, 3) * 20000 + 30000  # Around GEO
        targets = np.random.randn(3, 5, 3) * 20000 + 30000

        los = functions.compute_los_access(sources, targets)

        assert los.shape == (2, 3, 5)
        assert np.all((los == 0.0) | (los == 1.0))  # All boolean


class TestLunarLineOfSight:
    """Test lunar line-of-sight computations."""

    def test_lunar_los_clear(self):
        """Test lunar LOS when Moon is far from path."""
        source = np.array([[[c.A_EARTH / 1000.0 + 500, 0, 0]]])
        target = np.array([[[c.A_EARTH / 1000.0 + 600, 0, 0]]])
        moon = np.array([[384400e3, 0, 0]])  # Moon far away

        los = functions.compute_los_access_lunar(source, target, moon)

        assert los.shape == (1, 1, 1)
        assert los[0, 0, 0] == 1.0  # Clear LOS

    def test_lunar_los_obstruction(self):
        """Test lunar LOS when Moon blocks path."""
        # Source and target on opposite sides of Moon
        moon = np.array([[0, 0, 0]])  # Moon at origin
        source = np.array([[[-c.RADIUS_MOON - 100, 0, 0]]])
        target = np.array([[[c.RADIUS_MOON + 100, 0, 0]]])

        los = functions.compute_los_access_lunar(source, target, moon)

        assert los[0, 0, 0] == 0.0  # Moon blocks


class TestSolarPhaseAngle:
    """Test solar phase angle computations."""

    def test_spa_colinear_full_illumination(self):
        """Test SPA when source looks at fully illuminated target (0 degrees)."""
        # Sun, target, source in line (source sees fully lit side)
        # Source on same side as Sun → fully illuminated → 0°
        sun = np.array([[[1e8, 0, 0]]])
        target = np.array([[[0, 0, 0]]])
        source = np.array([[[10000, 0, 0]]])

        spa = functions.compute_solar_phase_ang_deg(source, target, sun)

        assert spa.shape == (1, 1, 1)
        assert np.isclose(spa[0, 0, 0], 0.0, atol=0.1)

    def test_spa_colinear_dark_side(self):
        """Test SPA when source looks at dark side (180 degrees)."""
        # Source, target, sun in line (source sees dark side)
        # Source on opposite side from Sun → dark side → 180°
        sun = np.array([[[1e8, 0, 0]]])
        target = np.array([[[0, 0, 0]]])
        source = np.array([[[-10000, 0, 0]]])

        spa = functions.compute_solar_phase_ang_deg(source, target, sun)

        assert np.isclose(spa[0, 0, 0], 180.0, atol=0.1)

    def test_spa_perpendicular(self):
        """Test SPA when source is perpendicular to sun line (90 degrees)."""
        sun = np.array([[[1e8, 0, 0]]])
        target = np.array([[[0, 0, 0]]])
        source = np.array([[[0, 10000, 0]]])

        spa = functions.compute_solar_phase_ang_deg(source, target, sun)

        assert np.isclose(spa[0, 0, 0], 90.0, atol=0.1)

    def test_spa_multiple_times(self):
        """Test SPA computation over multiple time steps."""
        sun = np.array([[1e8, 0, 0]] * 10)  # (10, 3)
        target = np.array([[[0, 0, 0] for _ in range(10)]])  # (1, 10, 3)

        # Source moves from +x to -x (SPA from 0 to 180)
        source_x = np.linspace(10000, -10000, 10)
        source = np.array([[[x, 0, 0] for x in source_x]])  # (1, 10, 3)

        spa = functions.compute_solar_phase_ang_deg(source, target, sun)

        assert spa.shape == (1, 1, 10)
        # First position (source at +x, sun at +x): ~0 deg, last (source at -x): ~180 deg
        assert spa[0, 0, 0] < 10
        assert spa[0, 0, -1] > 170


class TestExclusionAngle:
    """Test exclusion angle computations."""

    def test_exclusion_colinear(self):
        """Test exclusion angle when target and body are colinear (0 degrees)."""
        source = np.array([[[0, 0, 0]]])
        target = np.array([[[10000, 0, 0]]])
        body = np.array([[20000, 0, 0]])  # Sun in same direction

        excl = functions.compute_exclusion_angle_deg(source, target, body)

        assert np.isclose(excl[0, 0, 0], 0.0, atol=0.1)

    def test_exclusion_perpendicular(self):
        """Test exclusion angle when target and body are perpendicular (90 degrees)."""
        source = np.array([[[0, 0, 0]]])
        target = np.array([[[10000, 0, 0]]])
        body = np.array([[0, 20000, 0]])  # Sun perpendicular

        excl = functions.compute_exclusion_angle_deg(source, target, body)

        assert np.isclose(excl[0, 0, 0], 90.0, atol=0.1)

    def test_exclusion_opposite(self):
        """Test exclusion angle when target and body are opposite (180 degrees)."""
        source = np.array([[[0, 0, 0]]])
        target = np.array([[[10000, 0, 0]]])
        body = np.array([[-20000, 0, 0]])  # Sun opposite direction

        excl = functions.compute_exclusion_angle_deg(source, target, body)

        assert np.isclose(excl[0, 0, 0], 180.0, atol=0.1)


class TestEarthLimbAngle:
    """Test Earth limb angle computations."""

    def test_limb_straight_down(self):
        """Test limb angle when looking straight down at nadir."""
        # Source at altitude, target at Earth center
        alt = c.A_EARTH / 1000.0 + 500
        source = np.array([[[alt, 0, 0]]])
        target = np.array([[[0, 0, 0]]])  # Earth center

        limb = functions.compute_earth_limb_angle_deg(source, target)

        # Looking at nadir (90 deg elevation), limb is at ~60 deg
        # So limb clearance should be ~30 deg above limb
        assert limb[0, 0, 0] > 20  # Significantly above limb

    def test_limb_at_horizon(self):
        """Test limb angle when looking at horizon."""
        # Source at altitude, target positioned at horizon angle
        alt = c.A_EARTH / 1000.0 + 500
        r_source = alt
        earth_radius = c.A_EARTH / 1000.0

        # Horizon is at limb_angle from nadir: arcsin(R_earth / r_source)
        limb_angle_rad = np.arcsin(earth_radius / r_source)

        # Place target at horizon: angle from nadir = limb_angle
        # Source at (r, 0, 0), target at distance d in direction (cos(θ), sin(θ), 0)
        # where θ = limb_angle from nadir
        target_angle = limb_angle_rad
        target_distance = r_source * 2  # Arbitrary distance beyond source
        target_x = r_source - target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)

        source = np.array([[[r_source, 0, 0]]])
        target = np.array([[[target_x, target_y, 0]]])

        limb = functions.compute_earth_limb_angle_deg(source, target)

        # Looking at horizon, clearance should be near 0 (at the limb)
        assert np.abs(limb[0, 0, 0]) < 2  # Near limb

    def test_limb_below_horizon(self):
        """Test limb angle when looking below horizon (obstructed)."""
        # Source at altitude, target beyond horizon (90° from nadir)
        alt = c.A_EARTH / 1000.0 + 500
        source = np.array([[[alt, 0, 0]]])

        # Target perpendicular to nadir direction (90° angle from nadir)
        # This is well beyond horizon (which is at ~68° from nadir)
        # Nadir is toward (-1, 0, 0), so perpendicular is (0, 1, 0)
        target = np.array([[[alt, 1000, 0]]])  # 90° from nadir

        limb = functions.compute_earth_limb_angle_deg(source, target)

        # Looking below limb (beyond horizon), clearance should be negative
        assert limb[0, 0, 0] < 0


class TestRange:
    """Test range computations."""

    def test_range_simple(self):
        """Test range between two points."""
        source = np.array([[[0, 0, 0]]])
        target = np.array([[[3, 4, 0]]])  # 3-4-5 triangle

        range_km = functions.compute_range_km(source, target)

        assert np.isclose(range_km[0, 0, 0], 5.0, atol=1e-6)

    def test_range_same_position(self):
        """Test range when source and target are at same position."""
        source = np.array([[[1000, 2000, 3000]]])
        target = np.array([[[1000, 2000, 3000]]])

        range_km = functions.compute_range_km(source, target)

        assert np.isclose(range_km[0, 0, 0], 0.0, atol=1e-6)

    def test_range_multiple_pairs(self):
        """Test range with multiple source-target pairs."""
        sources = np.array([
            [[0, 0, 0], [0, 0, 0]],  # 1 source, 2 times
        ])
        targets = np.array([
            [[10, 0, 0], [10, 0, 0]],  # Target 1
            [[0, 20, 0], [0, 20, 0]],  # Target 2
        ])

        range_km = functions.compute_range_km(sources, targets)

        assert range_km.shape == (1, 2, 2)  # 1 source, 2 targets, 2 times
        assert np.allclose(range_km[0, 0, :], 10.0)
        assert np.allclose(range_km[0, 1, :], 20.0)


class TestRangeRate:
    """Test range rate computations."""

    def test_range_rate_separating(self):
        """Test range rate when objects are separating."""
        # Source stationary, target moving away at 1 km/s
        source = np.array([[[0, 0, 0, 0, 0, 0]]])  # pos, vel
        target = np.array([[[10, 0, 0, 1, 0, 0]]])  # Moving away in +x

        rr = functions.compute_range_rate(source, target)

        assert np.isclose(rr[0, 0, 0], 1.0, atol=1e-6)  # Separating

    def test_range_rate_approaching(self):
        """Test range rate when objects are approaching."""
        source = np.array([[[0, 0, 0, 0, 0, 0]]])
        target = np.array([[[10, 0, 0, -1, 0, 0]]])  # Moving toward source

        rr = functions.compute_range_rate(source, target)

        assert np.isclose(rr[0, 0, 0], -1.0, atol=1e-6)  # Approaching

    def test_range_rate_tangential(self):
        """Test range rate when relative motion is tangential."""
        source = np.array([[[0, 0, 0, 0, 0, 0]]])
        target = np.array([[[10, 0, 0, 0, 1, 0]]])  # Moving tangentially

        rr = functions.compute_range_rate(source, target)

        assert np.isclose(rr[0, 0, 0], 0.0, atol=1e-6)  # No radial component


class TestAngularRate:
    """Test angular rate computations."""

    def test_angular_rate_stationary(self):
        """Test angular rate when objects are stationary."""
        source = np.array([[[0, 0, 0, 0, 0, 0]]])
        target = np.array([[[10, 0, 0, 0, 0, 0]]])

        ar = functions.compute_angular_rate(source, target)

        assert np.isclose(ar[0, 0, 0], 0.0, atol=1e-6)

    def test_angular_rate_tangential_motion(self):
        """Test angular rate for pure tangential motion."""
        # Target at 10 km, moving tangentially at 1 km/s
        # Angular rate = v_perp / r = 1 / 10 rad/s = 5.73 deg/s
        source = np.array([[[0, 0, 0, 0, 0, 0]]])
        target = np.array([[[10, 0, 0, 0, 1, 0]]])

        ar = functions.compute_angular_rate(source, target)

        expected = np.rad2deg(1.0 / 10.0)  # ~5.73 deg/s
        assert np.isclose(ar[0, 0, 0], expected, atol=0.01)

    def test_angular_rate_radial_motion(self):
        """Test angular rate for pure radial motion (should be zero)."""
        source = np.array([[[0, 0, 0, 0, 0, 0]]])
        target = np.array([[[10, 0, 0, 1, 0, 0]]])  # Moving radially

        ar = functions.compute_angular_rate(source, target)

        assert np.isclose(ar[0, 0, 0], 0.0, atol=1e-6)


class TestAngleFromNadir:
    """Test off-nadir angle computations."""

    def test_nadir_straight_down(self):
        """Test off-nadir angle when looking straight at Earth center."""
        # Source at altitude, target at Earth center
        source = np.array([[[c.A_EARTH / 1000.0 + 500, 0, 0]]])
        target = np.array([[[0, 0, 0]]])

        angle = functions.compute_angle_from_nadir_deg(source, target)

        assert np.isclose(angle[0, 0, 0], 0.0, atol=0.1)  # Looking at nadir

    def test_nadir_horizon(self):
        """Test off-nadir angle when looking perpendicular to nadir."""
        # Source at altitude, target perpendicular to nadir direction
        alt = c.A_EARTH / 1000.0 + 500
        source = np.array([[[alt, 0, 0]]])

        # For LOS perpendicular to nadir (90°), target should be positioned
        # such that (target - source) is perpendicular to the nadir direction
        # Nadir from (alt, 0, 0) points toward origin: direction (-1, 0, 0)
        # Perpendicular: any direction like (0, 1, 0) or (0, 0, 1)
        target = np.array([[[alt, 1000, 0]]])  # LOS in y direction

        angle = functions.compute_angle_from_nadir_deg(source, target)

        # Angle should be 90 degrees (perpendicular to nadir)
        assert np.abs(angle[0, 0, 0] - 90.0) < 0.1

    def test_nadir_zenith(self):
        """Test off-nadir angle when looking away from Earth."""
        # Source at altitude, target further out
        source = np.array([[[c.A_EARTH / 1000.0 + 500, 0, 0]]])
        target = np.array([[[c.A_EARTH / 1000.0 + 1000, 0, 0]]])

        angle = functions.compute_angle_from_nadir_deg(source, target)

        # Looking away from Earth, angle should be ~180 degrees
        assert angle[0, 0, 0] > 170


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_distance_handling(self):
        """Test that zero distances don't cause division by zero."""
        # Same source and target position
        source = np.array([[[1000, 0, 0]]])
        target = np.array([[[1000, 0, 0]]])

        # Should not raise, returns finite values
        range_km = functions.compute_range_km(source, target)
        assert np.isfinite(range_km).all()

        # LOS should still work
        los = functions.compute_los_access(source, target)
        assert np.isfinite(los).all()

    def test_very_large_distances(self):
        """Test with very large distances (beyond Earth orbit)."""
        # Source at GEO, target at Moon distance
        source = np.array([[[42000, 0, 0]]])
        target = np.array([[[384400, 0, 0]]])

        range_km = functions.compute_range_km(source, target)
        assert np.isfinite(range_km).all()
        assert range_km[0, 0, 0] > 300000  # Reasonable value

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        source = np.array([[[-1000, -2000, -3000]]])
        target = np.array([[[1000, 2000, 3000]]])

        range_km = functions.compute_range_km(source, target)
        expected = np.sqrt((2000**2 + 4000**2 + 6000**2))
        assert np.isclose(range_km[0, 0, 0], expected, rtol=1e-5)
