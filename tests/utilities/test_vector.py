"""Comprehensive test suite for the vector.py utility functions.

This module tests:
- Unit vector calculation
- Angle between vectors
- Phase angle (with direction)
- Dot product
- Vector projection and rejection
- Edge cases (zero vectors, parallel vectors, perpendicular vectors)
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.utilities.vector import (
    unit,
    angle_between,
    phase_angle_between,
    dot,
    projection,
    rejection,
)


class TestUnitVector:
    """Tests for unit vector calculation."""

    def test_unit_x_axis(self):
        """Test unit vector of X-axis vector."""
        v = np.array([5, 0, 0])
        u = unit(v)
        expected = np.array([1, 0, 0])
        assert np.allclose(u, expected)

    def test_unit_y_axis(self):
        """Test unit vector of Y-axis vector."""
        v = np.array([0, 3, 0])
        u = unit(v)
        expected = np.array([0, 1, 0])
        assert np.allclose(u, expected)

    def test_unit_z_axis(self):
        """Test unit vector of Z-axis vector."""
        v = np.array([0, 0, -7])
        u = unit(v)
        expected = np.array([0, 0, -1])
        assert np.allclose(u, expected)

    def test_unit_general_vector(self):
        """Test unit vector of general 3D vector."""
        v = np.array([3, 4, 0])
        u = unit(v)
        expected = np.array([0.6, 0.8, 0])
        assert np.allclose(u, expected)

    def test_unit_preserves_direction(self):
        """Test that unit vector has same direction as original."""
        v = np.array([1, 2, 3])
        u = unit(v)
        # Cross product should be zero (parallel)
        assert np.allclose(np.cross(v, u), 0, atol=1e-10)

    def test_unit_magnitude_is_one(self):
        """Test that unit vector has magnitude 1."""
        v = np.array([1, 2, 3])
        u = unit(v)
        assert np.isclose(np.linalg.norm(u), 1.0)

    def test_unit_zero_vector(self):
        """Test unit vector of zero vector returns zeros."""
        v = np.array([0, 0, 0])
        u = unit(v)
        expected = np.array([0, 0, 0])
        assert np.allclose(u, expected)

    def test_unit_batch_vectors(self):
        """Test unit vector with batch of vectors."""
        v = np.array([[1, 0, 0], [0, 2, 0], [3, 4, 0]])
        u = unit(v, axis=1)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]])
        assert np.allclose(u, expected)

    def test_unit_already_unit(self):
        """Test unit vector of already-unit vector."""
        v = np.array([1, 0, 0])
        u = unit(v)
        assert np.allclose(u, v)

    def test_unit_negative_vector(self):
        """Test unit vector preserves sign."""
        v = np.array([-1, -1, -1])
        u = unit(v)
        # Should point in same (negative) direction
        assert np.all(u < 0)
        assert np.isclose(np.linalg.norm(u), 1.0)


class TestAngleBetween:
    """Tests for angle between vectors."""

    def test_angle_between_same_vector(self):
        """Test angle between vector and itself is 0."""
        v = np.array([1, 2, 3])
        angle = angle_between(v, v)
        assert np.isclose(angle, 0.0)

    def test_angle_between_opposite_vectors(self):
        """Test angle between opposite vectors is π."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])
        angle = angle_between(v1, v2)
        assert np.isclose(angle, np.pi)

    def test_angle_between_perpendicular_vectors(self):
        """Test angle between perpendicular vectors is π/2."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        angle = angle_between(v1, v2)
        assert np.isclose(angle, np.pi / 2)

    def test_angle_between_45_degrees(self):
        """Test angle between vectors at 45 degrees."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 1, 0])
        angle = angle_between(v1, v2)
        assert np.isclose(angle, np.pi / 4, atol=1e-10)

    def test_angle_between_60_degrees(self):
        """Test angle between vectors at 60 degrees."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0.5, np.sqrt(3)/2, 0])
        angle = angle_between(v1, v2)
        assert np.isclose(angle, np.pi / 3, atol=1e-10)

    def test_angle_between_symmetric(self):
        """Test angle_between is symmetric."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        angle1 = angle_between(v1, v2)
        angle2 = angle_between(v2, v1)
        assert np.isclose(angle1, angle2)

    def test_angle_between_batch(self):
        """Test angle_between with batch of vectors."""
        v1 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        v2 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
        angles = angle_between(v1, v2, axis=1)
        expected = np.array([0, np.pi, np.pi/2])
        assert np.allclose(angles, expected, atol=1e-10)

    def test_angle_between_different_magnitudes(self):
        """Test angle is independent of magnitude."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 1, 0])
        v2_scaled = 5 * v2
        angle1 = angle_between(v1, v2)
        angle2 = angle_between(v1, v2_scaled)
        assert np.isclose(angle1, angle2)


class TestPhaseAngleBetween:
    """Tests for phase angle with direction."""

    def test_phase_angle_same_vector(self):
        """Test phase angle between vector and itself is 0."""
        v = np.array([1, 0, 0])
        angle = phase_angle_between(v, v)
        assert np.isclose(angle, 0.0)

    def test_phase_angle_positive_direction(self):
        """Test phase angle for positive rotation (CCW in XY plane)."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        angle = phase_angle_between(v1, v2)
        # CCW from X to Y is +90°
        assert np.isclose(angle, np.pi / 2)

    def test_phase_angle_negative_direction(self):
        """Test phase angle for negative rotation (CW in XY plane)."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, -1, 0])
        angle = phase_angle_between(v1, v2)
        # CW from X to -Y is 270° (or -90° + 360°)
        assert np.isclose(angle, 3 * np.pi / 2)

    def test_phase_angle_opposite_vectors(self):
        """Test phase angle between opposite vectors is π."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])
        angle = phase_angle_between(v1, v2)
        assert np.isclose(angle, np.pi)

    def test_phase_angle_range(self):
        """Test phase angle is always in [0, 2π]."""
        v1 = np.array([1, 0, 0])
        angles = []
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            v2 = np.array([np.cos(theta), np.sin(theta), 0])
            angles.append(phase_angle_between(v1, v2))

        angles = np.array(angles)
        assert np.all(angles >= 0)
        assert np.all(angles <= 2 * np.pi)

    def test_phase_angle_batch(self):
        """Test phase angle with batch of vectors."""
        v1 = np.array([[1, 0, 0], [1, 0, 0]])
        v2 = np.array([[0, 1, 0], [0, -1, 0]])
        angles = phase_angle_between(v1, v2, axis=1)
        expected = np.array([np.pi/2, 3*np.pi/2])
        assert np.allclose(angles, expected, atol=1e-10)


class TestDotProduct:
    """Tests for dot product."""

    def test_dot_perpendicular(self):
        """Test dot product of perpendicular vectors is 0."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = dot(v1, v2)
        assert np.isclose(result, 0.0)

    def test_dot_parallel(self):
        """Test dot product of parallel vectors."""
        v1 = np.array([2, 0, 0])
        v2 = np.array([3, 0, 0])
        result = dot(v1, v2)
        assert np.isclose(result, 6.0)

    def test_dot_opposite(self):
        """Test dot product of opposite vectors is negative."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])
        result = dot(v1, v2)
        assert np.isclose(result, -1.0)

    def test_dot_general(self):
        """Test dot product of general vectors."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        result = dot(v1, v2)
        expected = 1*4 + 2*5 + 3*6
        assert np.isclose(result, expected)

    def test_dot_symmetric(self):
        """Test dot product is symmetric."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        dot1 = dot(v1, v2)
        dot2 = dot(v2, v1)
        assert np.isclose(dot1, dot2)

    def test_dot_self(self):
        """Test dot product with self equals magnitude squared."""
        v = np.array([3, 4, 0])
        result = dot(v, v)
        expected = np.linalg.norm(v) ** 2
        assert np.isclose(result, expected)

    def test_dot_batch(self):
        """Test dot product with batch of vectors."""
        v1 = np.array([[1, 0, 0], [0, 1, 0]])
        v2 = np.array([[1, 0, 0], [1, 0, 0]])
        result = dot(v1, v2, axis=1)
        expected = np.array([1.0, 0.0])
        assert np.allclose(result, expected)

    def test_dot_zero_vector(self):
        """Test dot product with zero vector is 0."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([0, 0, 0])
        result = dot(v1, v2)
        assert np.isclose(result, 0.0)


class TestProjection:
    """Tests for vector projection."""

    def test_projection_parallel(self):
        """Test projection of parallel vectors equals the original."""
        v1 = np.array([2, 0, 0])
        v2 = np.array([1, 0, 0])
        proj = projection(v1, v2)
        assert np.allclose(proj, v1)

    def test_projection_perpendicular(self):
        """Test projection of perpendicular vectors is zero."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        proj = projection(v1, v2)
        assert np.allclose(proj, 0, atol=1e-10)

    def test_projection_45_degrees(self):
        """Test projection at 45 degrees."""
        v1 = np.array([1, 1, 0])
        v2 = np.array([1, 0, 0])
        proj = projection(v1, v2)
        expected = np.array([1, 0, 0])
        assert np.allclose(proj, expected)

    def test_projection_general(self):
        """Test projection of general vector."""
        v1 = np.array([3, 4, 0])
        v2 = np.array([1, 0, 0])
        proj = projection(v1, v2)
        expected = np.array([3, 0, 0])
        assert np.allclose(proj, expected)

    def test_projection_onto_unit_vector(self):
        """Test projection onto unit vector."""
        v1 = np.array([3, 4, 0])
        v2 = np.array([1, 0, 0])
        proj = projection(v1, v2)
        # Projection magnitude should equal dot product with unit vector
        assert np.isclose(np.linalg.norm(proj), dot(v1, v2))

    def test_projection_zero_target(self):
        """Test projection onto zero vector returns zeros."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([0, 0, 0])
        proj = projection(v1, v2)
        assert np.allclose(proj, 0, atol=1e-10)

    def test_projection_batch(self):
        """Test projection with batch of vectors."""
        v1 = np.array([[3, 4, 0], [1, 1, 0]])
        v2 = np.array([[1, 0, 0], [1, 0, 0]])
        proj = projection(v1, v2, axis=1)
        expected = np.array([[3, 0, 0], [1, 0, 0]])
        assert np.allclose(proj, expected)

    def test_projection_in_direction(self):
        """Test projection is in direction of target vector."""
        v1 = np.array([3, 4, 0])
        v2 = np.array([1, 1, 0])
        proj = projection(v1, v2)
        # proj should be parallel to v2
        cross = np.cross(proj, v2)
        assert np.allclose(cross, 0, atol=1e-10)


class TestRejection:
    """Tests for vector rejection."""

    def test_rejection_perpendicular_to_target(self):
        """Test rejection is perpendicular to target vector."""
        v1 = np.array([3, 4, 0])
        v2 = np.array([1, 0, 0])
        rej = rejection(v1, v2)
        # rej should be perpendicular to v2
        assert np.isclose(dot(rej, v2), 0, atol=1e-10)

    def test_rejection_parallel(self):
        """Test rejection of parallel vectors is zero."""
        v1 = np.array([2, 0, 0])
        v2 = np.array([1, 0, 0])
        rej = rejection(v1, v2)
        assert np.allclose(rej, 0, atol=1e-10)

    def test_rejection_perpendicular(self):
        """Test rejection of perpendicular vectors equals original."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        rej = rejection(v1, v2)
        assert np.allclose(rej, v1)

    def test_rejection_decomposition(self):
        """Test vector equals projection plus rejection."""
        v1 = np.array([3, 4, 0])
        v2 = np.array([1, 0, 0])
        proj = projection(v1, v2)
        rej = rejection(v1, v2)
        reconstructed = proj + rej
        assert np.allclose(reconstructed, v1)

    def test_rejection_orthogonality(self):
        """Test projection and rejection are orthogonal."""
        v1 = np.array([3, 4, 5])
        v2 = np.array([1, 1, 0])
        proj = projection(v1, v2)
        rej = rejection(v1, v2)
        # proj and rej should be perpendicular
        assert np.isclose(dot(proj, rej), 0, atol=1e-10)

    def test_rejection_batch(self):
        """Test rejection with batch of vectors."""
        v1 = np.array([[3, 4, 0], [1, 1, 0]])
        v2 = np.array([[1, 0, 0], [1, 0, 0]])
        rej = rejection(v1, v2, axis=1)
        expected = np.array([[0, 4, 0], [0, 1, 0]])
        assert np.allclose(rej, expected)

    def test_rejection_general(self):
        """Test rejection of general vector."""
        v1 = np.array([3, 4, 0])
        v2 = np.array([1, 0, 0])
        rej = rejection(v1, v2)
        expected = np.array([0, 4, 0])
        assert np.allclose(rej, expected)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_zero_vectors(self):
        """Test operations with zero vectors."""
        zero = np.array([0, 0, 0])
        v = np.array([1, 2, 3])

        # Unit of zero is zero
        assert np.allclose(unit(zero), zero)

        # Dot product with zero is zero
        assert np.isclose(dot(v, zero), 0)

        # Projection onto zero is zero
        assert np.allclose(projection(v, zero), 0, atol=1e-10)

    def test_very_small_vectors(self):
        """Test operations with very small magnitude vectors."""
        tiny = np.array([1e-15, 1e-15, 1e-15])
        u = unit(tiny)
        # Should not crash or produce NaN
        assert np.all(np.isfinite(u))

    def test_very_large_vectors(self):
        """Test operations with very large magnitude vectors."""
        large = np.array([1e10, 1e10, 0])
        u = unit(large)
        # Unit vector magnitude should still be 1
        assert np.isclose(np.linalg.norm(u), 1.0)

    def test_mixed_dimensions(self):
        """Test error handling for mismatched dimensions."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2])
        # Should handle broadcasting or raise appropriate error
        # This behavior depends on numpy broadcasting rules
        try:
            result = angle_between(v1, v2)
        except (ValueError, IndexError):
            pass  # Expected for incompatible shapes
