"""Comprehensive test suite for the rotations.py utility functions.

This module tests:
- Individual axis rotations (rot1, rot2, rot3)
- Rotation matrix properties (orthogonality, determinant)
- Composed Euler rotations (rot313, rot321)
- Vectorization support
- Edge cases and known values
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.utilities.rotations import rot1, rot2, rot3, rot313, rot321


class TestBasicRotations:
    """Tests for basic rotation matrices."""

    def test_rot1_zero_angle(self):
        """Test rot1 with zero angle returns identity."""
        R = rot1(0.0)
        I = np.eye(3)
        assert np.allclose(R, I)

    def test_rot2_zero_angle(self):
        """Test rot2 with zero angle returns identity."""
        R = rot2(0.0)
        I = np.eye(3)
        assert np.allclose(R, I)

    def test_rot3_zero_angle(self):
        """Test rot3 with zero angle returns identity."""
        R = rot3(0.0)
        I = np.eye(3)
        assert np.allclose(R, I)

    def test_rot1_90_degrees(self):
        """Test rot1 with 90 degree rotation."""
        R = rot1(np.pi / 2)
        # Right-handed rotation about X-axis by 90°: Y → -Z
        v = np.array([0, 1, 0])
        v_rot = R @ v
        expected = np.array([0, 0, -1])
        assert np.allclose(v_rot, expected, atol=1e-10)

    def test_rot2_90_degrees(self):
        """Test rot2 with 90 degree rotation."""
        R = rot2(np.pi / 2)
        # Right-handed rotation about Y-axis by 90°: X → Z
        v = np.array([1, 0, 0])
        v_rot = R @ v
        expected = np.array([0, 0, 1])
        assert np.allclose(v_rot, expected, atol=1e-10)

    def test_rot3_90_degrees(self):
        """Test rot3 with 90 degree rotation."""
        R = rot3(np.pi / 2)
        # Right-handed rotation about Z-axis by 90°: X → -Y
        v = np.array([1, 0, 0])
        v_rot = R @ v
        expected = np.array([0, -1, 0])
        assert np.allclose(v_rot, expected, atol=1e-10)

    def test_rot1_180_degrees(self):
        """Test rot1 with 180 degree rotation."""
        R = rot1(np.pi)
        # Rotation about X-axis by 180°: (x,y,z) → (x,-y,-z)
        v = np.array([1, 1, 1])
        v_rot = R @ v
        expected = np.array([1, -1, -1])
        assert np.allclose(v_rot, expected, atol=1e-10)

    def test_rot2_180_degrees(self):
        """Test rot2 with 180 degree rotation."""
        R = rot2(np.pi)
        # Rotation about Y-axis by 180°: (x,y,z) → (-x,y,-z)
        v = np.array([1, 1, 1])
        v_rot = R @ v
        expected = np.array([-1, 1, -1])
        assert np.allclose(v_rot, expected, atol=1e-10)

    def test_rot3_180_degrees(self):
        """Test rot3 with 180 degree rotation."""
        R = rot3(np.pi)
        # Rotation about Z-axis by 180°: (x,y,z) → (-x,-y,z)
        v = np.array([1, 1, 1])
        v_rot = R @ v
        expected = np.array([-1, -1, 1])
        assert np.allclose(v_rot, expected, atol=1e-10)

    def test_rot1_360_degrees(self):
        """Test rot1 with 360 degree rotation returns identity."""
        R = rot1(2 * np.pi)
        I = np.eye(3)
        assert np.allclose(R, I, atol=1e-10)

    def test_rot2_360_degrees(self):
        """Test rot2 with 360 degree rotation returns identity."""
        R = rot2(2 * np.pi)
        I = np.eye(3)
        assert np.allclose(R, I, atol=1e-10)

    def test_rot3_360_degrees(self):
        """Test rot3 with 360 degree rotation returns identity."""
        R = rot3(2 * np.pi)
        I = np.eye(3)
        assert np.allclose(R, I, atol=1e-10)


class TestRotationMatrixProperties:
    """Tests for mathematical properties of rotation matrices."""

    def test_rot1_orthogonality(self):
        """Test that rot1 produces orthogonal matrix."""
        R = rot1(np.pi / 3)
        # R^T @ R should be identity
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rot2_orthogonality(self):
        """Test that rot2 produces orthogonal matrix."""
        R = rot2(np.pi / 3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rot3_orthogonality(self):
        """Test that rot3 produces orthogonal matrix."""
        R = rot3(np.pi / 3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rot1_determinant(self):
        """Test that rot1 has determinant +1."""
        R = rot1(np.pi / 4)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rot2_determinant(self):
        """Test that rot2 has determinant +1."""
        R = rot2(np.pi / 4)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rot3_determinant(self):
        """Test that rot3 has determinant +1."""
        R = rot3(np.pi / 4)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rot1_preserves_magnitude(self):
        """Test that rot1 preserves vector magnitude."""
        R = rot1(np.pi / 5)
        v = np.array([1, 2, 3])
        v_rot = R @ v
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rot))

    def test_rot2_preserves_magnitude(self):
        """Test that rot2 preserves vector magnitude."""
        R = rot2(np.pi / 5)
        v = np.array([1, 2, 3])
        v_rot = R @ v
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rot))

    def test_rot3_preserves_magnitude(self):
        """Test that rot3 preserves vector magnitude."""
        R = rot3(np.pi / 5)
        v = np.array([1, 2, 3])
        v_rot = R @ v
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rot))

    def test_rot1_inverse_is_transpose(self):
        """Test that R^T = R^(-1) for rot1."""
        R = rot1(np.pi / 6)
        R_inv = np.linalg.inv(R)
        assert np.allclose(R.T, R_inv, atol=1e-10)

    def test_rot2_inverse_is_transpose(self):
        """Test that R^T = R^(-1) for rot2."""
        R = rot2(np.pi / 6)
        R_inv = np.linalg.inv(R)
        assert np.allclose(R.T, R_inv, atol=1e-10)

    def test_rot3_inverse_is_transpose(self):
        """Test that R^T = R^(-1) for rot3."""
        R = rot3(np.pi / 6)
        R_inv = np.linalg.inv(R)
        assert np.allclose(R.T, R_inv, atol=1e-10)


class TestNegativeAngles:
    """Tests for negative angle rotations."""

    def test_rot1_negative_angle(self):
        """Test rot1 with negative angle rotates opposite direction."""
        R_pos = rot1(np.pi / 4)
        R_neg = rot1(-np.pi / 4)
        # R(-θ) should equal R(θ)^T
        assert np.allclose(R_neg, R_pos.T, atol=1e-10)

    def test_rot2_negative_angle(self):
        """Test rot2 with negative angle rotates opposite direction."""
        R_pos = rot2(np.pi / 4)
        R_neg = rot2(-np.pi / 4)
        assert np.allclose(R_neg, R_pos.T, atol=1e-10)

    def test_rot3_negative_angle(self):
        """Test rot3 with negative angle rotates opposite direction."""
        R_pos = rot3(np.pi / 4)
        R_neg = rot3(-np.pi / 4)
        assert np.allclose(R_neg, R_pos.T, atol=1e-10)


class TestVectorization:
    """Tests for vectorized angle inputs."""

    def test_rot1_array_input(self):
        """Test rot1 with array of angles."""
        angles = np.array([0, np.pi/2, np.pi])
        R = rot1(angles)
        assert R.shape == (3, 3, 3)

        # Check first rotation (0°) is identity
        assert np.allclose(R[0], np.eye(3))

        # Check second rotation (90°): Y → -Z
        v = np.array([0, 1, 0])
        v_rot = R[1] @ v
        assert np.allclose(v_rot, np.array([0, 0, -1]), atol=1e-10)

    def test_rot2_array_input(self):
        """Test rot2 with array of angles."""
        angles = np.array([0, np.pi/2, np.pi])
        R = rot2(angles)
        assert R.shape == (3, 3, 3)

        # Check first rotation (0°) is identity
        assert np.allclose(R[0], np.eye(3))

    def test_rot3_array_input(self):
        """Test rot3 with array of angles."""
        angles = np.array([0, np.pi/2, np.pi])
        R = rot3(angles)
        assert R.shape == (3, 3, 3)

        # Check first rotation (0°) is identity
        assert np.allclose(R[0], np.eye(3))

    def test_rot1_scalar_returns_2d(self):
        """Test rot1 with scalar returns 3x3 matrix, not (1,3,3)."""
        R = rot1(np.pi / 4)
        assert R.shape == (3, 3)

    def test_rot2_scalar_returns_2d(self):
        """Test rot2 with scalar returns 3x3 matrix, not (1,3,3)."""
        R = rot2(np.pi / 4)
        assert R.shape == (3, 3)

    def test_rot3_scalar_returns_2d(self):
        """Test rot3 with scalar returns 3x3 matrix, not (1,3,3)."""
        R = rot3(np.pi / 4)
        assert R.shape == (3, 3)


class TestComposedRotations:
    """Tests for composed Euler angle rotations."""

    def test_rot313_identity(self):
        """Test rot313 with zero angles returns identity."""
        R = rot313(0, 0, 0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_rot321_identity(self):
        """Test rot321 with zero angles returns identity."""
        R = rot321(0, 0, 0)
        assert np.allclose(R, np.eye(3), atol=1e-10)

    def test_rot313_composition(self):
        """Test rot313 equals manual composition."""
        angle1, angle2, angle3 = np.pi/6, np.pi/4, np.pi/3
        R = rot313(angle1, angle2, angle3)

        # Manual composition
        R_manual = rot3(angle3) @ rot1(angle2) @ rot3(angle1)

        assert np.allclose(R, R_manual, atol=1e-10)

    def test_rot321_composition(self):
        """Test rot321 equals manual composition."""
        angle1, angle2, angle3 = np.pi/6, np.pi/4, np.pi/3
        R = rot321(angle1, angle2, angle3)

        # Manual composition
        R_manual = rot1(angle3) @ rot2(angle2) @ rot3(angle1)

        assert np.allclose(R, R_manual, atol=1e-10)

    def test_rot313_orthogonality(self):
        """Test rot313 produces orthogonal matrix."""
        R = rot313(np.pi/6, np.pi/4, np.pi/3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rot321_orthogonality(self):
        """Test rot321 produces orthogonal matrix."""
        R = rot321(np.pi/6, np.pi/4, np.pi/3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rot313_determinant(self):
        """Test rot313 has determinant +1."""
        R = rot313(np.pi/6, np.pi/4, np.pi/3)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rot321_determinant(self):
        """Test rot321 has determinant +1."""
        R = rot321(np.pi/6, np.pi/4, np.pi/3)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rot313_preserves_magnitude(self):
        """Test rot313 preserves vector magnitude."""
        R = rot313(np.pi/6, np.pi/4, np.pi/3)
        v = np.array([1, 2, 3])
        v_rot = R @ v
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rot))

    def test_rot321_preserves_magnitude(self):
        """Test rot321 preserves vector magnitude."""
        R = rot321(np.pi/6, np.pi/4, np.pi/3)
        v = np.array([1, 2, 3])
        v_rot = R @ v
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(v_rot))


class TestRoundtrips:
    """Tests for rotation roundtrips."""

    def test_rot1_roundtrip(self):
        """Test rotating forward and back with rot1."""
        angle = np.pi / 3
        R = rot1(angle)
        R_inv = rot1(-angle)

        v = np.array([1, 2, 3])
        v_roundtrip = R_inv @ (R @ v)

        assert np.allclose(v, v_roundtrip, atol=1e-10)

    def test_rot2_roundtrip(self):
        """Test rotating forward and back with rot2."""
        angle = np.pi / 3
        R = rot2(angle)
        R_inv = rot2(-angle)

        v = np.array([1, 2, 3])
        v_roundtrip = R_inv @ (R @ v)

        assert np.allclose(v, v_roundtrip, atol=1e-10)

    def test_rot3_roundtrip(self):
        """Test rotating forward and back with rot3."""
        angle = np.pi / 3
        R = rot3(angle)
        R_inv = rot3(-angle)

        v = np.array([1, 2, 3])
        v_roundtrip = R_inv @ (R @ v)

        assert np.allclose(v, v_roundtrip, atol=1e-10)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_rot1_large_angle(self):
        """Test rot1 with angle > 2π."""
        R1 = rot1(np.pi / 4)
        R2 = rot1(np.pi / 4 + 2*np.pi)
        assert np.allclose(R1, R2, atol=1e-10)

    def test_rot2_large_angle(self):
        """Test rot2 with angle > 2π."""
        R1 = rot2(np.pi / 4)
        R2 = rot2(np.pi / 4 + 2*np.pi)
        assert np.allclose(R1, R2, atol=1e-10)

    def test_rot3_large_angle(self):
        """Test rot3 with angle > 2π."""
        R1 = rot3(np.pi / 4)
        R2 = rot3(np.pi / 4 + 2*np.pi)
        assert np.allclose(R1, R2, atol=1e-10)

    def test_rot1_multiple_full_rotations(self):
        """Test rot1 with multiple 360° rotations."""
        R = rot1(10 * 2 * np.pi)
        assert np.allclose(R, np.eye(3), atol=1e-9)

    def test_rot2_multiple_full_rotations(self):
        """Test rot2 with multiple 360° rotations."""
        R = rot2(10 * 2 * np.pi)
        assert np.allclose(R, np.eye(3), atol=1e-9)

    def test_rot3_multiple_full_rotations(self):
        """Test rot3 with multiple 360° rotations."""
        R = rot3(10 * 2 * np.pi)
        assert np.allclose(R, np.eye(3), atol=1e-9)
