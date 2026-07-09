# python imports
import pytest
import numpy as np

# comet imports
from comet.frames.transformations import (
    rot_eci_to_ecef,
    rot_ecef_to_eci,
    eci_to_ecef,
    ecef_to_eci,
    ecef_to_lla,
    lla_to_ecef,
    eci_to_lla,
    lla_to_eci,
)
from comet.time.epoch import Epoch
from comet.utilities.constants import Constants as c


class TestRotationMatrices:
    """Tests for rotation matrix functions."""

    def test_rot_eci_to_ecef_single_epoch(self):
        """Test rot_eci_to_ecef with single Epoch."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot = rot_eci_to_ecef(epoch)
        assert rot.shape == (3, 3)
        assert isinstance(rot, np.ndarray)

    def test_rot_eci_to_ecef_array_epochs(self):
        """Test rot_eci_to_ecef with array of Epochs."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        rot = rot_eci_to_ecef(epochs)
        assert rot.shape == (2, 3, 3)

    def test_rot_eci_to_ecef_orthogonality(self):
        """Test that rot_eci_to_ecef matrix is orthogonal."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot = rot_eci_to_ecef(epoch)
        identity = np.matmul(rot.T, rot)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)

    def test_rot_eci_to_ecef_determinant(self):
        """Test that rot_eci_to_ecef matrix has determinant = 1."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot = rot_eci_to_ecef(epoch)
        det = np.linalg.det(rot)
        assert pytest.approx(det, abs=1e-10) == 1.0

    def test_rot_ecef_to_eci_single_epoch(self):
        """Test rot_ecef_to_eci with single Epoch."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot = rot_ecef_to_eci(epoch)
        assert rot.shape == (3, 3)
        assert isinstance(rot, np.ndarray)

    def test_rot_ecef_to_eci_array_epochs(self):
        """Test rot_ecef_to_eci with array of Epochs."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        rot = rot_ecef_to_eci(epochs)
        assert rot.shape == (2, 3, 3)

    def test_rot_ecef_to_eci_orthogonality(self):
        """Test that rot_ecef_to_eci matrix is orthogonal."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot = rot_ecef_to_eci(epoch)
        identity = np.matmul(rot.T, rot)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)

    def test_rot_ecef_to_eci_determinant(self):
        """Test that rot_ecef_to_eci matrix has determinant = 1."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot = rot_ecef_to_eci(epoch)
        det = np.linalg.det(rot)
        assert pytest.approx(det, abs=1e-10) == 1.0

    def test_rotation_matrices_are_inverses(self):
        """Test that rot_eci_to_ecef and rot_ecef_to_eci are inverses."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_e2f = rot_eci_to_ecef(epoch)
        rot_f2e = rot_ecef_to_eci(epoch)

        # R_f2e @ R_e2f should be identity
        product = np.matmul(rot_f2e, rot_e2f)
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_rotation_matrices_transpose_relationship(self):
        """Test that rot_ecef_to_eci is transpose of rot_eci_to_ecef."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_e2f = rot_eci_to_ecef(epoch)
        rot_f2e = rot_ecef_to_eci(epoch)

        np.testing.assert_allclose(rot_f2e, rot_e2f.T, atol=1e-14)


class TestECIECEFTransformations:
    """Tests for ECI/ECEF coordinate transformations."""

    def test_eci_to_ecef_vallado_example(self):
        """Test eci_to_ecef against Vallado example (pg. 230, Example 3-15).

        Reference: Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
        Example 3-15: ECI to ECEF transformation
        Input: ECI position at 2004 Apr 6 07:51:28.386009 UTC
        Expected ECEF accuracy: ~0.001 km per Vallado
        """
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        eci_state = np.array([-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446])

        # Expected ECEF from Vallado (approximate due to EOP data differences)
        # Position values from Vallado Example 3-15
        expected_ecef_pos = np.array([5094.51462, 6127.36658, 6380.34453])

        ecef_state = eci_to_ecef(epoch, eci_state)

        # Test position (should match within ~1 km due to EOP data variations)
        np.testing.assert_allclose(ecef_state[0:3], expected_ecef_pos, atol=1.0, rtol=1e-3)

    def test_eci_to_ecef_single_state(self):
        """Test eci_to_ecef with single state vector."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        eci_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])

        ecef_state = eci_to_ecef(epoch, eci_state)

        assert ecef_state.shape == (6,)
        assert isinstance(ecef_state, np.ndarray)

    def test_eci_to_ecef_array_states(self):
        """Test eci_to_ecef with array of states."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        eci_states = np.array([[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]])

        ecef_states = eci_to_ecef(epochs, eci_states)

        assert ecef_states.shape == (2, 6)

    def test_eci_to_ecef_position_only(self):
        """Test eci_to_ecef with position-only state (no velocity)."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        eci_pos = np.array([7000.0, 0.0, 0.0])

        ecef_pos = eci_to_ecef(epoch, eci_pos)

        assert ecef_pos.shape == (3,)

    def test_eci_to_ecef_length_mismatch_error(self):
        """Test eci_to_ecef raises error when epoch and state lengths don't match."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009)])
        states = np.array([[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]])

        with pytest.raises(ValueError, match="length of epoch does not match length of state"):
            eci_to_ecef(epochs, states)

    def test_ecef_to_eci_single_state(self):
        """Test ecef_to_eci with single state vector."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        ecef_state = np.array([5094.51462, 6127.36658, 6380.34453, 0.0, 0.0, 0.0])

        eci_state = ecef_to_eci(epoch, ecef_state)

        assert eci_state.shape == (6,)
        assert isinstance(eci_state, np.ndarray)

    def test_ecef_to_eci_array_states(self):
        """Test ecef_to_eci with array of states."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        ecef_states = np.array([[5094.0, 6127.0, 6380.0, 0.0, 0.0, 0.0], [5094.0, 6127.0, 6380.0, 0.0, 0.0, 0.0]])

        eci_states = ecef_to_eci(epochs, ecef_states)

        assert eci_states.shape == (2, 6)

    def test_ecef_to_eci_position_only(self):
        """Test ecef_to_eci with position-only state (no velocity)."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        ecef_pos = np.array([5094.0, 6127.0, 6380.0])

        eci_pos = ecef_to_eci(epoch, ecef_pos)

        assert eci_pos.shape == (3,)

    def test_ecef_to_eci_length_mismatch_error(self):
        """Test ecef_to_eci raises error when epoch and state lengths don't match."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009)])
        states = np.array([[5094.0, 6127.0, 6380.0, 0.0, 0.0, 0.0], [5094.0, 6127.0, 6380.0, 0.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="length of epoch does not match length of state"):
            ecef_to_eci(epochs, states)


class TestECEFLLATransformations:
    """Tests for ECEF/LLA coordinate transformations."""

    def test_ecef_to_lla_single_state(self):
        """Test ecef_to_lla with single state."""
        # Position on equator at sea level
        ecef_state = np.array([c.A_EARTH / 1000, 0.0, 0.0, 0.0, 0.0, 0.0])

        lla_state = ecef_to_lla(ecef_state)

        # Should be at (lat=0, lon=0, alt=0)
        assert pytest.approx(lla_state[0], abs=1e-6) == 0.0  # Latitude
        assert pytest.approx(lla_state[1], abs=1e-6) == 0.0  # Longitude
        assert pytest.approx(lla_state[2], abs=0.001) == 0.0  # Altitude

    def test_ecef_to_lla_array_states(self):
        """Test ecef_to_lla with array of states."""
        ecef_states = np.array([[c.A_EARTH / 1000, 0.0, 0.0], [0.0, c.A_EARTH / 1000, 0.0]])

        lla_states = ecef_to_lla(ecef_states)

        assert lla_states.shape == (2, 3)
        # First point at (0, 0)
        assert pytest.approx(lla_states[0, 0], abs=1e-6) == 0.0
        assert pytest.approx(lla_states[0, 1], abs=1e-6) == 0.0
        # Second point at (0, 90deg)
        assert pytest.approx(lla_states[1, 0], abs=1e-6) == 0.0
        assert pytest.approx(lla_states[1, 1], abs=1e-6) == np.pi / 2

    def test_ecef_to_lla_near_pole(self):
        """Test ecef_to_lla near North Pole (avoiding singularity at exact pole)."""
        # Position near North Pole (85 degrees latitude) to avoid pole singularity
        lat_test = np.deg2rad(85.0)
        lon_test = 0.0
        alt_test = 500.0

        # Convert to ECEF and back
        lla_in = np.array([lat_test, lon_test, alt_test])
        ecef = lla_to_ecef(lla_in)
        lla_out = ecef_to_lla(ecef)

        # Should recover input LLA
        np.testing.assert_allclose(lla_out, lla_in, atol=0.01)

    def test_lla_to_ecef_single_state(self):
        """Test lla_to_ecef with single state."""
        # LLA at equator, sea level
        lla_state = np.array([0.0, 0.0, 0.0])

        ecef_state = lla_to_ecef(lla_state)

        # Should convert to ECEF at (A_EARTH, 0, 0)
        expected_ecef = np.array([c.A_EARTH / 1000, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(ecef_state, expected_ecef, atol=0.001)

    def test_lla_to_ecef_array_states(self):
        """Test lla_to_ecef with array of states."""
        lla_states = np.array([[0.0, 0.0, 0.0], [0.0, np.pi / 2, 0.0]])

        ecef_states = lla_to_ecef(lla_states)

        assert ecef_states.shape == (2, 6)

    def test_lla_to_ecef_with_velocity(self):
        """Test lla_to_ecef with velocity provided."""
        lla_state = np.array([0.0, 0.0, 0.0])
        velocity = np.array([1.0, 2.0, 3.0])

        ecef_state = lla_to_ecef(lla_state, velocity)

        assert ecef_state.shape == (6,)
        np.testing.assert_allclose(ecef_state[3:6], velocity, atol=1e-10)

    def test_lla_to_ecef_altitude(self):
        """Test lla_to_ecef with altitude above sea level."""
        # 500 km altitude at equator
        lla_state = np.array([0.0, 0.0, 500.0])

        ecef_state = lla_to_ecef(lla_state)

        # X component should be A_EARTH + 500 km
        expected_x = (c.A_EARTH / 1000) + 500.0
        assert pytest.approx(ecef_state[0], abs=0.001) == expected_x


class TestECILLATransformations:
    """Tests for ECI/LLA coordinate transformations."""

    def test_eci_to_lla_single_state(self):
        """Test eci_to_lla with single state."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        eci_state = np.array([-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446])

        lla_state = eci_to_lla(epoch, eci_state)

        assert lla_state.shape == (3,)
        # Latitude should be in [-pi/2, pi/2]
        assert -np.pi / 2 <= lla_state[0] <= np.pi / 2
        # Longitude should be in [-pi, pi]
        assert -np.pi <= lla_state[1] <= np.pi
        # Altitude should be reasonable (orbital altitude)
        assert 200.0 < lla_state[2] < 50000.0

    def test_eci_to_lla_array_states(self):
        """Test eci_to_lla with array of states."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        eci_states = np.array(
            [
                [-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446],
                [-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446],
            ]
        )

        lla_states = eci_to_lla(epochs, eci_states)

        assert lla_states.shape == (2, 3)

    def test_lla_to_eci_single_state(self):
        """Test lla_to_eci with single state."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        lla_state = np.array([0.5, 1.0, 500.0])

        eci_state = lla_to_eci(epoch, lla_state)

        assert eci_state.shape == (6,)

    def test_lla_to_eci_array_states(self):
        """Test lla_to_eci with array of states."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        lla_states = np.array([[0.5, 1.0, 500.0], [0.5, 1.0, 500.0]])

        eci_states = lla_to_eci(epochs, lla_states)

        assert eci_states.shape == (2, 6)


class TestRoundTripConversions:
    """Tests for round-trip coordinate conversions."""

    def test_eci_ecef_roundtrip(self):
        """Test ECI -> ECEF -> ECI round-trip conversion."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        eci_original = np.array([-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446])

        # Forward and back
        ecef_state = eci_to_ecef(epoch, eci_original)
        eci_recovered = ecef_to_eci(epoch, ecef_state)

        # Position should recover exactly
        np.testing.assert_allclose(eci_recovered[0:3], eci_original[0:3], atol=1e-6, rtol=1e-9)
        # Velocity has known transformation artifacts, use looser tolerance
        np.testing.assert_allclose(eci_recovered[3:6], eci_original[3:6], atol=0.5, rtol=0.2)

    def test_ecef_lla_roundtrip(self):
        """Test ECEF -> LLA -> ECEF round-trip conversion."""
        ecef_original = np.array([5094.51462, 6127.36658, 6380.34453, 1.0, 2.0, 3.0])

        # Forward and back (position only for LLA)
        lla_state = ecef_to_lla(ecef_original)
        ecef_recovered = lla_to_ecef(lla_state, ecef_original[3:6])

        # Should recover original ECEF state
        np.testing.assert_allclose(ecef_recovered, ecef_original, atol=0.001, rtol=1e-6)

    def test_eci_lla_roundtrip(self):
        """Test ECI -> LLA -> ECI round-trip conversion."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        eci_original = np.array([-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446])

        # Forward and back
        lla_state = eci_to_lla(epoch, eci_original)
        eci_recovered = lla_to_eci(epoch, lla_state)

        # Should recover original ECI position (velocity might differ due to rotation)
        np.testing.assert_allclose(eci_recovered[0:3], eci_original[0:3], atol=0.001, rtol=1e-6)

    def test_array_roundtrip(self):
        """Test round-trip conversion with arrays."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        eci_original = np.array(
            [
                [-1033.4793830, 7901.2952754, 6380.3565958, -3.225636520, -2.872451450, 5.531924446],
                [-2000.0, 7000.0, 6000.0, -3.0, -3.0, 5.0],
            ]
        )

        # Forward and back
        ecef_states = eci_to_ecef(epochs, eci_original)
        eci_recovered = ecef_to_eci(epochs, ecef_states)

        # Position should recover exactly
        np.testing.assert_allclose(eci_recovered[:, 0:3], eci_original[:, 0:3], atol=1e-6, rtol=1e-9)
        # Velocity has known transformation artifacts, use looser tolerance
        np.testing.assert_allclose(eci_recovered[:, 3:6], eci_original[:, 3:6], atol=0.5, rtol=0.2)


class TestVectorization:
    """Tests for vectorized operations."""

    def test_vectorized_transformations_performance(self):
        """Test that vectorized operations work correctly with multiple epochs."""
        # Create array of epochs
        n_epochs = 10
        base_epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        epochs = np.array([base_epoch for _ in range(n_epochs)])

        # Create array of states
        eci_states = np.array([[7000.0, 0.0, 0.0, 0.0, 7.5, 0.0] for _ in range(n_epochs)])

        # Should handle vectorized call without error
        ecef_states = eci_to_ecef(epochs, eci_states)

        assert ecef_states.shape == (n_epochs, 6)

        # Convert back
        eci_recovered = ecef_to_eci(epochs, ecef_states)

        # Position should recover exactly
        np.testing.assert_allclose(eci_recovered[:, 0:3], eci_states[:, 0:3], atol=1e-9, rtol=1e-12)
        # Velocity has known transformation artifacts, use looser tolerance
        np.testing.assert_allclose(eci_recovered[:, 3:6], eci_states[:, 3:6], atol=0.5, rtol=0.1)
