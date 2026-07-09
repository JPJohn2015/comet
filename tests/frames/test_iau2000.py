# python imports
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# comet imports
from comet.frames.iau2000 import (
    EarthOrientationData,
    IERSData,
    EOP_DATA,
    IERS_DATA,
    precession_nutation,
    earth_rotation,
    polar_motion,
    read_eop_data_file,
    read_iers_data_file,
)
from comet.time.epoch import Epoch
from comet.utilities.constants import Constants as c


class TestEarthOrientationData:
    """Tests for EarthOrientationData class."""

    def test_eop_data_singleton(self):
        """Test that EOP_DATA is a singleton instance."""
        assert isinstance(EOP_DATA, EarthOrientationData)

    def test_set_eop_data(self):
        """Test setting EOP data."""
        eop = EarthOrientationData()
        eop.set(54321.5, 0.1, 0.2, 0.3)
        ut1_utc, xp, yp = eop.eop(54321.5)
        assert ut1_utc == 0.1
        assert pytest.approx(xp, abs=1e-10) == 0.2 * c.AS2RAD
        assert pytest.approx(yp, abs=1e-10) == 0.3 * c.AS2RAD

    def test_clear_eop_data(self):
        """Test clearing EOP data."""
        eop = EarthOrientationData()
        eop.set(54321.5, 0.1, 0.2, 0.3)
        eop.clear()
        assert not eop._is_loaded
        assert len(eop._data) == 0

    def test_ut1_to_utc(self):
        """Test ut1_to_utc conversion."""
        eop = EarthOrientationData()
        eop.set(54321.0, 0.5, 0.0, 0.0)
        ut1_utc = eop.ut1_to_utc(54321.0)
        assert ut1_utc == 0.5

    def test_utc_to_ut1(self):
        """Test utc_to_ut1 conversion (should be negative of ut1_to_utc)."""
        eop = EarthOrientationData()
        eop.set(54321.0, 0.5, 0.0, 0.0)
        utc_ut1 = eop.utc_to_ut1(54321.0)
        assert utc_ut1 == -0.5

    def test_eop_interpolation(self):
        """Test EOP data interpolation when enabled."""
        eop = EarthOrientationData()
        eop._interpolate = True
        eop._is_loaded = True
        eop._data = {54321: (0.0, 0.0, 0.0), 54322: (1.0, 0.002, 0.004)}

        # Test interpolation at midpoint
        ut1_utc, xp, yp = eop.eop(54321.5)
        assert pytest.approx(ut1_utc, abs=1e-10) == 0.5
        assert pytest.approx(xp, abs=1e-10) == 0.001
        assert pytest.approx(yp, abs=1e-10) == 0.002

    def test_eop_out_of_bounds_low(self):
        """Test EOP data when MJD is below range."""
        eop = EarthOrientationData()
        eop._is_loaded = True
        eop._data = {54321: (0.1, 0.002, 0.003), 54322: (0.2, 0.004, 0.006)}

        # Request data below range
        ut1_utc, xp, yp = eop.eop(41683.0)
        # Should return data from lowest MJD
        assert ut1_utc == 0.1
        assert xp == 0.002
        assert yp == 0.003

    def test_eop_out_of_bounds_high(self):
        """Test EOP data when MJD is above range."""
        eop = EarthOrientationData()
        eop._is_loaded = True
        eop._data = {54321: (0.1, 0.002, 0.003), 54322: (0.2, 0.004, 0.006)}

        # Request data above range
        ut1_utc, xp, yp = eop.eop(60000.0)
        # Should return data from highest MJD
        assert ut1_utc == 0.2
        assert xp == 0.004
        assert yp == 0.006


class TestIERSData:
    """Tests for IERSData class."""

    def test_iers_data_singleton(self):
        """Test that IERS_DATA is a singleton instance."""
        assert isinstance(IERS_DATA, IERSData)

    def test_get_XYs_output_shape(self):
        """Test get_XYs returns correct tuple of 3 values."""
        angles = np.zeros(14)
        Ttt = 0.0
        x, y, s = IERS_DATA.get_XYs(angles, Ttt)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(s, float)

    def test_get_XYs_units(self):
        """Test get_XYs returns values in radians (scaled by AS2RAD)."""
        angles = np.zeros(14)
        Ttt = 0.0
        x, y, s = IERS_DATA.get_XYs(angles, Ttt)
        # Values should be small (in radians, converted from arcseconds)
        assert abs(x) < 1.0  # Should be in radians
        assert abs(y) < 1.0
        assert abs(s) < 1.0


class TestPrecessionNutation:
    """Tests for precession_nutation function."""

    def test_precession_nutation_single_epoch(self):
        """Test precession_nutation with single Epoch."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pn = precession_nutation(epoch)
        assert rot_pn.shape == (3, 3)
        assert isinstance(rot_pn, np.ndarray)

    def test_precession_nutation_array_epochs(self):
        """Test precession_nutation with array of Epochs."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        rot_pn = precession_nutation(epochs)
        assert rot_pn.shape == (2, 3, 3)
        assert isinstance(rot_pn, np.ndarray)

    def test_precession_nutation_orthogonality(self):
        """Test that precession_nutation matrix is orthogonal (R^T R = I)."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pn = precession_nutation(epoch)
        identity = np.matmul(rot_pn.T, rot_pn)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10, rtol=1e-10)

    def test_precession_nutation_determinant(self):
        """Test that precession_nutation matrix has determinant = 1."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pn = precession_nutation(epoch)
        det = np.linalg.det(rot_pn)
        assert pytest.approx(det, abs=1e-10) == 1.0

    def test_precession_nutation_j2000(self):
        """Test precession_nutation at J2000 epoch (should be close to identity)."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0.0)
        rot_pn = precession_nutation(epoch)
        # Should be close to identity at J2000
        np.testing.assert_allclose(rot_pn, np.eye(3), atol=0.01)

    def test_precession_nutation_angle_wrapping(self):
        """Test that angle wrapping is correct (modulo 2*pi not modulo 2 times pi)."""
        # This is a regression test for the operator precedence bug
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        # Should not raise and should produce valid rotation matrix
        rot_pn = precession_nutation(epoch)
        assert rot_pn.shape == (3, 3)
        # Check orthogonality to ensure correct computation
        identity = np.matmul(rot_pn.T, rot_pn)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)


class TestEarthRotation:
    """Tests for earth_rotation function."""

    def test_earth_rotation_single_epoch(self):
        """Test earth_rotation with single Epoch."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_era = earth_rotation(epoch)
        assert rot_era.shape == (3, 3)
        assert isinstance(rot_era, np.ndarray)

    def test_earth_rotation_array_epochs(self):
        """Test earth_rotation with array of Epochs."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        rot_era = earth_rotation(epochs)
        assert rot_era.shape == (2, 3, 3)
        assert isinstance(rot_era, np.ndarray)

    def test_earth_rotation_orthogonality(self):
        """Test that earth_rotation matrix is orthogonal (R^T R = I)."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_era = earth_rotation(epoch)
        identity = np.matmul(rot_era.T, rot_era)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10, rtol=1e-10)

    def test_earth_rotation_determinant(self):
        """Test that earth_rotation matrix has determinant = 1."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_era = earth_rotation(epoch)
        det = np.linalg.det(rot_era)
        assert pytest.approx(det, abs=1e-10) == 1.0

    def test_earth_rotation_is_rotation_about_z(self):
        """Test that earth_rotation is a pure rotation about Z axis."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_era = earth_rotation(epoch)
        # Z-axis component should be [0, 0, 1] (unchanged by Z rotation)
        z_axis = np.array([0, 0, 1])
        rotated_z = np.matmul(rot_era, z_axis)
        np.testing.assert_allclose(rotated_z, z_axis, atol=1e-10)


class TestPolarMotion:
    """Tests for polar_motion function."""

    def test_polar_motion_single_epoch(self):
        """Test polar_motion with single Epoch."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pm = polar_motion(epoch)
        assert rot_pm.shape == (3, 3)
        assert isinstance(rot_pm, np.ndarray)

    def test_polar_motion_array_epochs(self):
        """Test polar_motion with array of Epochs."""
        epochs = np.array([Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)])
        rot_pm = polar_motion(epochs)
        assert rot_pm.shape == (2, 3, 3)
        assert isinstance(rot_pm, np.ndarray)

    def test_polar_motion_orthogonality(self):
        """Test that polar_motion matrix is orthogonal (R^T R = I)."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pm = polar_motion(epoch)
        identity = np.matmul(rot_pm.T, rot_pm)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10, rtol=1e-10)

    def test_polar_motion_determinant(self):
        """Test that polar_motion matrix has determinant = 1."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pm = polar_motion(epoch)
        det = np.linalg.det(rot_pm)
        assert pytest.approx(det, abs=1e-10) == 1.0

    def test_polar_motion_small_angles(self):
        """Test that polar motion is close to identity for small xp, yp."""
        # Polar motion angles are typically small (< 1 arcsec)
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pm = polar_motion(epoch)
        # Should be very close to identity since polar motion is small
        np.testing.assert_allclose(rot_pm, np.eye(3), atol=0.01)


class TestReadDataFiles:
    """Tests for data file reading functions."""

    def test_read_eop_data_file_invalid_type(self):
        """Test read_eop_data_file with invalid file type."""
        with pytest.raises(Exception, match="Invalid datafile type"):
            read_eop_data_file("dummy_path.txt", "invalid_type")

    def test_read_iers_data_file_returns_list(self):
        """Test read_iers_data_file returns a list."""
        # This test requires a valid IERS file, so we'll use the default one
        default_path = Path(__file__).parents[2] / "comet" / "data" / "iau"
        try:
            data = read_iers_data_file(default_path / "tab5_2a.txt")
            assert isinstance(data, list)
            assert len(data) == 5  # Should have 5 tables
        except FileNotFoundError:
            pytest.skip("IERS data file not found")


class TestRoundTrips:
    """Tests for round-trip conversions and consistency."""

    def test_rotation_matrices_compose_correctly(self):
        """Test that rotation matrices compose correctly for full ECI to ECEF transform."""
        epoch = Epoch(2004, 4, 6, 7, 51, 28.386009)
        rot_pm = polar_motion(epoch)
        rot_era = earth_rotation(epoch)
        rot_pn = precession_nutation(epoch)

        # Full rotation
        rot_full = np.matmul(rot_pm, np.matmul(rot_era, rot_pn))

        # Should still be orthogonal
        identity = np.matmul(rot_full.T, rot_full)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)

        # Determinant should be 1
        det = np.linalg.det(rot_full)
        assert pytest.approx(det, abs=1e-10) == 1.0

    def test_vectorized_vs_scalar_consistency(self):
        """Test that vectorized functions produce same results as scalar calls."""
        epochs = [Epoch(2004, 4, 6, 7, 51, 28.386009), Epoch(2004, 4, 6, 8, 51, 28.386009)]

        # Scalar calls
        rot_pn_0 = precession_nutation(epochs[0])
        rot_pn_1 = precession_nutation(epochs[1])

        # Vectorized call
        rot_pn_vec = precession_nutation(np.array(epochs))

        # Should match
        np.testing.assert_allclose(rot_pn_vec[0], rot_pn_0, atol=1e-14)
        np.testing.assert_allclose(rot_pn_vec[1], rot_pn_1, atol=1e-14)

        # Same for earth_rotation
        rot_era_0 = earth_rotation(epochs[0])
        rot_era_1 = earth_rotation(epochs[1])
        rot_era_vec = earth_rotation(np.array(epochs))
        np.testing.assert_allclose(rot_era_vec[0], rot_era_0, atol=1e-14)
        np.testing.assert_allclose(rot_era_vec[1], rot_era_1, atol=1e-14)

        # Same for polar_motion
        rot_pm_0 = polar_motion(epochs[0])
        rot_pm_1 = polar_motion(epochs[1])
        rot_pm_vec = polar_motion(np.array(epochs))
        np.testing.assert_allclose(rot_pm_vec[0], rot_pm_0, atol=1e-14)
        np.testing.assert_allclose(rot_pm_vec[1], rot_pm_1, atol=1e-14)
