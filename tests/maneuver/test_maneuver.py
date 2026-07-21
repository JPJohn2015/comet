# python imports
import pytest
import numpy as np

# COMET imports
from comet.maneuver import Maneuver, ImpulsiveManeuver, FiniteManeuver
from comet.time import Epoch
from comet.time import Duration
from comet.frames import ManeuverFrame
from comet.components import Thruster


class TestManeuverConstruction:
    """Test Maneuver construction."""

    def test_maneuver_basic(self):
        """Test basic Maneuver construction."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = np.array([0.1, 0.0, 0.0])
        maneuver = Maneuver(epoch, dv)
        assert maneuver.epoch == epoch
        assert np.allclose(maneuver.dv, dv)
        assert maneuver.frame == ManeuverFrame.ECI
        assert maneuver.id is not None

    def test_maneuver_with_frame(self):
        """Test Maneuver with custom frame."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.05, 0.05, 0.0]
        maneuver = Maneuver(epoch, dv, ManeuverFrame.RIC)
        assert maneuver.frame == ManeuverFrame.RIC

    def test_maneuver_string_frame(self):
        """Test Maneuver with string frame."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        maneuver = Maneuver(epoch, dv, "ECI")
        assert maneuver.frame == ManeuverFrame.ECI


class TestManeuverMethods:
    """Test Maneuver methods."""

    def test_copy(self):
        """Test copy method."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        m1 = Maneuver(epoch, dv)
        m2 = m1.copy()
        assert m1 is not m2
        assert m1.epoch == m2.epoch
        assert np.allclose(m1.dv, m2.dv)

    def test_get_magnitude(self):
        """Test get_magnitude method."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.3, 0.4, 0.0]
        maneuver = Maneuver(epoch, dv)
        assert pytest.approx(maneuver.get_magnitude(), abs=1e-10) == 0.5

    def test_duration_to_maneuver(self):
        """Test duration_to_maneuver method."""
        epoch1 = Epoch(2024, 1, 1, 12, 0, 0)
        epoch2 = Epoch(2024, 1, 1, 13, 0, 0)
        dv = [0.1, 0.0, 0.0]
        maneuver = Maneuver(epoch2, dv)
        duration = maneuver.duration_to_maneuver(epoch1)
        assert isinstance(duration, Duration)

    def test_update(self):
        """Test update method."""
        epoch1 = Epoch(2024, 1, 1, 12, 0, 0)
        epoch2 = Epoch(2024, 1, 1, 13, 0, 0)
        dv1 = [0.1, 0.0, 0.0]
        dv2 = [0.2, 0.0, 0.0]
        maneuver = Maneuver(epoch1, dv1)
        maneuver.update(epoch=epoch2, dv=dv2)
        assert maneuver.epoch == epoch2
        assert np.allclose(maneuver.dv, dv2)


class TestManeuverSerialization:
    """Test Maneuver serialization."""

    def test_to_dict(self):
        """Test to_dict method."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        maneuver = Maneuver(epoch, dv)
        d = maneuver.to_dict()
        assert d["type"] == "Maneuver"
        assert "epoch" in d
        assert "dv" in d

    def test_from_dict(self):
        """Test from_dict method."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        m1 = Maneuver(epoch, dv)
        d = m1.to_dict()
        m2 = Maneuver.from_dict(d)
        assert m2.epoch == m1.epoch
        assert np.allclose(m2.dv, m1.dv)


class TestImpulsiveManeuver:
    """Test ImpulsiveManeuver."""

    def test_construction(self):
        """Test ImpulsiveManeuver construction."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        impulse = ImpulsiveManeuver(epoch, dv)
        assert isinstance(impulse, Maneuver)
        assert impulse.epoch == epoch
        assert np.allclose(impulse.dv, dv)

    def test_equality(self):
        """Test ImpulsiveManeuver equality."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        i1 = ImpulsiveManeuver(epoch, dv)
        i2 = ImpulsiveManeuver(epoch, dv)
        # IDs will be different, so they won't be equal
        assert i1.id != i2.id

    def test_serialization(self):
        """Test ImpulsiveManeuver serialization."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        i1 = ImpulsiveManeuver(epoch, dv)
        d = i1.to_dict()
        assert d["type"] == "ImpulsiveManeuver"
        i2 = ImpulsiveManeuver.from_dict(d)
        assert i2.epoch == i1.epoch
        assert np.allclose(i2.dv, i1.dv)


class TestFiniteManeuver:
    """Test FiniteManeuver."""

    def test_construction_default_thruster(self):
        """Test FiniteManeuver with default thruster."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        finite = FiniteManeuver(epoch, dv)
        assert isinstance(finite, Maneuver)
        assert finite.thruster is not None

    def test_construction_custom_thruster(self):
        """Test FiniteManeuver with custom thruster."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        thruster = Thruster(thrust=100.0, isp=300.0)
        finite = FiniteManeuver(epoch, dv, thruster=thruster)
        assert finite.thruster == thruster

    def test_burn_duration(self):
        """Test burn_duration method."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        finite = FiniteManeuver(epoch, dv)
        duration = finite.burn_duration(1000.0)
        assert duration > 0

    def test_mass_expended(self):
        """Test mass_expended method."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        finite = FiniteManeuver(epoch, dv)
        mass_exp = finite.mass_expended(1000.0)
        assert mass_exp < 0  # Mass is lost

    def test_burn_start_stop(self):
        """Test burn_start_stop method."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        finite = FiniteManeuver(epoch, dv)
        start, stop = finite.burn_start_stop(1000.0)
        assert isinstance(start, Epoch)
        assert isinstance(stop, Epoch)
        assert start < epoch < stop

    def test_serialization(self):
        """Test FiniteManeuver serialization."""
        epoch = Epoch(2024, 1, 1, 12, 0, 0)
        dv = [0.1, 0.0, 0.0]
        f1 = FiniteManeuver(epoch, dv)
        d = f1.to_dict()
        assert d["type"] == "FiniteManeuver"
        f2 = FiniteManeuver.from_dict(d)
        assert f2.epoch == f1.epoch
        assert np.allclose(f2.dv, f1.dv)
