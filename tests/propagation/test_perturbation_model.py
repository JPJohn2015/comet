# python imports
import pytest

# comet imports
from comet.propagation.perturbation_model import (
    AtmosphereModel,
    NBodyModel,
    SRPModel,
    GravityPotentialModel,
)


class TestAtmosphereModel:
    """Tests for AtmosphereModel enum."""

    def test_atmosphere_model_values(self):
        """Test AtmosphereModel has correct values."""
        assert AtmosphereModel.EXPONENTIAL.value == 0
        assert AtmosphereModel.COESA76.value == 1

    def test_atmosphere_model_membership(self):
        """Test AtmosphereModel membership."""
        assert AtmosphereModel.EXPONENTIAL in AtmosphereModel
        assert AtmosphereModel.COESA76 in AtmosphereModel

    def test_atmosphere_model_count(self):
        """Test AtmosphereModel has exactly 2 members."""
        assert len(AtmosphereModel) == 2

    def test_atmosphere_model_access_by_name(self):
        """Test AtmosphereModel can be accessed by name."""
        assert AtmosphereModel["EXPONENTIAL"] == AtmosphereModel.EXPONENTIAL
        assert AtmosphereModel["COESA76"] == AtmosphereModel.COESA76

    def test_atmosphere_model_access_by_value(self):
        """Test AtmosphereModel can be accessed by value."""
        assert AtmosphereModel(0) == AtmosphereModel.EXPONENTIAL
        assert AtmosphereModel(1) == AtmosphereModel.COESA76


class TestNBodyModel:
    """Tests for NBodyModel enum."""

    def test_nbody_model_values(self):
        """Test NBodyModel has correct values."""
        assert NBodyModel.MOON.value == 0
        assert NBodyModel.SUN.value == 1
        assert NBodyModel.SUNandMOON.value == 2

    def test_nbody_model_membership(self):
        """Test NBodyModel membership."""
        assert NBodyModel.MOON in NBodyModel
        assert NBodyModel.SUN in NBodyModel
        assert NBodyModel.SUNandMOON in NBodyModel

    def test_nbody_model_count(self):
        """Test NBodyModel has exactly 3 members."""
        assert len(NBodyModel) == 3

    def test_nbody_model_access_by_name(self):
        """Test NBodyModel can be accessed by name."""
        assert NBodyModel["MOON"] == NBodyModel.MOON
        assert NBodyModel["SUN"] == NBodyModel.SUN
        assert NBodyModel["SUNandMOON"] == NBodyModel.SUNandMOON

    def test_nbody_model_access_by_value(self):
        """Test NBodyModel can be accessed by value."""
        assert NBodyModel(0) == NBodyModel.MOON
        assert NBodyModel(1) == NBodyModel.SUN
        assert NBodyModel(2) == NBodyModel.SUNandMOON


class TestSRPModel:
    """Tests for SRPModel enum."""

    def test_srp_model_values(self):
        """Test SRPModel has correct values."""
        assert SRPModel.SUN.value == 0
        assert SRPModel.ALBEDO.value == 1
        assert SRPModel.SUNandALBEDO.value == 2

    def test_srp_model_membership(self):
        """Test SRPModel membership."""
        assert SRPModel.SUN in SRPModel
        assert SRPModel.ALBEDO in SRPModel
        assert SRPModel.SUNandALBEDO in SRPModel

    def test_srp_model_count(self):
        """Test SRPModel has exactly 3 members."""
        assert len(SRPModel) == 3

    def test_srp_model_access_by_name(self):
        """Test SRPModel can be accessed by name."""
        assert SRPModel["SUN"] == SRPModel.SUN
        assert SRPModel["ALBEDO"] == SRPModel.ALBEDO
        assert SRPModel["SUNandALBEDO"] == SRPModel.SUNandALBEDO

    def test_srp_model_access_by_value(self):
        """Test SRPModel can be accessed by value."""
        assert SRPModel(0) == SRPModel.SUN
        assert SRPModel(1) == SRPModel.ALBEDO
        assert SRPModel(2) == SRPModel.SUNandALBEDO


class TestGravityPotentialModel:
    """Tests for GravityPotentialModel enum."""

    def test_gravity_potential_model_values(self):
        """Test GravityPotentialModel has correct values."""
        assert GravityPotentialModel.J2.value == 0
        assert GravityPotentialModel.J2andJ3.value == 1

    def test_gravity_potential_model_membership(self):
        """Test GravityPotentialModel membership."""
        assert GravityPotentialModel.J2 in GravityPotentialModel
        assert GravityPotentialModel.J2andJ3 in GravityPotentialModel

    def test_gravity_potential_model_count(self):
        """Test GravityPotentialModel has exactly 2 members."""
        assert len(GravityPotentialModel) == 2

    def test_gravity_potential_model_access_by_name(self):
        """Test GravityPotentialModel can be accessed by name."""
        assert GravityPotentialModel["J2"] == GravityPotentialModel.J2
        assert GravityPotentialModel["J2andJ3"] == GravityPotentialModel.J2andJ3

    def test_gravity_potential_model_access_by_value(self):
        """Test GravityPotentialModel can be accessed by value."""
        assert GravityPotentialModel(0) == GravityPotentialModel.J2
        assert GravityPotentialModel(1) == GravityPotentialModel.J2andJ3
