# python imports
import pytest
import numpy as np

# comet imports
from comet.propagation.force_model import ForceModel
from comet.propagation.perturbation_model import (
    AtmosphereModel,
    NBodyModel,
    SRPModel,
    GravityPotentialModel,
)


class TestForceModelConstruction:
    """Tests for ForceModel construction."""

    def test_default_construction(self):
        """Test ForceModel with default values (all False)."""
        fm = ForceModel()
        assert fm.drag is False
        assert fm.srp is False
        assert fm.nbody is False
        assert fm.gravity is False
        assert fm.atmosphere_model == AtmosphereModel.EXPONENTIAL
        assert fm.srp_model == SRPModel.SUN
        assert fm.nbody_model == NBodyModel.SUNandMOON
        assert fm.gravity_model == GravityPotentialModel.J2andJ3

    def test_custom_construction_perturbations(self):
        """Test ForceModel with custom perturbation flags."""
        fm = ForceModel(drag=True, srp=True, nbody=False, gravity=True)
        assert fm.drag is True
        assert fm.srp is True
        assert fm.nbody is False
        assert fm.gravity is True

    def test_custom_construction_models(self):
        """Test ForceModel with custom models."""
        fm = ForceModel(
            atmosphere_model=AtmosphereModel.COESA76,
            srp_model=SRPModel.ALBEDO,
            nbody_model=NBodyModel.MOON,
            gravity_model=GravityPotentialModel.J2,
        )
        assert fm.atmosphere_model == AtmosphereModel.COESA76
        assert fm.srp_model == SRPModel.ALBEDO
        assert fm.nbody_model == NBodyModel.MOON
        assert fm.gravity_model == GravityPotentialModel.J2

    def test_full_custom_construction(self):
        """Test ForceModel with all custom parameters."""
        fm = ForceModel(
            drag=True,
            srp=True,
            nbody=True,
            gravity=True,
            atmosphere_model=AtmosphereModel.COESA76,
            srp_model=SRPModel.SUNandALBEDO,
            nbody_model=NBodyModel.SUNandMOON,
            gravity_model=GravityPotentialModel.J2andJ3,
        )
        assert fm.drag is True
        assert fm.srp is True
        assert fm.nbody is True
        assert fm.gravity is True
        assert fm.atmosphere_model == AtmosphereModel.COESA76
        assert fm.srp_model == SRPModel.SUNandALBEDO
        assert fm.nbody_model == NBodyModel.SUNandMOON
        assert fm.gravity_model == GravityPotentialModel.J2andJ3


class TestForceModelCopy:
    """Tests for ForceModel copy method."""

    def test_copy_creates_new_instance(self):
        """Test copy creates a new instance."""
        fm1 = ForceModel(drag=True, srp=True, gravity=True)
        fm2 = fm1.copy()
        assert fm1 is not fm2
        assert fm1.drag == fm2.drag
        assert fm1.srp == fm2.srp
        assert fm1.gravity == fm2.gravity

    def test_copy_is_independent(self):
        """Test that modifying copy doesn't affect original."""
        fm1 = ForceModel(drag=True)
        fm2 = fm1.copy()
        fm2.update_atmosphere_model(AtmosphereModel.COESA76)
        assert fm1.atmosphere_model == AtmosphereModel.EXPONENTIAL
        assert fm2.atmosphere_model == AtmosphereModel.COESA76


class TestForceModelIsTwoBody:
    """Tests for is_two_body method."""

    def test_is_two_body_all_false(self):
        """Test is_two_body returns True when all perturbations are False."""
        fm = ForceModel()
        assert fm.is_two_body() is True

    def test_is_two_body_with_drag(self):
        """Test is_two_body returns False when drag is enabled."""
        fm = ForceModel(drag=True)
        assert fm.is_two_body() is False

    def test_is_two_body_with_srp(self):
        """Test is_two_body returns False when SRP is enabled."""
        fm = ForceModel(srp=True)
        assert fm.is_two_body() is False

    def test_is_two_body_with_nbody(self):
        """Test is_two_body returns False when nbody is enabled."""
        fm = ForceModel(nbody=True)
        assert fm.is_two_body() is False

    def test_is_two_body_with_gravity(self):
        """Test is_two_body returns False when gravity is enabled."""
        fm = ForceModel(gravity=True)
        assert fm.is_two_body() is False

    def test_is_two_body_with_multiple(self):
        """Test is_two_body returns False when multiple perturbations are enabled."""
        fm = ForceModel(drag=True, srp=True, nbody=True, gravity=True)
        assert fm.is_two_body() is False


class TestForceModelIsSunNeeded:
    """Tests for is_sun_needed method."""

    def test_is_sun_needed_default(self):
        """Test is_sun_needed with default models."""
        fm = ForceModel()
        # Default: srp_model=SUN, nbody_model=SUNandMOON
        # Sun is needed
        assert fm.is_sun_needed() is True

    def test_is_sun_needed_sun_srp_model(self):
        """Test is_sun_needed when SRP model is SUN."""
        fm = ForceModel(srp_model=SRPModel.SUN, nbody_model=NBodyModel.MOON)
        assert fm.is_sun_needed() is True

    def test_is_sun_needed_sun_nbody_model(self):
        """Test is_sun_needed when NBody model is SUN."""
        fm = ForceModel(srp_model=SRPModel.ALBEDO, nbody_model=NBodyModel.SUN)
        assert fm.is_sun_needed() is True

    def test_is_sun_needed_sunandmoon_nbody_model(self):
        """Test is_sun_needed when NBody model is SUNandMOON."""
        fm = ForceModel(srp_model=SRPModel.ALBEDO, nbody_model=NBodyModel.SUNandMOON)
        assert fm.is_sun_needed() is True

    def test_is_sun_needed_only_albedo_and_moon(self):
        """Test is_sun_needed when only ALBEDO and MOON (Sun NOT needed)."""
        fm = ForceModel(srp_model=SRPModel.ALBEDO, nbody_model=NBodyModel.MOON)
        assert fm.is_sun_needed() is False

    def test_is_sun_needed_sunandalbedo(self):
        """Test is_sun_needed when SRP model is SUNandALBEDO."""
        fm = ForceModel(srp_model=SRPModel.SUNandALBEDO, nbody_model=NBodyModel.MOON)
        assert fm.is_sun_needed() is True


class TestForceModelIsMoonNeeded:
    """Tests for is_moon_needed method."""

    def test_is_moon_needed_default(self):
        """Test is_moon_needed with default models."""
        fm = ForceModel()
        # Default: nbody_model=SUNandMOON
        # Moon is needed
        assert fm.is_moon_needed() is True

    def test_is_moon_needed_sun_only(self):
        """Test is_moon_needed when NBody model is SUN (Moon NOT needed)."""
        fm = ForceModel(nbody_model=NBodyModel.SUN)
        assert fm.is_moon_needed() is False

    def test_is_moon_needed_moon_only(self):
        """Test is_moon_needed when NBody model is MOON."""
        fm = ForceModel(nbody_model=NBodyModel.MOON)
        assert fm.is_moon_needed() is True

    def test_is_moon_needed_sunandmoon(self):
        """Test is_moon_needed when NBody model is SUNandMOON."""
        fm = ForceModel(nbody_model=NBodyModel.SUNandMOON)
        assert fm.is_moon_needed() is True


class TestForceModelUpdaters:
    """Tests for ForceModel update methods."""

    def test_update_atmosphere_model(self):
        """Test update_atmosphere_model changes the model."""
        fm = ForceModel()
        fm.update_atmosphere_model(AtmosphereModel.COESA76)
        assert fm.atmosphere_model == AtmosphereModel.COESA76

    def test_update_srp_model(self):
        """Test update_srp_model changes the model."""
        fm = ForceModel()
        fm.update_srp_model(SRPModel.ALBEDO)
        assert fm.srp_model == SRPModel.ALBEDO

    def test_update_nbody_model(self):
        """Test update_nbody_model changes the model."""
        fm = ForceModel()
        fm.update_nbody_model(NBodyModel.SUN)
        assert fm.nbody_model == NBodyModel.SUN

    def test_update_gravity_model(self):
        """Test update_gravity_model changes the model."""
        fm = ForceModel()
        fm.update_gravity_model(GravityPotentialModel.J2andJ3)
        assert fm.gravity_model == GravityPotentialModel.J2andJ3


class TestForceModelSerialization:
    """Tests for ForceModel serialization."""

    def test_to_dict(self):
        """Test to_dict creates correct dictionary."""
        fm = ForceModel(
            drag=True,
            srp=True,
            nbody=False,
            gravity=True,
            atmosphere_model=AtmosphereModel.COESA76,
            srp_model=SRPModel.ALBEDO,
            nbody_model=NBodyModel.MOON,
            gravity_model=GravityPotentialModel.J2,
        )
        fm_dict = fm.to_dict()
        assert fm_dict["type"] == "ForceModel"
        assert fm_dict["drag"] is True
        assert fm_dict["srp"] is True
        assert fm_dict["nbody"] is False
        assert fm_dict["gravity"] is True
        assert fm_dict["atmosphere_model"] == AtmosphereModel.COESA76.value
        assert fm_dict["srp_model"] == SRPModel.ALBEDO.value
        assert fm_dict["nbody_model"] == NBodyModel.MOON.value
        assert fm_dict["gravity_model"] == GravityPotentialModel.J2.value

    def test_from_dict(self):
        """Test from_dict reconstructs ForceModel."""
        fm_dict = {
            "type": "ForceModel",
            "drag": True,
            "srp": False,
            "nbody": True,
            "gravity": True,
            "atmosphere_model": AtmosphereModel.COESA76.value,
            "srp_model": SRPModel.SUN.value,
            "nbody_model": NBodyModel.SUNandMOON.value,
            "gravity_model": GravityPotentialModel.J2andJ3.value,
        }
        fm = ForceModel.from_dict(fm_dict)
        assert fm.drag is True
        assert fm.srp is False
        assert fm.nbody is True
        assert fm.gravity is True
        assert fm.atmosphere_model == AtmosphereModel.COESA76
        assert fm.srp_model == SRPModel.SUN
        assert fm.nbody_model == NBodyModel.SUNandMOON
        assert fm.gravity_model == GravityPotentialModel.J2andJ3

    def test_from_dict_invalid_type(self):
        """Test from_dict raises error for invalid type."""
        fm_dict = {"type": "InvalidType"}
        with pytest.raises(ValueError, match="Invalid construction dictionary"):
            ForceModel.from_dict(fm_dict)

    def test_round_trip_serialization(self):
        """Test to_dict and from_dict round trip."""
        fm1 = ForceModel(
            drag=True,
            srp=True,
            nbody=True,
            gravity=False,
            atmosphere_model=AtmosphereModel.EXPONENTIAL,
            srp_model=SRPModel.SUNandALBEDO,
            nbody_model=NBodyModel.MOON,
            gravity_model=GravityPotentialModel.J2andJ3,
        )
        fm_dict = fm1.to_dict()
        fm2 = ForceModel.from_dict(fm_dict)
        assert fm1.drag == fm2.drag
        assert fm1.srp == fm2.srp
        assert fm1.nbody == fm2.nbody
        assert fm1.gravity == fm2.gravity
        assert fm1.atmosphere_model == fm2.atmosphere_model
        assert fm1.srp_model == fm2.srp_model
        assert fm1.nbody_model == fm2.nbody_model
        assert fm1.gravity_model == fm2.gravity_model


class TestForceModelComparison:
    """Tests for ForceModel comparison operators."""

    def test_equality_same_values(self):
        """Test equality for ForceModel with same values."""
        fm1 = ForceModel(drag=True, srp=True, nbody=False, gravity=True)
        fm2 = ForceModel(drag=True, srp=True, nbody=False, gravity=True)
        assert fm1 == fm2

    def test_equality_different_flags(self):
        """Test inequality for ForceModel with different flags."""
        fm1 = ForceModel(drag=True)
        fm2 = ForceModel(drag=False)
        assert not (fm1 == fm2)

    def test_equality_different_models(self):
        """Test inequality for ForceModel with different models."""
        fm1 = ForceModel(atmosphere_model=AtmosphereModel.EXPONENTIAL)
        fm2 = ForceModel(atmosphere_model=AtmosphereModel.COESA76)
        assert not (fm1 == fm2)

    def test_inequality_different_values(self):
        """Test != operator for different values."""
        fm1 = ForceModel(drag=True)
        fm2 = ForceModel(drag=False)
        assert fm1 != fm2

    def test_inequality_same_values(self):
        """Test != operator for same values."""
        fm1 = ForceModel(drag=True, srp=True)
        fm2 = ForceModel(drag=True, srp=True)
        assert not (fm1 != fm2)

    def test_equality_wrong_type(self):
        """Test equality raises NotImplementedError for wrong type."""
        fm = ForceModel()
        with pytest.raises(NotImplementedError, match="Comparison is not defined"):
            fm == "not a ForceModel"

    def test_inequality_wrong_type(self):
        """Test inequality raises NotImplementedError for wrong type."""
        fm = ForceModel()
        with pytest.raises(NotImplementedError, match="Comparison is not defined"):
            fm != 123


class TestForceModelStringRepresentation:
    """Tests for ForceModel string representations."""

    def test_str_two_body(self):
        """Test __str__ for two-body ForceModel."""
        fm = ForceModel()
        str_repr = str(fm)
        assert "TWO-BODY" in str_repr

    def test_str_with_drag(self):
        """Test __str__ includes drag information."""
        fm = ForceModel(drag=True)
        str_repr = str(fm)
        assert "Drag=" in str_repr
        assert "EXPONENTIAL" in str_repr

    def test_str_with_srp(self):
        """Test __str__ includes SRP information."""
        fm = ForceModel(srp=True)
        str_repr = str(fm)
        assert "SRP=" in str_repr
        assert "SUN" in str_repr

    def test_str_with_nbody(self):
        """Test __str__ includes NBody information."""
        fm = ForceModel(nbody=True)
        str_repr = str(fm)
        assert "NBody=" in str_repr
        assert "SUNandMOON" in str_repr

    def test_str_with_gravity(self):
        """Test __str__ includes Gravity information."""
        fm = ForceModel(gravity=True)
        str_repr = str(fm)
        assert "Gravity=" in str_repr
        assert "J2andJ3" in str_repr

    def test_str_with_multiple_perturbations(self):
        """Test __str__ with multiple perturbations."""
        fm = ForceModel(drag=True, srp=True, nbody=True, gravity=True)
        str_repr = str(fm)
        assert "Drag=" in str_repr
        assert "SRP=" in str_repr
        assert "NBody=" in str_repr
        assert "Gravity=" in str_repr

    def test_repr_representation(self):
        """Test __repr__ returns valid representation."""
        fm = ForceModel(drag=True, srp=False, nbody=True, gravity=False)
        repr_str = repr(fm)
        assert "ForceModel" in repr_str
        assert "drag=True" in repr_str
        assert "srp=False" in repr_str
        assert "nbody=True" in repr_str
        assert "gravity=False" in repr_str
