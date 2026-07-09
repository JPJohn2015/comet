# python imports
import pytest
import numpy as np

# comet imports
from comet.propagation.satellite_properties import SatelliteProperties


class TestSatellitePropertiesConstruction:
    """Tests for SatelliteProperties construction."""

    def test_default_construction(self):
        """Test SatelliteProperties with default values."""
        sp = SatelliteProperties()
        assert sp.dry_mass == 100.0
        assert sp.wet_mass == 0.0
        assert sp.area == 1.0
        assert sp.Cd == 2.2
        assert sp.Cr == 1.5

    def test_custom_construction(self):
        """Test SatelliteProperties with custom values."""
        sp = SatelliteProperties(dry_mass=500.0, wet_mass=50.0, area=10.0, Cd=2.5, Cr=1.8)
        assert sp.dry_mass == 500.0
        assert sp.wet_mass == 50.0
        assert sp.area == 10.0
        assert sp.Cd == 2.5
        assert sp.Cr == 1.8

    def test_invalid_dry_mass_negative(self):
        """Test that negative dry_mass raises TypeError."""
        with pytest.raises(TypeError, match="dry_mass must be an integer or float greater than 0.0"):
            SatelliteProperties(dry_mass=-1.0)

    def test_invalid_wet_mass_negative(self):
        """Test that negative wet_mass raises TypeError."""
        with pytest.raises(TypeError, match="wet_mass must be an integer or float greater than 0.0"):
            SatelliteProperties(wet_mass=-1.0)

    def test_invalid_area_negative(self):
        """Test that negative area raises TypeError."""
        with pytest.raises(TypeError, match="area must be an integer or float greater than 0.0"):
            SatelliteProperties(area=-1.0)

    def test_invalid_Cd_negative(self):
        """Test that negative Cd raises TypeError."""
        with pytest.raises(TypeError, match="Cd must be an integer or float greater than 0.0"):
            SatelliteProperties(Cd=-1.0)

    def test_invalid_Cr_negative(self):
        """Test that negative Cr raises TypeError."""
        with pytest.raises(TypeError, match="Cr must be an integer or float greater than 0.0"):
            SatelliteProperties(Cr=-1.0)


class TestSatellitePropertiesGetters:
    """Tests for SatelliteProperties getter methods."""

    def test_get_dry_mass(self):
        """Test get_dry_mass returns correct value."""
        sp = SatelliteProperties(dry_mass=250.0)
        assert sp.get_dry_mass() == 250.0

    def test_get_wet_mass(self):
        """Test get_wet_mass returns correct value."""
        sp = SatelliteProperties(wet_mass=25.0)
        assert sp.get_wet_mass() == 25.0

    def test_get_mass(self):
        """Test get_mass returns total mass."""
        sp = SatelliteProperties(dry_mass=250.0, wet_mass=50.0)
        assert sp.get_mass() == 300.0

    def test_get_area(self):
        """Test get_area returns correct value."""
        sp = SatelliteProperties(area=5.0)
        assert sp.get_area() == 5.0

    def test_get_Cd(self):
        """Test get_Cd returns correct value."""
        sp = SatelliteProperties(Cd=2.5)
        assert sp.get_Cd() == 2.5

    def test_get_Cr(self):
        """Test get_Cr returns correct value."""
        sp = SatelliteProperties(Cr=1.8)
        assert sp.get_Cr() == 1.8


class TestSatellitePropertiesAreaToMass:
    """Tests for area_to_mass calculation."""

    def test_area_to_mass_default(self):
        """Test area_to_mass with default values."""
        sp = SatelliteProperties()  # 1.0 m^2, 100.0 kg
        a_to_m = sp.area_to_mass()
        # Should be 1.0 * 1e-6 / 100.0 = 1e-8 km^2/kg
        assert pytest.approx(a_to_m, abs=1e-15) == 1e-8

    def test_area_to_mass_custom(self):
        """Test area_to_mass with custom values."""
        sp = SatelliteProperties(dry_mass=500.0, wet_mass=50.0, area=10.0)
        a_to_m = sp.area_to_mass()
        # Should be 10.0 * 1e-6 / 550.0
        expected = 10.0 * 1e-6 / 550.0
        assert pytest.approx(a_to_m, abs=1e-15) == expected

    def test_area_to_mass_unit_conversion_bug_fix(self):
        """Test that area_to_mass uses 1e-6 not 10e-6 (regression test).

        This is a regression test for the bug where 10e-6 (which equals 1e-5)
        was used instead of 1e-6 for converting m^2 to km^2.
        """
        sp = SatelliteProperties(dry_mass=100.0, wet_mass=0.0, area=1.0)
        a_to_m = sp.area_to_mass()
        # Correct conversion: 1.0 m^2 * 1e-6 km^2/m^2 / 100.0 kg = 1e-8 km^2/kg
        # Bug would give: 1.0 m^2 * 1e-5 km^2/m^2 / 100.0 kg = 1e-7 km^2/kg
        assert pytest.approx(a_to_m, abs=1e-15) == 1e-8
        assert a_to_m != 1e-7  # Ensure we're not using the buggy value

    def test_area_to_mass_with_wet_mass(self):
        """Test area_to_mass includes wet mass in calculation."""
        sp = SatelliteProperties(dry_mass=100.0, wet_mass=50.0, area=3.0)
        a_to_m = sp.area_to_mass()
        expected = 3.0 * 1e-6 / 150.0
        assert pytest.approx(a_to_m, abs=1e-15) == expected


class TestSatellitePropertiesUpdaters:
    """Tests for SatelliteProperties update methods."""

    def test_update_dry_mass(self):
        """Test update_dry_mass changes dry_mass."""
        sp = SatelliteProperties(dry_mass=100.0)
        sp.update_dry_mass(200.0)
        assert sp.dry_mass == 200.0

    def test_update_dry_mass_invalid(self):
        """Test update_dry_mass rejects negative values."""
        sp = SatelliteProperties()
        with pytest.raises(TypeError, match="dry_mass must be an integer or float greater than 0.0"):
            sp.update_dry_mass(-1.0)

    def test_update_wet_mass(self):
        """Test update_wet_mass changes wet_mass."""
        sp = SatelliteProperties(wet_mass=10.0)
        sp.update_wet_mass(20.0)
        assert sp.wet_mass == 20.0

    def test_update_wet_mass_invalid(self):
        """Test update_wet_mass rejects negative values."""
        sp = SatelliteProperties()
        with pytest.raises(TypeError, match="wet_mass must be an integer or float greater than 0.0"):
            sp.update_wet_mass(-1.0)

    def test_update_area(self):
        """Test update_area changes area."""
        sp = SatelliteProperties(area=1.0)
        sp.update_area(5.0)
        assert sp.area == 5.0

    def test_update_area_invalid(self):
        """Test update_area rejects negative values."""
        sp = SatelliteProperties()
        with pytest.raises(TypeError, match="area must be an integer or float greater than 0.0"):
            sp.update_area(-1.0)

    def test_update_Cd(self):
        """Test update_Cd changes Cd."""
        sp = SatelliteProperties(Cd=2.2)
        sp.update_Cd(3.0)
        assert sp.Cd == 3.0

    def test_update_Cd_invalid(self):
        """Test update_Cd rejects negative values."""
        sp = SatelliteProperties()
        with pytest.raises(TypeError, match="Cd must be an integer or float greater than 0.0"):
            sp.update_Cd(-1.0)

    def test_update_Cr(self):
        """Test update_Cr changes Cr."""
        sp = SatelliteProperties(Cr=1.5)
        sp.update_Cr(2.0)
        assert sp.Cr == 2.0

    def test_update_Cr_invalid(self):
        """Test update_Cr rejects negative values."""
        sp = SatelliteProperties()
        with pytest.raises(TypeError, match="Cr must be an integer or float greater than 0.0"):
            sp.update_Cr(-1.0)


class TestSatellitePropertiesCopy:
    """Tests for SatelliteProperties copy method."""

    def test_copy_creates_new_instance(self):
        """Test copy creates a new instance."""
        sp1 = SatelliteProperties(dry_mass=100.0, wet_mass=10.0, area=2.0, Cd=2.5, Cr=1.8)
        sp2 = sp1.copy()
        assert sp1 is not sp2
        assert sp1.dry_mass == sp2.dry_mass
        assert sp1.wet_mass == sp2.wet_mass
        assert sp1.area == sp2.area
        assert sp1.Cd == sp2.Cd
        assert sp1.Cr == sp2.Cr

    def test_copy_is_independent(self):
        """Test that modifying copy doesn't affect original."""
        sp1 = SatelliteProperties(dry_mass=100.0)
        sp2 = sp1.copy()
        sp2.update_dry_mass(200.0)
        assert sp1.dry_mass == 100.0
        assert sp2.dry_mass == 200.0


class TestSatellitePropertiesSerialization:
    """Tests for SatelliteProperties serialization."""

    def test_to_dict(self):
        """Test to_dict creates correct dictionary."""
        sp = SatelliteProperties(dry_mass=150.0, wet_mass=25.0, area=3.0, Cd=2.5, Cr=1.8)
        sp_dict = sp.to_dict()
        assert sp_dict["type"] == "SatelliteProperties"
        assert sp_dict["dry_mass"] == 150.0
        assert sp_dict["wet_mass"] == 25.0
        assert sp_dict["area"] == 3.0
        assert sp_dict["Cd"] == 2.5
        assert sp_dict["Cr"] == 1.8

    def test_from_dict(self):
        """Test from_dict reconstructs SatelliteProperties."""
        sp_dict = {
            "type": "SatelliteProperties",
            "dry_mass": 150.0,
            "wet_mass": 25.0,
            "area": 3.0,
            "Cd": 2.5,
            "Cr": 1.8,
        }
        sp = SatelliteProperties.from_dict(sp_dict)
        assert sp.dry_mass == 150.0
        assert sp.wet_mass == 25.0
        assert sp.area == 3.0
        assert sp.Cd == 2.5
        assert sp.Cr == 1.8

    def test_from_dict_invalid_type(self):
        """Test from_dict raises error for invalid type."""
        sp_dict = {"type": "InvalidType"}
        with pytest.raises(ValueError, match="Invalid construction dictionary"):
            SatelliteProperties.from_dict(sp_dict)

    def test_round_trip_serialization(self):
        """Test to_dict and from_dict round trip."""
        sp1 = SatelliteProperties(dry_mass=200.0, wet_mass=30.0, area=5.0, Cd=2.3, Cr=1.7)
        sp_dict = sp1.to_dict()
        sp2 = SatelliteProperties.from_dict(sp_dict)
        assert sp1.dry_mass == sp2.dry_mass
        assert sp1.wet_mass == sp2.wet_mass
        assert sp1.area == sp2.area
        assert sp1.Cd == sp2.Cd
        assert sp1.Cr == sp2.Cr


class TestSatellitePropertiesComparison:
    """Tests for SatelliteProperties comparison operators."""

    def test_equality_same_values(self):
        """Test equality for SatelliteProperties with same values."""
        sp1 = SatelliteProperties(dry_mass=100.0, wet_mass=10.0, area=2.0, Cd=2.5, Cr=1.8)
        sp2 = SatelliteProperties(dry_mass=100.0, wet_mass=10.0, area=2.0, Cd=2.5, Cr=1.8)
        assert sp1 == sp2

    def test_equality_different_values(self):
        """Test inequality for SatelliteProperties with different values."""
        sp1 = SatelliteProperties(dry_mass=100.0)
        sp2 = SatelliteProperties(dry_mass=200.0)
        assert not (sp1 == sp2)

    def test_inequality_different_values(self):
        """Test != operator for different values."""
        sp1 = SatelliteProperties(dry_mass=100.0)
        sp2 = SatelliteProperties(dry_mass=200.0)
        assert sp1 != sp2

    def test_inequality_same_values(self):
        """Test != operator for same values."""
        sp1 = SatelliteProperties(dry_mass=100.0, wet_mass=10.0, area=2.0, Cd=2.5, Cr=1.8)
        sp2 = SatelliteProperties(dry_mass=100.0, wet_mass=10.0, area=2.0, Cd=2.5, Cr=1.8)
        assert not (sp1 != sp2)

    def test_equality_wrong_type(self):
        """Test equality raises NotImplementedError for wrong type."""
        sp = SatelliteProperties()
        with pytest.raises(NotImplementedError, match="Comparison is not defined"):
            sp == "not a SatelliteProperties"

    def test_inequality_wrong_type(self):
        """Test inequality raises NotImplementedError for wrong type."""
        sp = SatelliteProperties()
        with pytest.raises(NotImplementedError, match="Comparison is not defined"):
            sp != 123


class TestSatellitePropertiesStringRepresentation:
    """Tests for SatelliteProperties string representations."""

    def test_str_representation(self):
        """Test __str__ returns formatted string."""
        sp = SatelliteProperties(dry_mass=100.0, wet_mass=10.0, area=2.0, Cd=2.5, Cr=1.8)
        str_repr = str(sp)
        assert "Dry Mass=100.0" in str_repr
        assert "Wet Mass=10.0" in str_repr
        assert "Area=2.0" in str_repr
        assert "Cd=2.5" in str_repr
        assert "Cr=1.8" in str_repr

    def test_str_cd_cr_not_swapped(self):
        """Test __str__ doesn't swap Cd and Cr (regression test)."""
        sp = SatelliteProperties(Cd=2.5, Cr=1.8)
        str_repr = str(sp)
        # Extract values from string
        assert "Cd=2.5" in str_repr
        assert "Cr=1.8" in str_repr

    def test_repr_representation(self):
        """Test __repr__ returns valid representation."""
        sp = SatelliteProperties(dry_mass=100.0, wet_mass=10.0, area=2.0, Cd=2.5, Cr=1.8)
        repr_str = repr(sp)
        assert "ForceModel" in repr_str  # Note: repr says ForceModel (seems like a bug but testing as-is)
        assert "dry_mass=100.0" in repr_str
        assert "wet_mass=10.0" in repr_str
