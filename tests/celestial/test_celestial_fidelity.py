# python imports
import pytest

# comet imports
from comet.celestial.celestial_fidelity import CelestialFidelity


class TestCelestialFidelity:
    """Tests for CelestialFidelity enum."""

    def test_celestial_fidelity_values(self):
        """Test CelestialFidelity has correct values."""
        assert CelestialFidelity.LoFi.value == 0
        assert CelestialFidelity.HiFi.value == 1

    def test_celestial_fidelity_membership(self):
        """Test CelestialFidelity membership."""
        assert CelestialFidelity.LoFi in CelestialFidelity
        assert CelestialFidelity.HiFi in CelestialFidelity

    def test_celestial_fidelity_count(self):
        """Test CelestialFidelity has exactly 2 members."""
        assert len(CelestialFidelity) == 2

    def test_celestial_fidelity_string_behavior(self):
        """Test CelestialFidelity string behavior."""
        assert str(CelestialFidelity.LoFi) == "CelestialFidelity.LoFi"
        assert str(CelestialFidelity.HiFi) == "CelestialFidelity.HiFi"

    def test_celestial_fidelity_equality(self):
        """Test CelestialFidelity equality comparisons."""
        assert CelestialFidelity.LoFi == CelestialFidelity.LoFi
        assert CelestialFidelity.HiFi == CelestialFidelity.HiFi
        assert CelestialFidelity.LoFi != CelestialFidelity.HiFi

    def test_celestial_fidelity_access_by_name(self):
        """Test CelestialFidelity can be accessed by name."""
        assert CelestialFidelity["LoFi"] == CelestialFidelity.LoFi
        assert CelestialFidelity["HiFi"] == CelestialFidelity.HiFi

    def test_celestial_fidelity_access_by_value(self):
        """Test CelestialFidelity can be accessed by value."""
        assert CelestialFidelity(0) == CelestialFidelity.LoFi
        assert CelestialFidelity(1) == CelestialFidelity.HiFi

    def test_celestial_fidelity_invalid_value(self):
        """Test CelestialFidelity raises error for invalid value."""
        with pytest.raises(ValueError):
            CelestialFidelity(2)

    def test_celestial_fidelity_invalid_name(self):
        """Test CelestialFidelity raises error for invalid name."""
        with pytest.raises(KeyError):
            CelestialFidelity["InvalidFidelity"]
