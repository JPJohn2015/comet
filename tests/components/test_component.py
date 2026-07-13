# python imports
import pytest
import numpy as np

# COMET imports
from comet.components.component import Component


class TestComponentConstruction:
    """Test Component construction."""

    def test_component_default(self):
        """Test Component with default parameters."""
        comp = Component()
        assert comp.mass == 0.0
        assert np.allclose(comp.body_vector, [1, 0, 0])
        assert comp.id is not None

    def test_component_with_mass(self):
        """Test Component with custom mass."""
        comp = Component(mass=10.5)
        assert comp.mass == 10.5
        assert np.allclose(comp.body_vector, [1, 0, 0])

    def test_component_with_body_vector(self):
        """Test Component with custom body vector."""
        body_vec = np.array([0, 1, 0])
        comp = Component(body_vector=body_vec)
        assert comp.mass == 0.0
        assert np.allclose(comp.body_vector, body_vec)

    def test_component_with_all_parameters(self):
        """Test Component with all custom parameters."""
        body_vec = np.array([0, 0, 1])
        comp = Component(mass=25.0, body_vector=body_vec)
        assert comp.mass == 25.0
        assert np.allclose(comp.body_vector, body_vec)

    def test_component_unique_ids(self):
        """Test that each Component gets a unique ID."""
        comp1 = Component()
        comp2 = Component()
        comp3 = Component()
        assert comp1.id != comp2.id
        assert comp2.id != comp3.id
        assert comp1.id != comp3.id


class TestComponentGetMass:
    """Test Component get_mass method."""

    def test_get_mass_default(self):
        """Test get_mass with default mass."""
        comp = Component()
        assert comp.get_mass() == 0.0

    def test_get_mass_custom(self):
        """Test get_mass with custom mass."""
        comp = Component(mass=42.3)
        assert comp.get_mass() == 42.3

    def test_get_mass_returns_float(self):
        """Test get_mass returns expected type."""
        comp = Component(mass=10)
        mass = comp.get_mass()
        assert isinstance(mass, (int, float))


class TestComponentGetBodyVector:
    """Test Component get_body_vector method."""

    def test_get_body_vector_default(self):
        """Test get_body_vector with default vector."""
        comp = Component()
        vec = comp.get_body_vector()
        assert np.allclose(vec, [1, 0, 0])

    def test_get_body_vector_custom(self):
        """Test get_body_vector with custom vector."""
        body_vec = np.array([0.577, 0.577, 0.577])  # Normalized diagonal
        comp = Component(body_vector=body_vec)
        vec = comp.get_body_vector()
        assert np.allclose(vec, body_vec)

    def test_get_body_vector_list_input(self):
        """Test get_body_vector works with list input."""
        body_vec = [0, 1, 0]
        comp = Component(body_vector=body_vec)
        vec = comp.get_body_vector()
        assert np.allclose(vec, body_vec)


class TestComponentGetId:
    """Test Component get_id method."""

    def test_get_id_returns_int(self):
        """Test get_id returns an integer."""
        comp = Component()
        comp_id = comp.get_id()
        assert isinstance(comp_id, int)

    def test_get_id_matches_attribute(self):
        """Test get_id matches the id attribute."""
        comp = Component()
        assert comp.get_id() == comp.id

    def test_get_id_unique_across_instances(self):
        """Test get_id returns unique values for different instances."""
        comp1 = Component()
        comp2 = Component()
        assert comp1.get_id() != comp2.get_id()


class TestComponentEdgeCases:
    """Test Component edge cases."""

    def test_component_negative_mass(self):
        """Test Component with negative mass (physically invalid but not rejected)."""
        comp = Component(mass=-5.0)
        assert comp.mass == -5.0

    def test_component_zero_body_vector(self):
        """Test Component with zero body vector."""
        comp = Component(body_vector=[0, 0, 0])
        assert np.allclose(comp.body_vector, [0, 0, 0])

    def test_component_large_mass(self):
        """Test Component with very large mass."""
        comp = Component(mass=1e6)
        assert comp.mass == 1e6

    def test_component_body_vector_not_normalized(self):
        """Test Component accepts non-normalized body vectors."""
        body_vec = [10, 20, 30]
        comp = Component(body_vector=body_vec)
        assert np.allclose(comp.body_vector, body_vec)


class TestComponentMultipleInstances:
    """Test multiple Component instances."""

    def test_multiple_components_independent(self):
        """Test that multiple Components are independent."""
        comp1 = Component(mass=10.0, body_vector=[1, 0, 0])
        comp2 = Component(mass=20.0, body_vector=[0, 1, 0])

        assert comp1.mass != comp2.mass
        assert not np.allclose(comp1.body_vector, comp2.body_vector)
        assert comp1.id != comp2.id

    def test_component_modification_doesnt_affect_others(self):
        """Test modifying one Component doesn't affect others."""
        comp1 = Component(mass=10.0)
        comp2 = Component(mass=10.0)

        comp1.mass = 20.0
        assert comp2.mass == 10.0
