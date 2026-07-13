# python imports
import pytest
import numpy as np

# COMET imports
from comet.components.thruster import Thruster
from comet.components.component import Component
from comet.time.duration import Duration
from comet.utilities.constants import Constants as c


class TestThrusterConstruction:
    """Test Thruster construction."""

    def test_thruster_default(self):
        """Test Thruster with default parameters."""
        thruster = Thruster()
        assert thruster.thrust == 10.0
        assert thruster.isp == 200.0
        assert thruster.mass == 0.0
        assert np.allclose(thruster.body_vector, [1, 0, 0])
        assert thruster.id is not None

    def test_thruster_with_thrust(self):
        """Test Thruster with custom thrust."""
        thruster = Thruster(thrust=50.0)
        assert thruster.thrust == 50.0
        assert thruster.isp == 200.0

    def test_thruster_with_isp(self):
        """Test Thruster with custom ISP."""
        thruster = Thruster(isp=300.0)
        assert thruster.thrust == 10.0
        assert thruster.isp == 300.0

    def test_thruster_with_all_parameters(self):
        """Test Thruster with all custom parameters."""
        body_vec = [0, 1, 0]
        thruster = Thruster(thrust=100.0, isp=250.0, mass=15.0, body_vector=body_vec)
        assert thruster.thrust == 100.0
        assert thruster.isp == 250.0
        assert thruster.mass == 15.0
        assert np.allclose(thruster.body_vector, body_vec)

    def test_thruster_inherits_from_component(self):
        """Test that Thruster inherits from Component."""
        thruster = Thruster()
        assert isinstance(thruster, Component)
        assert hasattr(thruster, 'get_mass')
        assert hasattr(thruster, 'get_body_vector')
        assert hasattr(thruster, 'get_id')


class TestThrusterCopy:
    """Test Thruster copy method."""

    def test_copy_creates_new_instance(self):
        """Test copy creates a new Thruster instance."""
        thruster1 = Thruster(thrust=50.0, isp=300.0)
        thruster2 = thruster1.copy()
        assert thruster1 is not thruster2

    def test_copy_preserves_properties(self):
        """Test copy preserves thruster properties."""
        thruster1 = Thruster(thrust=50.0, isp=300.0, mass=10.0, body_vector=[0, 0, 1])
        thruster2 = thruster1.copy()
        assert thruster2.thrust == thruster1.thrust
        assert thruster2.isp == thruster1.isp
        assert thruster2.mass == thruster1.mass
        assert np.allclose(thruster2.body_vector, thruster1.body_vector)

    def test_copy_increments_id(self):
        """Test copy creates new ID (doesn't copy ID)."""
        thruster1 = Thruster()
        thruster2 = thruster1.copy()
        assert thruster2.id != thruster1.id


class TestThrusterGetters:
    """Test Thruster getter methods."""

    def test_get_thrust(self):
        """Test get_thrust method."""
        thruster = Thruster(thrust=75.0)
        assert thruster.get_thrust() == 75.0

    def test_get_isp(self):
        """Test get_isp method."""
        thruster = Thruster(isp=350.0)
        assert thruster.get_isp() == 350.0


class TestThrusterMassFlowRate:
    """Test Thruster mass_flow_rate method."""

    def test_mass_flow_rate_default(self):
        """Test mass flow rate with default parameters."""
        thruster = Thruster()
        dmdt = thruster.mass_flow_rate()
        expected = -thruster.thrust / (thruster.isp * c.SURFACE_GRAVITY)
        assert pytest.approx(dmdt, abs=1e-10) == expected

    def test_mass_flow_rate_negative(self):
        """Test mass flow rate is negative (mass is lost)."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        dmdt = thruster.mass_flow_rate()
        assert dmdt < 0

    def test_mass_flow_rate_high_isp(self):
        """Test higher ISP gives lower mass flow rate."""
        thruster1 = Thruster(thrust=100.0, isp=200.0)
        thruster2 = Thruster(thrust=100.0, isp=400.0)
        assert abs(thruster2.mass_flow_rate()) < abs(thruster1.mass_flow_rate())

    def test_mass_flow_rate_high_thrust(self):
        """Test higher thrust gives higher mass flow rate."""
        thruster1 = Thruster(thrust=50.0, isp=200.0)
        thruster2 = Thruster(thrust=100.0, isp=200.0)
        assert abs(thruster2.mass_flow_rate()) > abs(thruster1.mass_flow_rate())


class TestThrusterAcceleration:
    """Test Thruster acceleration method."""

    def test_acceleration_basic(self):
        """Test acceleration calculation."""
        thruster = Thruster(thrust=100.0)
        mass = 1000.0
        accel = thruster.acceleration(mass)
        expected = thruster.thrust / mass
        assert pytest.approx(accel, abs=1e-10) == expected

    def test_acceleration_higher_mass_lower_accel(self):
        """Test higher mass gives lower acceleration."""
        thruster = Thruster(thrust=100.0)
        accel1 = thruster.acceleration(1000.0)
        accel2 = thruster.acceleration(2000.0)
        assert accel2 < accel1

    def test_acceleration_units(self):
        """Test acceleration returns m/s^2."""
        thruster = Thruster(thrust=100.0)  # N
        mass = 1000.0  # kg
        accel = thruster.acceleration(mass)
        # 100 N / 1000 kg = 0.1 m/s^2
        assert pytest.approx(accel, abs=1e-10) == 0.1


class TestThrusterBurnGivenDv:
    """Test Thruster burn_given_dv method."""

    def test_burn_given_dv_scalar(self):
        """Test burn calculation with scalar delta-V."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        mass = 1000.0
        dv = 0.1  # km/s
        burn_time, mass_expelled = thruster.burn_given_dv(mass, dv)

        assert burn_time > 0
        assert mass_expelled < 0  # Mass is lost

    def test_burn_given_dv_vector(self):
        """Test burn calculation with vector delta-V."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        mass = 1000.0
        dv_vec = np.array([0.06, 0.08, 0.0])  # Magnitude = 0.1 km/s
        burn_time, mass_expelled = thruster.burn_given_dv(mass, dv_vec)

        assert burn_time > 0
        assert mass_expelled < 0

    def test_burn_given_dv_list(self):
        """Test burn calculation with list delta-V."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        mass = 1000.0
        dv_list = [0.06, 0.08, 0.0]  # Magnitude = 0.1 km/s
        burn_time, mass_expelled = thruster.burn_given_dv(mass, dv_list)

        assert burn_time > 0
        assert mass_expelled < 0

    def test_burn_given_dv_mass_conservation(self):
        """Test that mass expelled matches burn time."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        mass = 1000.0
        dv = 0.1  # km/s
        burn_time, mass_expelled = thruster.burn_given_dv(mass, dv)

        # mass_expelled should equal dmdt * burn_time
        dmdt = thruster.mass_flow_rate()
        expected_mass = dmdt * burn_time
        assert pytest.approx(mass_expelled, abs=1e-10) == expected_mass

    def test_burn_given_dv_higher_dv(self):
        """Test higher delta-V requires longer burn."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        mass = 1000.0

        burn_time1, _ = thruster.burn_given_dv(mass, 0.05)
        burn_time2, _ = thruster.burn_given_dv(mass, 0.10)

        assert burn_time2 > burn_time1


class TestThrusterBurnGivenTime:
    """Test Thruster burn_given_time method."""

    def test_burn_given_time_mass_conservation(self):
        """Test that mass expelled matches burn time."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        mass = 1000.0
        duration = 10.0  # short burn to avoid numerical issues
        dv, mass_expelled = thruster.burn_given_time(mass, duration)

        # mass_expelled should equal dmdt * duration
        dmdt = thruster.mass_flow_rate()
        expected_mass = dmdt * duration
        assert pytest.approx(mass_expelled, abs=1e-10) == expected_mass

    def test_burn_given_time_duration_object(self):
        """Test burn calculation accepts Duration object."""
        thruster = Thruster(thrust=100.0, isp=200.0)
        mass = 1000.0
        duration = Duration(seconds=10)
        dv, mass_expelled = thruster.burn_given_time(mass, duration)

        # Should return valid results
        assert isinstance(dv, (int, float))
        assert mass_expelled < 0  # Mass is lost


class TestThrusterEdgeCases:
    """Test Thruster edge cases."""

    def test_zero_thrust(self):
        """Test Thruster with zero thrust."""
        thruster = Thruster(thrust=0.0)
        assert thruster.mass_flow_rate() == 0.0
        assert thruster.acceleration(1000.0) == 0.0

    def test_very_high_isp(self):
        """Test Thruster with very high ISP."""
        thruster = Thruster(thrust=100.0, isp=10000.0)
        dmdt = thruster.mass_flow_rate()
        # High ISP means low mass flow rate (adjust tolerance)
        assert abs(dmdt) < 0.002

    def test_very_low_mass(self):
        """Test acceleration with very low mass."""
        thruster = Thruster(thrust=100.0)
        # Very low mass should give very high acceleration
        accel = thruster.acceleration(1.0)
        assert accel == 100.0  # m/s^2
