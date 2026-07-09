# python imports
import pytest
import numpy as np

# comet imports
from comet.propagation.dynamics import (
    two_body,
    third_body_acceleration,
    solar_pressure_acceleration,
)
from comet.utilities.constants import Constants as c


class TestTwoBodyDynamics:
    """Tests for two_body dynamics function."""

    def test_two_body_output_shape(self):
        """Test two_body returns correct output shape."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        dydt = two_body(t, y)
        assert dydt.shape == (6,)

    def test_two_body_velocity_part(self):
        """Test two_body velocity derivative equals velocity."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        dydt = two_body(t, y)
        # First three elements should be velocity
        np.testing.assert_array_equal(dydt[:3], y[3:])

    def test_two_body_acceleration_direction(self):
        """Test two_body acceleration points toward Earth center."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        dydt = two_body(t, y)
        acceleration = dydt[3:]
        position = y[:3]
        # Acceleration should be antiparallel to position (pointing toward center)
        # Check that acceleration dot position is negative
        assert np.dot(acceleration, position) < 0

    def test_two_body_acceleration_magnitude(self):
        """Test two_body acceleration magnitude follows inverse square law."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        dydt = two_body(t, y)
        acceleration = dydt[3:]
        r = np.linalg.norm(y[:3])
        expected_mag = c.MU_EARTH / (r**2)
        actual_mag = np.linalg.norm(acceleration)
        assert pytest.approx(actual_mag, rel=1e-10) == expected_mag

    def test_two_body_circular_orbit(self):
        """Test two_body for circular orbit conditions."""
        # For circular orbit at 7000 km, v = sqrt(mu/r)
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        t = 0.0
        y = np.array([r, 0.0, 0.0, 0.0, v, 0.0])
        dydt = two_body(t, y)
        # Acceleration should be perpendicular to velocity for circular orbit
        acceleration = dydt[3:]
        velocity = y[3:]
        # Dot product should be close to zero
        assert pytest.approx(np.dot(acceleration, velocity), abs=1e-6) == 0.0

    def test_two_body_independent_of_time(self):
        """Test two_body gives same result for different times (autonomous system)."""
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        dydt1 = two_body(0.0, y)
        dydt2 = two_body(100.0, y)
        dydt3 = two_body(1000.0, y)
        np.testing.assert_array_equal(dydt1, dydt2)
        np.testing.assert_array_equal(dydt1, dydt3)

    def test_two_body_different_positions(self):
        """Test two_body produces different results for different positions."""
        t = 0.0
        y1 = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        y2 = np.array([8000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        dydt1 = two_body(t, y1)
        dydt2 = two_body(t, y2)
        # Accelerations should differ
        assert not np.allclose(dydt1[3:], dydt2[3:])


class TestThirdBodyAcceleration:
    """Tests for third_body_acceleration function."""

    def test_third_body_output_shape(self):
        """Test third_body_acceleration returns three components."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        r_third = np.array([150e6, 0.0, 0.0])  # Sun-like distance
        ax, ay, az = third_body_acceleration(t, y, c.MU_SUN, r_third)
        assert isinstance(ax, (float, np.floating))
        assert isinstance(ay, (float, np.floating))
        assert isinstance(az, (float, np.floating))

    def test_third_body_sun_acceleration_magnitude(self):
        """Test third_body_acceleration for Sun has reasonable magnitude."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        r_sun = np.array([c.AU, 0.0, 0.0])
        ax, ay, az = third_body_acceleration(t, y, c.MU_SUN, r_sun)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        # Solar third-body perturbation at LEO when Sun at 1 AU is ~1e-9 km/s^2
        assert 1e-11 < accel_mag < 1e-7

    def test_third_body_moon_acceleration_magnitude(self):
        """Test third_body_acceleration for Moon has reasonable magnitude."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        r_moon = np.array([384400.0, 0.0, 0.0])  # Moon distance
        ax, ay, az = third_body_acceleration(t, y, c.MU_MOON, r_moon)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        # Lunar third-body perturbation at LEO is ~1e-9 km/s^2
        assert 1e-11 < accel_mag < 1e-7

    def test_third_body_opposite_side(self):
        """Test third_body_acceleration when satellite is on opposite side of Earth."""
        t = 0.0
        # Satellite on one side of Earth
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        # Third body on opposite side
        r_third = np.array([-150e6, 0.0, 0.0])
        ax, ay, az = third_body_acceleration(t, y, c.MU_SUN, r_third)
        accel = np.array([ax, ay, az])
        # Acceleration should generally point away from Earth (positive x)
        assert accel[0] > 0

    def test_third_body_formula_correctness(self):
        """Test third_body_acceleration follows correct formula.

        a = mu_3 * ((r_3 - r_sat) / |r_3 - r_sat|^3 - r_3 / |r_3|^3)
        """
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        r_third = np.array([150e6, 0.0, 0.0])

        ax, ay, az = third_body_acceleration(t, y, c.MU_SUN, r_third)
        accel = np.array([ax, ay, az])

        # Manually calculate expected acceleration
        sat_to_3rd = r_third - y[:3]
        expected = c.MU_SUN * (
            (sat_to_3rd / (np.linalg.norm(sat_to_3rd) ** 3))
            - (r_third / (np.linalg.norm(r_third) ** 3))
        )

        np.testing.assert_allclose(accel, expected, rtol=1e-12)


class TestSolarPressureAcceleration:
    """Tests for solar_pressure_acceleration function."""

    def test_solar_pressure_output_shape(self):
        """Test solar_pressure_acceleration returns three components."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        Cr = 1.5
        a_to_m = 1e-8  # km^2/kg
        r_sun = np.array([c.AU, 0.0, 0.0])
        ax, ay, az = solar_pressure_acceleration(t, y, Cr, a_to_m, r_sun)
        assert isinstance(ax, (float, np.floating))
        assert isinstance(ay, (float, np.floating))
        assert isinstance(az, (float, np.floating))

    def test_solar_pressure_direction(self):
        """Test solar_pressure_acceleration points away from Sun."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        Cr = 1.5
        a_to_m = 1e-8
        r_sun = np.array([c.AU, 0.0, 0.0])  # Sun at +X
        ax, ay, az = solar_pressure_acceleration(t, y, Cr, a_to_m, r_sun)
        accel = np.array([ax, ay, az])
        sat_to_sun = r_sun - y[:3]
        # Acceleration should be antiparallel to sat-to-sun vector (away from sun)
        # Because of negative sign in formula
        dot_product = np.dot(accel, sat_to_sun)
        assert dot_product < 0

    def test_solar_pressure_magnitude(self):
        """Test solar_pressure_acceleration has reasonable magnitude."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        Cr = 1.5
        a_to_m = 1e-8  # km^2/kg
        r_sun = np.array([c.AU, 0.0, 0.0])
        ax, ay, az = solar_pressure_acceleration(t, y, Cr, a_to_m, r_sun)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        # SRP acceleration with small area-to-mass is very small (~1e-13 km/s^2)
        assert 1e-15 < accel_mag < 1e-10

    def test_solar_pressure_proportional_to_Cr(self):
        """Test solar_pressure_acceleration is proportional to Cr."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        a_to_m = 1e-8
        r_sun = np.array([c.AU, 0.0, 0.0])

        Cr1 = 1.0
        ax1, ay1, az1 = solar_pressure_acceleration(t, y, Cr1, a_to_m, r_sun)
        accel1 = np.array([ax1, ay1, az1])

        Cr2 = 2.0
        ax2, ay2, az2 = solar_pressure_acceleration(t, y, Cr2, a_to_m, r_sun)
        accel2 = np.array([ax2, ay2, az2])

        # Doubling Cr should double acceleration
        np.testing.assert_allclose(accel2, 2.0 * accel1, rtol=1e-10)

    def test_solar_pressure_proportional_to_area_to_mass(self):
        """Test solar_pressure_acceleration is proportional to area-to-mass ratio."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        Cr = 1.5
        r_sun = np.array([c.AU, 0.0, 0.0])

        a_to_m1 = 1e-8
        ax1, ay1, az1 = solar_pressure_acceleration(t, y, Cr, a_to_m1, r_sun)
        accel1 = np.array([ax1, ay1, az1])

        a_to_m2 = 2e-8
        ax2, ay2, az2 = solar_pressure_acceleration(t, y, Cr, a_to_m2, r_sun)
        accel2 = np.array([ax2, ay2, az2])

        # Doubling area-to-mass should double acceleration
        np.testing.assert_allclose(accel2, 2.0 * accel1, rtol=1e-10)

    def test_solar_pressure_formula_correctness(self):
        """Test solar_pressure_acceleration follows correct formula.

        a = -P_sun * Cr * (A/m) * (sat_to_sun / |sat_to_sun|)
        """
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        Cr = 1.5
        a_to_m = 1e-8
        r_sun = np.array([c.AU, 0.0, 0.0])

        ax, ay, az = solar_pressure_acceleration(t, y, Cr, a_to_m, r_sun)
        accel = np.array([ax, ay, az])

        # Manually calculate expected acceleration
        sat_to_sun = r_sun - y[:3]
        expected = (-c.SOLAR_PRESSURE * Cr * a_to_m) * (sat_to_sun / np.linalg.norm(sat_to_sun))

        np.testing.assert_allclose(accel, expected, rtol=1e-12)


class TestDynamicsEdgeCases:
    """Tests for edge cases in dynamics functions."""

    def test_two_body_at_different_radii(self):
        """Test two_body works at various orbital radii."""
        t = 0.0
        # LEO
        y_leo = np.array([6678.0, 0.0, 0.0, 0.0, 7.7, 0.0])
        dydt_leo = two_body(t, y_leo)
        assert dydt_leo.shape == (6,)

        # GEO
        y_geo = np.array([42164.0, 0.0, 0.0, 0.0, 3.1, 0.0])
        dydt_geo = two_body(t, y_geo)
        assert dydt_geo.shape == (6,)

        # Moon orbit
        y_lunar = np.array([384400.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        dydt_lunar = two_body(t, y_lunar)
        assert dydt_lunar.shape == (6,)

    def test_third_body_with_zero_mu(self):
        """Test third_body_acceleration with zero gravitational parameter."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        r_third = np.array([150e6, 0.0, 0.0])
        ax, ay, az = third_body_acceleration(t, y, 0.0, r_third)
        assert ax == 0.0
        assert ay == 0.0
        assert az == 0.0

    def test_solar_pressure_with_zero_coefficients(self):
        """Test solar_pressure_acceleration with zero Cr or area-to-mass."""
        t = 0.0
        y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        r_sun = np.array([c.AU, 0.0, 0.0])

        # Zero Cr
        ax, ay, az = solar_pressure_acceleration(t, y, 0.0, 1e-8, r_sun)
        assert ax == 0.0
        assert ay == 0.0
        assert az == 0.0

        # Zero area-to-mass
        ax, ay, az = solar_pressure_acceleration(t, y, 1.5, 0.0, r_sun)
        assert ax == 0.0
        assert ay == 0.0
        assert az == 0.0
