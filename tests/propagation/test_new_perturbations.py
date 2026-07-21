# python imports
import pytest
import numpy as np

# comet imports
from comet.propagation.propagator import Propagator
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.propagation.perturbation_model import AtmosphereModel, SRPModel
from comet.propagation import dynamics
from comet.state import State
from comet.time import Epoch
from comet.time import Duration
from comet.utilities.constants import Constants as c
from comet.celestial import Sun


class TestExponentialDragModel:
    """Tests for exponential atmospheric drag model."""

    def test_exponential_drag_causes_orbit_decay(self):
        """Test that exponential drag causes measurable orbit decay in LEO.

        A satellite in LEO with realistic drag properties should experience
        measurable altitude decay over several days.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # LEO orbit at 400 km altitude
        r = c.RADIUS_EARTH + 400.0  # km
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])

        # Satellite with realistic drag properties (CubeSat-like)
        sat_props = SatelliteProperties(
            dry_mass=10.0,  # kg
            area=0.01,  # m² = 0.01e-6 km² (10 cm x 10 cm)
            Cd=2.2,  # Typical drag coefficient
        )

        fm = ForceModel(drag=True, atmosphere_model=AtmosphereModel.EXPONENTIAL)
        prop = Propagator(epoch, state, force_model=fm, sat_properties=sat_props)

        # Propagate for 7 days
        final_epoch = epoch + Duration(days=7)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        pos_final = final_state.position()
        r_final = np.linalg.norm(pos_final)

        # Altitude should have decreased due to drag
        altitude_initial = r - c.RADIUS_EARTH
        altitude_final = r_final - c.RADIUS_EARTH

        altitude_loss = altitude_initial - altitude_final
        assert altitude_loss > 0.0, "Drag should cause altitude decay"
        # Typical LEO decay is 0.1-10 km/week depending on area/mass and solar activity
        assert altitude_loss > 0.01, "Decay should be measurable (> 10 m)"
        assert altitude_loss < 100.0, "Decay should be realistic (< 100 km/week)"

    def test_exponential_drag_acceleration_magnitude(self):
        """Test that exponential drag produces reasonable acceleration magnitude."""
        # LEO orbit at 400 km
        r = c.RADIUS_EARTH + 400.0
        v_circ = np.sqrt(c.MU_EARTH / r)
        y = np.array([r, 0.0, 0.0, 0.0, v_circ, 0.0])

        # Realistic satellite properties
        Cd = 2.2
        area_to_mass = 0.01e-6 / 10.0  # 0.01 m² / 10 kg in km²/kg

        ax, ay, az = dynamics.atmospheric_drag_exponential(0.0, y, Cd, area_to_mass)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

        # At 400 km with small area-to-mass, drag acceleration is ~1e-10 km/s²
        assert accel_mag > 1e-12, "Drag acceleration should be measurable"
        assert accel_mag < 1e-6, "Drag acceleration should be realistic"

    def test_exponential_drag_opposes_velocity(self):
        """Test that exponential drag acceleration opposes velocity direction."""
        r = c.RADIUS_EARTH + 400.0
        v_circ = np.sqrt(c.MU_EARTH / r)
        y = np.array([r, 0.0, 0.0, 0.0, v_circ, 0.0])

        Cd = 2.2
        area_to_mass = 0.01e-6 / 10.0

        ax, ay, az = dynamics.atmospheric_drag_exponential(0.0, y, Cd, area_to_mass)
        accel = np.array([ax, ay, az])

        # Account for rotating atmosphere
        omega_cross_r = np.array([
            -c.OMEGA_EARTH * y[1],
            c.OMEGA_EARTH * y[0],
            0.0
        ])
        v_rel = y[3:] - omega_cross_r

        # Drag should be antiparallel to relative velocity
        dot_product = np.dot(accel, v_rel)
        assert dot_product < 0.0, "Drag should oppose motion"

    def test_exponential_drag_increases_with_lower_altitude(self):
        """Test that drag increases at lower altitudes (higher density)."""
        Cd = 2.2
        area_to_mass = 0.01e-6 / 10.0
        v_circ_base = 7.5  # Approximate LEO velocity

        # Test at 300 km and 500 km
        r_low = c.RADIUS_EARTH + 300.0
        r_high = c.RADIUS_EARTH + 500.0

        y_low = np.array([r_low, 0.0, 0.0, 0.0, v_circ_base, 0.0])
        y_high = np.array([r_high, 0.0, 0.0, 0.0, v_circ_base, 0.0])

        ax_low, ay_low, az_low = dynamics.atmospheric_drag_exponential(0.0, y_low, Cd, area_to_mass)
        ax_high, ay_high, az_high = dynamics.atmospheric_drag_exponential(0.0, y_high, Cd, area_to_mass)

        accel_mag_low = np.sqrt(ax_low**2 + ay_low**2 + az_low**2)
        accel_mag_high = np.sqrt(ax_high**2 + ay_high**2 + az_high**2)

        assert accel_mag_low > accel_mag_high, "Lower altitude should have higher drag"


class TestCOESA76DragModel:
    """Tests for COESA76 atmospheric drag model."""

    def test_coesa76_drag_causes_orbit_decay(self):
        """Test that COESA76 drag causes orbit decay."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = c.RADIUS_EARTH + 400.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])

        sat_props = SatelliteProperties(
            dry_mass=10.0,
            area=0.01,
            Cd=2.2,
        )

        fm = ForceModel(drag=True, atmosphere_model=AtmosphereModel.COESA76)
        prop = Propagator(epoch, state, force_model=fm, sat_properties=sat_props)

        final_epoch = epoch + Duration(days=7)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        pos_final = final_state.position()
        r_final = np.linalg.norm(pos_final)

        altitude_initial = r - c.RADIUS_EARTH
        altitude_final = r_final - c.RADIUS_EARTH
        altitude_loss = altitude_initial - altitude_final

        assert altitude_loss > 0.0, "COESA76 drag should cause altitude decay"
        assert altitude_loss > 0.01, "Decay should be measurable"
        assert altitude_loss < 100.0, "Decay should be realistic"

    def test_coesa76_differs_from_exponential(self):
        """Test that COESA76 produces different results than exponential model.

        The two models use different density formulations, so they should
        produce measurably different accelerations at the same altitude.
        """
        r = c.RADIUS_EARTH + 400.0
        v_circ = np.sqrt(c.MU_EARTH / r)
        y = np.array([r, 0.0, 0.0, 0.0, v_circ, 0.0])

        Cd = 2.2
        area_to_mass = 0.01e-6 / 10.0

        # Get accelerations from both models
        ax_exp, ay_exp, az_exp = dynamics.atmospheric_drag_exponential(0.0, y, Cd, area_to_mass)
        ax_c76, ay_c76, az_c76 = dynamics.atmospheric_drag_coesa76(0.0, y, Cd, area_to_mass)

        accel_exp = np.sqrt(ax_exp**2 + ay_exp**2 + az_exp**2)
        accel_c76 = np.sqrt(ax_c76**2 + ay_c76**2 + az_c76**2)

        # Models should produce different results (but same order of magnitude)
        relative_diff = abs(accel_exp - accel_c76) / accel_exp
        assert relative_diff > 0.01, "Models should differ by at least 1%"

    def test_coesa76_drag_acceleration_magnitude(self):
        """Test that COESA76 drag produces reasonable acceleration magnitude."""
        r = c.RADIUS_EARTH + 400.0
        v_circ = np.sqrt(c.MU_EARTH / r)
        y = np.array([r, 0.0, 0.0, 0.0, v_circ, 0.0])

        Cd = 2.2
        area_to_mass = 0.01e-6 / 10.0

        ax, ay, az = dynamics.atmospheric_drag_coesa76(0.0, y, Cd, area_to_mass)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

        # Should be in realistic range for 400 km altitude with small area-to-mass
        assert accel_mag > 1e-12, "Drag acceleration should be measurable"
        assert accel_mag < 1e-6, "Drag acceleration should be realistic"


class TestAlbedoRadiationPressure:
    """Tests for Earth albedo radiation pressure model."""

    def test_albedo_acceleration_is_measurable(self):
        """Test that albedo SRP produces measurable acceleration.

        For a satellite with large area-to-mass ratio in LEO on the day side,
        albedo should produce measurable acceleration.
        """
        epoch = Epoch(2000, 6, 21, 12, 0, 0)  # Summer solstice, daytime
        # LEO orbit at 400 km, on sunlit side
        r = c.RADIUS_EARTH + 400.0
        v = np.sqrt(c.MU_EARTH / r)

        # Position satellite on Sun-facing side (approximate)
        sun = Sun()
        r_sun = sun.get_position(epoch)
        # Place satellite between Earth and Sun
        e_sun = r_sun / np.linalg.norm(r_sun)
        pos = r * e_sun
        # Circular velocity perpendicular to position
        vel = v * np.array([-e_sun[1], e_sun[0], 0.0])
        vel = vel / np.linalg.norm(vel) * v

        y = np.hstack([pos, vel])

        # Large area-to-mass ratio (solar sail-like)
        Cr = 1.5
        area_to_mass = 0.1e-6  # 0.1 m²/kg in km²/kg

        ax, ay, az = dynamics.albedo_pressure_acceleration(0.0, y, Cr, area_to_mass, r_sun)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

        # Albedo acceleration should be measurable
        assert accel_mag > 0.0, "Albedo should produce acceleration"
        assert accel_mag < 1e-5, "Albedo acceleration should be smaller than direct SRP"

    def test_albedo_zero_on_night_side(self):
        """Test that albedo is zero when satellite is on night side."""
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = c.RADIUS_EARTH + 400.0

        sun = Sun()
        r_sun = sun.get_position(epoch)
        # Place satellite on opposite side from Sun
        e_sun = r_sun / np.linalg.norm(r_sun)
        pos = -r * e_sun  # Away from Sun
        vel = np.array([0.0, 0.0, 7.5])  # Arbitrary velocity

        y = np.hstack([pos, vel])

        Cr = 1.5
        area_to_mass = 0.1e-6

        ax, ay, az = dynamics.albedo_pressure_acceleration(0.0, y, Cr, area_to_mass, r_sun)
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

        # Should be zero on night side
        assert accel_mag == pytest.approx(0.0, abs=1e-15)

    def test_albedo_pushes_away_from_earth(self):
        """Test that albedo acceleration pushes satellite away from Earth.

        Reflected sunlight should push the satellite away from Earth
        (in the direction of the satellite position vector).
        """
        epoch = Epoch(2000, 6, 21, 12, 0, 0)
        r = c.RADIUS_EARTH + 400.0
        v = 7.5

        sun = Sun()
        r_sun = sun.get_position(epoch)
        e_sun = r_sun / np.linalg.norm(r_sun)

        # Satellite on Sun-facing side
        pos = r * e_sun
        vel = v * np.array([-e_sun[1], e_sun[0], 0.0])
        vel = vel / np.linalg.norm(vel) * v

        y = np.hstack([pos, vel])

        Cr = 1.5
        area_to_mass = 0.1e-6

        ax, ay, az = dynamics.albedo_pressure_acceleration(0.0, y, Cr, area_to_mass, r_sun)
        accel = np.array([ax, ay, az])

        # Acceleration should be in direction away from Earth
        e_sat = pos / np.linalg.norm(pos)
        dot_product = np.dot(accel, e_sat)

        assert dot_product > 0.0, "Albedo should push away from Earth"

    def test_albedo_with_direct_srp_combined(self):
        """Test that albedo and direct SRP can be used together.

        When using SUNandALBEDO model, both accelerations should contribute.
        """
        epoch = Epoch(2000, 6, 21, 12, 0, 0)
        r = c.RADIUS_EARTH + 400.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])

        # Very large area-to-mass ratio (solar sail)
        sat_props = SatelliteProperties(
            dry_mass=1.0,
            area=10.0,  # 10 m² (large solar sail)
            Cr=1.8,  # High reflectivity
        )

        # Two-body only (baseline)
        fm_twobody = ForceModel()
        prop_twobody = Propagator(epoch, state, force_model=fm_twobody, sat_properties=sat_props)

        # With SRP (sun and albedo)
        fm_srp = ForceModel(srp=True, srp_model=SRPModel.SUNandALBEDO)
        prop_srp = Propagator(epoch, state, force_model=fm_srp, sat_properties=sat_props)

        # Propagate for 3 days for larger effect
        final_epoch = epoch + Duration(days=3)
        prop_twobody.propagate_to_epoch(final_epoch)
        prop_srp.propagate_to_epoch(final_epoch)

        pos_twobody = prop_twobody.get_state().position()
        pos_srp = prop_srp.get_state().position()

        # Position should differ due to SRP
        position_diff = np.linalg.norm(pos_srp - pos_twobody)
        assert position_diff > 0.001, "SRP should cause measurable position difference"


class TestDragAndSRPIntegration:
    """Integration tests for drag and SRP models in full propagation."""

    def test_leo_with_all_perturbations(self):
        """Test LEO propagation with drag, SRP, J2, and third-body.

        This validates that all new perturbations work together correctly
        and produce physically reasonable results.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = c.RADIUS_EARTH + 400.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])

        sat_props = SatelliteProperties(
            dry_mass=10.0,
            area=0.01,
            Cd=2.2,
            Cr=1.5,
        )

        # Full perturbations
        from comet.propagation.perturbation_model import NBodyModel, GravityPotentialModel
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

        prop = Propagator(epoch, state, force_model=fm, sat_properties=sat_props)

        # Propagate for 7 days
        final_epoch = epoch + Duration(days=7)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        pos_final = final_state.position()
        r_final = np.linalg.norm(pos_final)

        # Should complete without error and show orbit decay
        altitude_initial = r - c.RADIUS_EARTH
        altitude_final = r_final - c.RADIUS_EARTH

        assert altitude_final > 0.0, "Satellite should still be in orbit"
        assert altitude_final < altitude_initial, "Drag should cause decay"
        # Decay should be in reasonable range
        assert altitude_initial - altitude_final < 100.0, "Decay should be realistic"

    def test_exponential_vs_coesa76_long_term(self):
        """Test that exponential and COESA76 produce different long-term decay.

        Over multiple days, the different density models should lead to
        measurably different orbital decay rates.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = c.RADIUS_EARTH + 400.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])

        sat_props = SatelliteProperties(dry_mass=10.0, area=0.01, Cd=2.2)

        # Exponential model
        fm_exp = ForceModel(drag=True, atmosphere_model=AtmosphereModel.EXPONENTIAL)
        prop_exp = Propagator(epoch, state, force_model=fm_exp, sat_properties=sat_props)

        # COESA76 model
        fm_c76 = ForceModel(drag=True, atmosphere_model=AtmosphereModel.COESA76)
        prop_c76 = Propagator(epoch, state, force_model=fm_c76, sat_properties=sat_props)

        # Propagate for 14 days
        final_epoch = epoch + Duration(days=14)
        prop_exp.propagate_to_epoch(final_epoch)
        prop_c76.propagate_to_epoch(final_epoch)

        pos_exp = prop_exp.get_state().position()
        pos_c76 = prop_c76.get_state().position()

        r_exp = np.linalg.norm(pos_exp)
        r_c76 = np.linalg.norm(pos_c76)

        # The two models should produce different final altitudes
        altitude_diff = abs((r_exp - c.RADIUS_EARTH) - (r_c76 - c.RADIUS_EARTH))
        assert altitude_diff > 0.01, "Different drag models should produce different decay"
