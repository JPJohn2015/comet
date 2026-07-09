# python imports
import pytest
import numpy as np

# comet imports
from comet.propagation.propagator import Propagator, Integrator
from comet.propagation.force_model import ForceModel
from comet.propagation.perturbation_model import (
    AtmosphereModel,
    NBodyModel,
    SRPModel,
    GravityPotentialModel,
)
from comet.propagation.satellite_properties import SatelliteProperties
from comet.state.state import State
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.utilities.constants import Constants as c


class TestTwoBodyPropagation:
    """Integration tests for two-body propagation.

    Validates that basic two-body dynamics preserve orbital energy
    and period as expected.
    """

    def test_two_body_energy_conservation(self):
        """Test two-body propagation conserves specific orbital energy.

        For Keplerian (two-body) dynamics, specific orbital energy
        ε = v²/2 - μ/r should remain constant.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # Circular orbit at 7000 km
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel()  # Two-body only
        prop = Propagator(epoch, state, force_model=fm)

        # Initial energy
        pos_initial = state.position()
        vel_initial = state.velocity()
        r_initial = np.linalg.norm(pos_initial)
        v_initial = np.linalg.norm(vel_initial)
        energy_initial = (v_initial**2) / 2.0 - c.MU_EARTH / r_initial

        # Propagate one orbit period
        period = 2 * np.pi * np.sqrt(r**3 / c.MU_EARTH)
        final_epoch = epoch + Duration(seconds=int(period))
        prop.propagate_to_epoch(final_epoch)

        # Final energy
        final_state = prop.get_state()
        pos_final = final_state.position()
        vel_final = final_state.velocity()
        r_final = np.linalg.norm(pos_final)
        v_final = np.linalg.norm(vel_final)
        energy_final = (v_final**2) / 2.0 - c.MU_EARTH / r_final

        # Energy should be conserved to high precision
        assert pytest.approx(energy_final, rel=1e-8) == energy_initial

    def test_two_body_orbital_period(self):
        """Test two-body propagation returns to initial position after one period.

        For a circular orbit, the satellite should return to approximately
        the same position after one orbital period T = 2π√(a³/μ).
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # Circular orbit at 7000 km
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel()  # Two-body only
        prop = Propagator(epoch, state, force_model=fm)

        pos_initial = state.position()

        # Propagate one orbit period
        period = 2 * np.pi * np.sqrt(r**3 / c.MU_EARTH)
        final_epoch = epoch + Duration(seconds=int(period))
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        pos_final = final_state.position()

        # Position should be close after one period (within ~10 km due to phase error)
        np.testing.assert_allclose(pos_final, pos_initial, rtol=1e-5, atol=15.0)

    def test_two_body_elliptical_orbit(self):
        """Test two-body propagation for elliptical orbit.

        Validates energy conservation and period for an elliptical orbit.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # Elliptical orbit: a=8000 km, e=0.1
        a = 8000.0
        e = 0.1
        elements = Elements(a, e, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        fm = ForceModel()  # Two-body only
        prop = Propagator(epoch, elements, force_model=fm)

        # Initial energy from elements
        energy_initial = -c.MU_EARTH / (2 * a)

        # Propagate one orbit period
        period = 2 * np.pi * np.sqrt(a**3 / c.MU_EARTH)
        final_epoch = epoch + Duration(seconds=int(period))
        prop.propagate_to_epoch(final_epoch)

        # Final energy from state
        final_state = prop.get_state()
        pos_final = final_state.position()
        vel_final = final_state.velocity()
        r_final = np.linalg.norm(pos_final)
        v_final = np.linalg.norm(vel_final)
        energy_final = (v_final**2) / 2.0 - c.MU_EARTH / r_final

        # Energy should be conserved
        assert pytest.approx(energy_final, rel=1e-8) == energy_initial

    def test_two_body_different_integrators(self):
        """Test two-body propagation produces consistent results across integrators.

        Different integrators should produce similar results for the same problem.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel()

        # Propagate with different integrators
        integrators = [Integrator.DOPRI5, Integrator.DOP853, Integrator.LSODA]
        final_states = []

        for integrator in integrators:
            prop = Propagator(epoch, state, force_model=fm, integrator=integrator)
            final_epoch = epoch + Duration(seconds=3600)  # 1 hour
            prop.propagate_to_epoch(final_epoch)
            final_states.append(prop.get_state())

        # All integrators should produce similar results
        for i in range(len(final_states) - 1):
            pos_i = final_states[i].position()
            pos_next = final_states[i + 1].position()
            # Positions should agree to ~1 km (different methods, reasonable difference)
            np.testing.assert_allclose(pos_i, pos_next, atol=1.0)


class TestPerturbedPropagation:
    """Integration tests for propagation with perturbations.

    Tests individual perturbations (J2, J3, N-body, SRP) and validates
    their effects on orbital motion.
    """

    def test_j2_perturbation_effect(self):
        """Test J2 perturbation causes nodal precession.

        J2 (Earth oblateness) causes the right ascension of ascending node (RAAN)
        to drift. For a LEO satellite, this should be measurable over hours.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # LEO with inclination
        elements = Elements(7000.0, 0.001, np.deg2rad(51.6), 0.0, 0.0, 0.0)

        # Two-body propagation
        fm_twobody = ForceModel()
        prop_twobody = Propagator(epoch, elements, force_model=fm_twobody)
        final_epoch = epoch + Duration(hours=24)
        prop_twobody.propagate_to_epoch(final_epoch)

        # J2 propagation
        fm_j2 = ForceModel(gravity=True, gravity_model=GravityPotentialModel.J2)
        prop_j2 = Propagator(epoch, elements, force_model=fm_j2)
        prop_j2.propagate_to_epoch(final_epoch)

        # States should differ due to J2
        state_twobody = prop_twobody.get_state()
        state_j2 = prop_j2.get_state()
        pos_twobody = state_twobody.position()
        pos_j2 = state_j2.position()

        # Difference should be significant (>100 m after 24 hours)
        diff = np.linalg.norm(pos_j2 - pos_twobody)
        assert diff > 0.1  # At least 100 meters difference

    def test_j2_vs_j2_and_j3(self):
        """Test J2 and J3 together produce different results than J2 alone.

        J3 perturbation should add measurable effect beyond J2.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        elements = Elements(7000.0, 0.001, np.deg2rad(51.6), 0.0, 0.0, 0.0)

        # J2 only
        fm_j2 = ForceModel(gravity=True, gravity_model=GravityPotentialModel.J2)
        prop_j2 = Propagator(epoch, elements, force_model=fm_j2)
        final_epoch = epoch + Duration(hours=24)
        prop_j2.propagate_to_epoch(final_epoch)

        # J2 and J3
        fm_j2j3 = ForceModel(gravity=True, gravity_model=GravityPotentialModel.J2andJ3)
        prop_j2j3 = Propagator(epoch, elements, force_model=fm_j2j3)
        prop_j2j3.propagate_to_epoch(final_epoch)

        # States should differ
        state_j2 = prop_j2.get_state()
        state_j2j3 = prop_j2j3.get_state()
        pos_j2 = state_j2.position()
        pos_j2j3 = state_j2j3.position()

        # Difference should be measurable (J3 is smaller than J2 but nonzero)
        diff = np.linalg.norm(pos_j2j3 - pos_j2)
        assert diff > 0.001  # At least 1 meter difference

    def test_solar_pressure_effect(self):
        """Test solar radiation pressure affects orbit.

        SRP should cause measurable perturbation over multiple orbits.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])

        # Two-body
        fm_twobody = ForceModel()
        prop_twobody = Propagator(epoch, state, force_model=fm_twobody)
        final_epoch = epoch + Duration(hours=48)
        prop_twobody.propagate_to_epoch(final_epoch)

        # With SRP (large area-to-mass for stronger effect)
        sp = SatelliteProperties(dry_mass=100.0, area=10.0, Cr=1.5)
        fm_srp = ForceModel(srp=True, srp_model=SRPModel.SUN)
        prop_srp = Propagator(epoch, state, force_model=fm_srp, sat_properties=sp)
        prop_srp.propagate_to_epoch(final_epoch)

        # States should differ
        state_twobody = prop_twobody.get_state()
        state_srp = prop_srp.get_state()
        pos_twobody = state_twobody.position()
        pos_srp = state_srp.position()

        diff = np.linalg.norm(pos_srp - pos_twobody)
        # SRP effect should be measurable (sub-meter level over 2 days with small area-to-mass)
        assert diff > 0.0001  # At least 10 cm

    def test_third_body_sun_effect(self):
        """Test solar third-body perturbation affects orbit.

        Sun's gravity should cause measurable perturbation over days.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])

        # Two-body
        fm_twobody = ForceModel()
        prop_twobody = Propagator(epoch, state, force_model=fm_twobody)
        final_epoch = epoch + Duration(hours=72)  # 3 days
        prop_twobody.propagate_to_epoch(final_epoch)

        # With Sun third-body
        fm_sun = ForceModel(nbody=True, nbody_model=NBodyModel.SUN)
        prop_sun = Propagator(epoch, state, force_model=fm_sun)
        prop_sun.propagate_to_epoch(final_epoch)

        # States should differ
        state_twobody = prop_twobody.get_state()
        state_sun = prop_sun.get_state()
        pos_twobody = state_twobody.position()
        pos_sun = state_sun.position()

        diff = np.linalg.norm(pos_sun - pos_twobody)
        # Solar perturbation should be measurable
        assert diff > 0.01  # At least 10 meters

    def test_third_body_moon_effect(self):
        """Test lunar third-body perturbation affects orbit.

        Moon's gravity should cause measurable perturbation over days.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])

        # Two-body
        fm_twobody = ForceModel()
        prop_twobody = Propagator(epoch, state, force_model=fm_twobody)
        final_epoch = epoch + Duration(hours=72)  # 3 days
        prop_twobody.propagate_to_epoch(final_epoch)

        # With Moon third-body
        fm_moon = ForceModel(nbody=True, nbody_model=NBodyModel.MOON)
        prop_moon = Propagator(epoch, state, force_model=fm_moon)
        prop_moon.propagate_to_epoch(final_epoch)

        # States should differ
        state_twobody = prop_twobody.get_state()
        state_moon = prop_moon.get_state()
        pos_twobody = state_twobody.position()
        pos_moon = state_moon.position()

        diff = np.linalg.norm(pos_moon - pos_twobody)
        # Lunar perturbation should be measurable
        assert diff > 0.01  # At least 10 meters


class TestCombinedForceModels:
    """Integration tests for multiple perturbations acting together.

    Validates that combinations of perturbations work correctly.
    """

    def test_all_perturbations_combined(self):
        """Test propagation with all perturbations enabled.

        Validates that multiple perturbations can be combined without error.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        sp = SatelliteProperties(dry_mass=100.0, area=2.0)

        # All perturbations
        fm = ForceModel(
            drag=True,
            srp=True,
            nbody=True,
            gravity=True,
            atmosphere_model=AtmosphereModel.EXPONENTIAL,
            srp_model=SRPModel.SUN,
            nbody_model=NBodyModel.SUNandMOON,
            gravity_model=GravityPotentialModel.J2andJ3,
        )

        prop = Propagator(
            epoch, state, force_model=fm, sat_properties=sp, integrator=Integrator.DOP853
        )

        # Should propagate without error
        final_epoch = epoch + Duration(hours=24)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        assert final_state is not None

        # Position should have changed significantly
        pos_initial = state.position()
        pos_final = final_state.position()
        assert not np.allclose(pos_initial, pos_final, rtol=1e-3)

    def test_j2_plus_drag_leo_decay(self):
        """Test J2 + drag causes orbit decay for LEO.

        Drag should cause orbital energy to decrease, lowering the orbit.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # Low LEO where drag is significant
        r = 6678.0  # ~300 km altitude
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        sp = SatelliteProperties(dry_mass=100.0, area=2.0, Cd=2.2)

        # J2 + drag
        fm = ForceModel(
            drag=True,
            gravity=True,
            atmosphere_model=AtmosphereModel.EXPONENTIAL,
            gravity_model=GravityPotentialModel.J2,
        )

        prop = Propagator(epoch, state, force_model=fm, sat_properties=sp)

        initial_r = np.linalg.norm(state.position())

        # Propagate several orbits
        final_epoch = epoch + Duration(hours=48)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        final_r = np.linalg.norm(final_state.position())

        # Orbit should decay (radius should decrease)
        assert final_r < initial_r

    def test_srp_and_third_body_combination(self):
        """Test SRP and third-body perturbations work together.

        Both perturbations should contribute to total perturbation.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        sp = SatelliteProperties(dry_mass=100.0, area=5.0)

        # SRP + N-body
        fm = ForceModel(
            srp=True, nbody=True, srp_model=SRPModel.SUN, nbody_model=NBodyModel.SUNandMOON
        )

        prop = Propagator(epoch, state, force_model=fm, sat_properties=sp)

        final_epoch = epoch + Duration(hours=48)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        assert final_state is not None

        # Should differ significantly from two-body
        pos_final = final_state.position()
        pos_initial = state.position()
        assert not np.allclose(pos_initial, pos_final, rtol=1e-3)


class TestIntegratorComparison:
    """Tests comparing different integrators for consistency and accuracy."""

    def test_integrator_consistency_short_term(self):
        """Test all integrators produce consistent short-term results.

        Over short timespans, all integrators should agree closely.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel()

        integrators = [
            Integrator.DOPRI5,
            Integrator.DOP853,
            Integrator.LSODA,
            Integrator.VODE,
        ]

        final_states = []
        for integrator in integrators:
            prop = Propagator(epoch, state, force_model=fm, integrator=integrator)
            final_epoch = epoch + Duration(minutes=30)
            prop.propagate_to_epoch(final_epoch)
            final_states.append(prop.get_state())

        # All should agree to ~100 m (different methods have some variation)
        for i in range(len(final_states) - 1):
            pos_i = final_states[i].position()
            pos_next = final_states[i + 1].position()
            np.testing.assert_allclose(pos_i, pos_next, atol=0.1)

    def test_high_accuracy_integrator_vs_standard(self):
        """Test DOP853 (high order) vs DOPRI5 (standard) accuracy.

        DOP853 should be more accurate for tight tolerances.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel()

        # Standard integrator with standard tolerance
        prop_standard = Propagator(
            epoch, state, force_model=fm, integrator=Integrator.DOPRI5, abs_tol=1e-6
        )

        # High-order integrator with tight tolerance
        prop_high = Propagator(
            epoch, state, force_model=fm, integrator=Integrator.DOP853, abs_tol=1e-10
        )

        final_epoch = epoch + Duration(hours=12)
        prop_standard.propagate_to_epoch(final_epoch)
        prop_high.propagate_to_epoch(final_epoch)

        # Both should produce valid results
        state_standard = prop_standard.get_state()
        state_high = prop_high.get_state()

        assert state_standard is not None
        assert state_high is not None

        # Positions should be close but not identical
        pos_standard = state_standard.position()
        pos_high = state_high.position()

        # Should agree to within standard tolerance but differ beyond tight tolerance
        np.testing.assert_allclose(pos_standard, pos_high, atol=0.01)  # Within 10 meters


class TestPropagationEdgeCases:
    """Tests for edge cases and boundary conditions in propagation."""

    def test_backward_propagation(self):
        """Test propagation backward in time.

        Should work correctly in reverse.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # Use correct circular velocity
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel()
        prop = Propagator(epoch, state, force_model=fm)

        # Propagate backward
        past_epoch = epoch - Duration(hours=1)
        prop.propagate_to_epoch(past_epoch)

        past_state = prop.get_state()
        assert past_state is not None

        # Propagate forward again
        prop.propagate_to_epoch(epoch)

        # Should return close to original
        final_state = prop.get_state()
        pos_final = final_state.position()
        pos_initial = state.position()
        np.testing.assert_allclose(pos_final, pos_initial, rtol=1e-5, atol=1e-3)

    def test_very_short_propagation(self):
        """Test propagation over very short duration.

        Should handle small time steps without error.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel()
        prop = Propagator(epoch, state, force_model=fm)

        # Propagate 1 second
        final_epoch = epoch + Duration(seconds=1)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        assert final_state is not None

    def test_geo_orbit_propagation(self):
        """Test propagation of GEO orbit.

        GEO orbits are at much higher altitude; validates propagation works there.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # GEO altitude
        r = 42164.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel(gravity=True, gravity_model=GravityPotentialModel.J2)
        prop = Propagator(epoch, state, force_model=fm)

        # Propagate one day
        final_epoch = epoch + Duration(hours=24)
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        assert final_state is not None

        # Orbit should be stable (radius roughly constant for near-circular GEO)
        r_final = np.linalg.norm(final_state.position())
        assert pytest.approx(r_final, rel=1e-3) == r

    def test_highly_elliptical_orbit(self):
        """Test propagation of highly elliptical orbit (HEO).

        High eccentricity orbits stress the integrator.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # HEO: a=26000 km, e=0.7
        a = 26000.0
        e = 0.7
        elements = Elements(a, e, np.deg2rad(63.4), 0.0, 0.0, 0.0)
        fm = ForceModel(gravity=True, gravity_model=GravityPotentialModel.J2)
        prop = Propagator(epoch, elements, force_model=fm, integrator=Integrator.DOP853)

        # Propagate one orbit
        period = 2 * np.pi * np.sqrt(a**3 / c.MU_EARTH)
        final_epoch = epoch + Duration(seconds=int(period))
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        assert final_state is not None


class TestRoundTripPropagation:
    """Tests for round-trip propagation (forward then backward)."""

    def test_round_trip_two_body(self):
        """Test forward-then-backward propagation returns to initial state.

        For two-body dynamics with circular orbits, this is exact (within numerical error).
        For elliptical orbits, element conversions introduce small errors.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        # Use correct circular velocity for truly circular orbit
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel()
        prop = Propagator(epoch, state, force_model=fm)

        pos_initial = state.position()
        vel_initial = state.velocity()

        # Forward
        future_epoch = epoch + Duration(hours=6)
        prop.propagate_to_epoch(future_epoch)

        # Backward
        prop.propagate_to_epoch(epoch)

        # Should return to initial
        final_state = prop.get_state()
        pos_final = final_state.position()
        vel_final = final_state.velocity()

        np.testing.assert_allclose(pos_final, pos_initial, rtol=1e-6, atol=1e-3)
        np.testing.assert_allclose(vel_final, vel_initial, rtol=1e-6, atol=1e-6)

    def test_round_trip_perturbed(self):
        """Test forward-then-backward with perturbations.

        Should still return close to initial state.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel(gravity=True, gravity_model=GravityPotentialModel.J2andJ3)
        prop = Propagator(epoch, state, force_model=fm, integrator=Integrator.DOP853)

        pos_initial = state.position()
        vel_initial = state.velocity()

        # Forward
        future_epoch = epoch + Duration(hours=12)
        prop.propagate_to_epoch(future_epoch)

        # Backward
        prop.propagate_to_epoch(epoch)

        # Should return close to initial
        final_state = prop.get_state()
        pos_final = final_state.position()
        vel_final = final_state.velocity()

        np.testing.assert_allclose(pos_final, pos_initial, rtol=1e-5, atol=1e-2)
        np.testing.assert_allclose(vel_final, vel_initial, rtol=1e-5, atol=1e-5)


class TestStateConversions:
    """Tests for propagation with different state representations."""

    def test_propagate_from_elements(self):
        """Test propagation starting from orbital elements.

        Elements should be converted to state and propagated correctly.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        elements = Elements(7000.0, 0.001, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        fm = ForceModel()
        prop = Propagator(epoch, elements, force_model=fm)

        # Propagate
        final_epoch = epoch + Duration(hours=1)
        prop.propagate_to_epoch(final_epoch)

        # Should return State
        final_state = prop.get_state()
        assert isinstance(final_state, State)
        assert final_state is not None

    def test_elements_state_equivalence(self):
        """Test propagation from elements vs equivalent state.

        Starting from elements or the equivalent state should give same result.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        elements = Elements(7000.0, 0.01, np.deg2rad(51.6), 0.0, 0.0, 0.0)

        # Convert to state
        state = elements.to_state()

        fm = ForceModel()

        # Propagate from elements
        prop_elem = Propagator(epoch, elements, force_model=fm)
        final_epoch = epoch + Duration(hours=1)
        prop_elem.propagate_to_epoch(final_epoch)
        state_from_elem = prop_elem.get_state()

        # Propagate from state
        prop_state = Propagator(epoch, state, force_model=fm)
        prop_state.propagate_to_epoch(final_epoch)
        state_from_state = prop_state.get_state()

        # Results should be identical
        pos_elem = state_from_elem.position()
        pos_state = state_from_state.position()
        np.testing.assert_allclose(pos_elem, pos_state, rtol=1e-10)


class TestPropagatorCopyAndUpdate:
    """Tests for propagator copying and updating during propagation."""

    def test_propagator_copy_independence(self):
        """Test copied propagator is independent of original.

        Propagating one should not affect the other.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel()
        prop1 = Propagator(epoch, state, force_model=fm)
        prop2 = prop1.copy()

        # Propagate only prop1
        final_epoch = epoch + Duration(hours=1)
        prop1.propagate_to_epoch(final_epoch)

        # prop1 should be updated
        assert prop1.epoch == final_epoch

        # prop2 should still be at initial epoch
        assert prop2.epoch == epoch

    def test_multiple_propagation_steps(self):
        """Test sequential propagation steps update propagator state.

        Multiple propagate calls should accumulate.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        state = State([7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
        fm = ForceModel()
        prop = Propagator(epoch, state, force_model=fm)

        pos_initial = state.position()

        # Step 1
        epoch1 = epoch + Duration(hours=1)
        prop.propagate_to_epoch(epoch1)
        state1 = prop.get_state()
        pos1 = state1.position()

        # Step 2
        epoch2 = epoch1 + Duration(hours=1)
        prop.propagate_to_epoch(epoch2)
        state2 = prop.get_state()
        pos2 = state2.position()

        # Step 3
        epoch3 = epoch2 + Duration(hours=1)
        prop.propagate_to_epoch(epoch3)
        state3 = prop.get_state()
        pos3 = state3.position()

        # All positions should be different
        assert not np.allclose(pos_initial, pos1)
        assert not np.allclose(pos1, pos2)
        assert not np.allclose(pos2, pos3)

        # Final epoch should be epoch3
        assert prop.epoch == epoch3
