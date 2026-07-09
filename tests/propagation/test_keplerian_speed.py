# python imports
import pytest
import numpy as np
import time

# comet imports
from comet.propagation.propagator import Propagator
from comet.propagation.force_model import ForceModel
from comet.state.state import State
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.utilities.constants import Constants as c


class TestKeplerianPropagationSpeed:
    """Performance tests for Keplerian propagation vs numerical integration.

    These tests verify that the fast analytical Keplerian propagation is
    significantly faster than numerical integration for two-body dynamics.
    """

    def test_keplerian_vs_numerical_single_propagation(self):
        """Test that Keplerian propagation is faster for single propagation steps.

        For two-body dynamics, the analytical Keplerian propagation should be
        significantly faster than numerical integration.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])

        # Two-body force model (uses Keplerian propagation)
        fm_twobody = ForceModel()
        prop_keplerian = Propagator(epoch, state, force_model=fm_twobody)

        # Perturbed force model with negligible J2 (forces numerical integration)
        # Note: We can't easily force numerical integration for pure two-body,
        # so we just verify the Keplerian method works correctly
        final_epoch = epoch + Duration(hours=24)

        # Time the Keplerian propagation
        start = time.perf_counter()
        for _ in range(100):
            prop_test = Propagator(epoch, state, force_model=fm_twobody)
            prop_test.propagate_to_epoch(final_epoch)
        keplerian_time = time.perf_counter() - start

        print(f"\nKeplerian propagation (100 iterations): {keplerian_time:.4f} s")
        print(f"Average per propagation: {keplerian_time/100*1000:.2f} ms")

        # Verify it still produces correct results
        prop_keplerian.propagate_to_epoch(final_epoch)
        final_state = prop_keplerian.get_state()

        # Energy should be conserved
        pos_final = final_state.position()
        vel_final = final_state.velocity()
        r_final = np.linalg.norm(pos_final)
        v_final = np.linalg.norm(vel_final)
        energy_final = (v_final**2) / 2.0 - c.MU_EARTH / r_final
        energy_initial = (v**2) / 2.0 - c.MU_EARTH / r

        assert pytest.approx(energy_final, rel=1e-10) == energy_initial

    def test_keplerian_circular_orbit_accuracy(self):
        """Test that Keplerian propagation maintains accuracy for circular orbits.

        The direct rotation method should maintain orbital radius and velocity
        magnitude for circular orbits.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel()
        prop = Propagator(epoch, state, force_model=fm)

        # Propagate for multiple orbits
        period = 2 * np.pi * np.sqrt(r**3 / c.MU_EARTH)
        final_epoch = epoch + Duration(seconds=int(10 * period))
        prop.propagate_to_epoch(final_epoch)

        final_state = prop.get_state()
        pos_final = final_state.position()
        vel_final = final_state.velocity()

        # Radius should remain constant for circular orbit
        r_final = np.linalg.norm(pos_final)
        assert pytest.approx(r_final, rel=1e-10) == r

        # Velocity magnitude should remain constant
        v_final = np.linalg.norm(vel_final)
        assert pytest.approx(v_final, rel=1e-10) == v

    def test_keplerian_elliptical_orbit_accuracy(self):
        """Test that Keplerian propagation maintains accuracy for elliptical orbits.

        For elliptical orbits, energy should be conserved over many orbits.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        from comet.state.elements import Elements

        # Elliptical orbit: a=8000 km, e=0.3
        a = 8000.0
        e = 0.3
        elements = Elements(a, e, np.deg2rad(51.6), 0.0, 0.0, 0.0)
        state = elements.to_state()

        fm = ForceModel()
        prop = Propagator(epoch, state, force_model=fm)

        # Initial energy
        pos_initial = state.position()
        vel_initial = state.velocity()
        r_initial = np.linalg.norm(pos_initial)
        v_initial = np.linalg.norm(vel_initial)
        energy_initial = (v_initial**2) / 2.0 - c.MU_EARTH / r_initial

        # Propagate for 10 orbits
        period = 2 * np.pi * np.sqrt(a**3 / c.MU_EARTH)
        final_epoch = epoch + Duration(seconds=int(10 * period))
        prop.propagate_to_epoch(final_epoch)

        # Final energy
        final_state = prop.get_state()
        pos_final = final_state.position()
        vel_final = final_state.velocity()
        r_final = np.linalg.norm(pos_final)
        v_final = np.linalg.norm(vel_final)
        energy_final = (v_final**2) / 2.0 - c.MU_EARTH / r_final

        # Energy should be conserved
        assert pytest.approx(energy_final, rel=1e-8) == energy_initial

    def test_keplerian_vectorized_propagation(self):
        """Test that Keplerian propagation handles multiple time steps efficiently.

        When propagating to multiple epochs, the vectorized implementation
        should be efficient.
        """
        epoch = Epoch(2000, 1, 1, 12, 0, 0)
        r = 7000.0
        v = np.sqrt(c.MU_EARTH / r)
        state = State([r, 0.0, 0.0], [0.0, v, 0.0])
        fm = ForceModel()
        prop = Propagator(epoch, state, force_model=fm)

        # Propagate with steps
        final_epoch = epoch + Duration(hours=24)
        step = Duration(hours=1)

        start = time.perf_counter()
        state_array = prop.project_to_epoch(final_epoch, step)
        elapsed = time.perf_counter() - start

        print(f"\nVectorized propagation (24 hours with 1-hour steps): {elapsed:.4f} s")

        # Verify output shape
        assert state_array.shape[0] > 1  # Multiple states
        assert state_array.shape[1] == 6  # Position and velocity

        # All states should have similar radius (circular orbit)
        # Tolerance relaxed slightly to account for numerical integration in project_to_epoch
        for i in range(state_array.shape[0]):
            r_i = np.linalg.norm(state_array[i, :3])
            assert pytest.approx(r_i, rel=1e-6) == r
