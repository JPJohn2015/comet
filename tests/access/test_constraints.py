"""Unit tests for concrete access constraints.

Tests all constraint implementations from constraints.py, organized by category:
- Line-of-sight and illumination
- Solar and lunar geometry
- Range and rate
- Pointing
- Time windows

Each test validates:
- Metric computation correctness
- Access thresholding with bounds
- State needs declaration
- Edge cases and special geometries
"""

# python imports
import pytest
import numpy as np

# COMET imports
from comet.access.constraints import (
    # Line-of-sight and illumination
    EarthLineOfSightConstraint,
    LunarLineOfSightConstraint,
    TargetIsSunlitConstraint,
    SelfIsSunlitConstraint,
    SelfIsEclipseConstraint,
    # Solar and lunar geometry
    SolarPhaseAngleConstraint,
    InPlaneSolarPhaseAngleConstraint,
    SolarExclusionAngleConstraint,
    LunarExclusionAngleConstraint,
    EarthLimbExclusionAngleConstraint,
    # Range and rate
    RangeConstraint,
    RangeRateConstraint,
    AngularRateConstraint,
    # Pointing
    AngleFromNadirConstraint,
    # Time windows
    EpochConstraint,
    EpochWindowsConstraint,
)
from comet.access.core import (
    CustomStates,
    EvalMode,
    StateNeed,
    resolve_inputs,
)
from comet.time import Epoch
from comet.utilities.constants import Constants as c


class TestLineOfSightConstraints:
    """Test line-of-sight and illumination constraints."""

    def test_earth_los_clear_space(self):
        """Test EarthLineOfSightConstraint with clear LOS."""
        # Two objects at GEO with clear LOS
        source_pos = np.array([[[42000, 0, 0]]])
        target_pos = np.array([[[0, 42000, 0]]])

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
        )

        constraint = EarthLineOfSightConstraint()
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric[0, 0, 0] == 1.0  # LOS is clear
        assert access[0, 0, 0] == 1.0
        assert StateNeed.SOURCE in constraint.needs
        assert StateNeed.TARGET in constraint.needs

    def test_earth_los_obstructed(self):
        """Test EarthLineOfSightConstraint with Earth obstruction."""
        # Opposite sides of Earth at LEO
        alt = c.A_EARTH / 1000.0 + 500
        source_pos = np.array([[[alt, 0, 0]]])
        target_pos = np.array([[[-alt, 0, 0]]])

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
        )

        constraint = EarthLineOfSightConstraint()
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric[0, 0, 0] == 0.0  # LOS is blocked
        assert access[0, 0, 0] == 0.0

    def test_target_sunlit(self):
        """Test TargetIsSunlitConstraint."""
        # Target in sunlight
        target_pos = np.array([[[7000, 0, 0]]])
        sun_pos = np.array([[1.5e8, 0, 0]])  # Sun in +x direction

        custom_states = CustomStates(
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
            sun_position=sun_pos,
        )

        constraint = TargetIsSunlitConstraint()
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric.squeeze() == 1.0  # Target is sunlit
        assert StateNeed.TARGET in constraint.needs
        assert StateNeed.SUN in constraint.needs

    def test_self_eclipse(self):
        """Test SelfIsEclipseConstraint."""
        # Source on opposite side of Earth from Sun
        source_pos = np.array([[[-7000, 0, 0]]])
        sun_pos = np.array([[1.5e8, 0, 0]])

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            sun_position=sun_pos,
        )

        constraint = SelfIsEclipseConstraint()
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric.squeeze() == 1.0  # Source is in eclipse
        assert access.squeeze() == 1.0


class TestSolarLunarConstraints:
    """Test solar and lunar geometry constraints."""

    def test_solar_phase_angle(self):
        """Test SolarPhaseAngleConstraint."""
        # Source viewing illuminated side
        source_pos = np.array([[[10000, 0, 0]]])
        target_pos = np.array([[[0, 0, 0]]])
        sun_pos = np.array([[1.5e8, 0, 0]])

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
            sun_position=sun_pos,
        )

        constraint = SolarPhaseAngleConstraint(min_value=0, max_value=90)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 0.0, atol=0.1)  # Fully illuminated
        assert access[0, 0, 0] == 1.0  # Within bounds
        assert StateNeed.SOURCE in constraint.needs
        assert StateNeed.TARGET in constraint.needs
        assert StateNeed.SUN in constraint.needs

    def test_solar_phase_angle_out_of_bounds(self):
        """Test SolarPhaseAngleConstraint with phase angle outside bounds."""
        # Source viewing dark side
        source_pos = np.array([[[-10000, 0, 0]]])
        target_pos = np.array([[[0, 0, 0]]])
        sun_pos = np.array([[1.5e8, 0, 0]])

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
            sun_position=sun_pos,
        )

        constraint = SolarPhaseAngleConstraint(min_value=0, max_value=90)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 180.0, atol=0.1)  # Dark side
        assert access[0, 0, 0] == 0.0  # Outside bounds

    def test_solar_exclusion_angle(self):
        """Test SolarExclusionAngleConstraint."""
        # Source looking at target perpendicular to Sun direction
        source_pos = np.array([[[0, 0, 0]]])
        target_pos = np.array([[[0, 10000, 0]]])  # Perpendicular
        sun_pos = np.array([[1.5e8, 0, 0]])

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
            sun_position=sun_pos,
        )

        constraint = SolarExclusionAngleConstraint(min_value=80, max_value=100)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 90.0, atol=0.1)  # 90° from Sun
        assert access[0, 0, 0] == 1.0  # Within bounds

    def test_lunar_exclusion_angle(self):
        """Test LunarExclusionAngleConstraint."""
        source_pos = np.array([[[0, 0, 0]]])
        target_pos = np.array([[[10000, 0, 0]]])
        moon_pos = np.array([[0, 384400e3, 0]])  # Moon perpendicular

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
            moon_position=moon_pos,
        )

        constraint = LunarExclusionAngleConstraint(min_value=15)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 90.0, atol=0.1)
        assert access[0, 0, 0] == 1.0
        assert StateNeed.MOON in constraint.needs

    def test_earth_limb_angle(self):
        """Test EarthLimbExclusionAngleConstraint."""
        # Source looking straight down at Earth center
        alt = c.A_EARTH / 1000.0 + 500
        source_pos = np.array([[[alt, 0, 0]]])
        target_pos = np.array([[[0, 0, 0]]])

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
        )

        constraint = EarthLimbExclusionAngleConstraint(min_value=0)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric[0, 0, 0] > 20  # Well above limb
        assert access[0, 0, 0] == 1.0


class TestRangeRateConstraints:
    """Test range and rate constraints."""

    def test_range_constraint_within_bounds(self):
        """Test RangeConstraint with range within bounds."""
        source_pos = np.array([[[0, 0, 0]]])
        target_pos = np.array([[[3000, 4000, 0]]])  # 5000 km away

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
        )

        constraint = RangeConstraint(min_value=1000, max_value=8000)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 5000.0, atol=1e-6)
        assert access[0, 0, 0] == 1.0
        assert StateNeed.SOURCE in constraint.needs
        assert StateNeed.TARGET in constraint.needs

    def test_range_constraint_out_of_bounds(self):
        """Test RangeConstraint with range outside bounds."""
        source_pos = np.array([[[0, 0, 0]]])
        target_pos = np.array([[[100, 0, 0]]])  # 100 km away

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
        )

        constraint = RangeConstraint(min_value=1000, max_value=8000)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 100.0, atol=1e-6)
        assert access[0, 0, 0] == 0.0  # Below minimum

    def test_range_rate_approaching(self):
        """Test RangeRateConstraint with approaching objects."""
        source_state = np.array([[[0, 0, 0, 0, 0, 0]]])
        target_state = np.array([[[10, 0, 0, -1, 0, 0]]])  # Moving toward source

        custom_states = CustomStates(
            source_states=source_state,
            target_states=target_state,
        )

        constraint = RangeRateConstraint(min_value=-10, max_value=0)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], -1.0, atol=1e-6)  # Approaching
        assert access[0, 0, 0] == 1.0

    def test_range_rate_separating(self):
        """Test RangeRateConstraint with separating objects."""
        source_state = np.array([[[0, 0, 0, 0, 0, 0]]])
        target_state = np.array([[[10, 0, 0, 1, 0, 0]]])  # Moving away

        custom_states = CustomStates(
            source_states=source_state,
            target_states=target_state,
        )

        constraint = RangeRateConstraint(min_value=-10, max_value=0)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 1.0, atol=1e-6)  # Separating
        assert access[0, 0, 0] == 0.0  # Outside bounds (positive > 0)

    def test_angular_rate(self):
        """Test AngularRateConstraint."""
        # Target moving tangentially at 10 km, velocity 1 km/s
        # Angular rate = v_perp / r = 1 / 10 = 0.1 rad/s = 5.73 deg/s
        source_state = np.array([[[0, 0, 0, 0, 0, 0]]])
        target_state = np.array([[[10, 0, 0, 0, 1, 0]]])

        custom_states = CustomStates(
            source_states=source_state,
            target_states=target_state,
        )

        constraint = AngularRateConstraint(max_value=10.0)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        expected = np.rad2deg(1.0 / 10.0)
        assert np.isclose(metric[0, 0, 0], expected, atol=0.01)
        assert access[0, 0, 0] == 1.0


class TestPointingConstraints:
    """Test pointing constraints."""

    def test_angle_from_nadir_straight_down(self):
        """Test AngleFromNadirConstraint looking at nadir."""
        alt = c.A_EARTH / 1000.0 + 500
        source_pos = np.array([[[alt, 0, 0]]])
        target_pos = np.array([[[0, 0, 0]]])  # Earth center

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
        )

        constraint = AngleFromNadirConstraint(max_value=30)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 0.0, atol=0.1)  # Looking at nadir
        assert access[0, 0, 0] == 1.0
        assert StateNeed.SOURCE in constraint.needs
        assert StateNeed.TARGET in constraint.needs

    def test_angle_from_nadir_perpendicular(self):
        """Test AngleFromNadirConstraint with perpendicular view."""
        alt = c.A_EARTH / 1000.0 + 500
        source_pos = np.array([[[alt, 0, 0]]])
        target_pos = np.array([[[alt, 1000, 0]]])  # Perpendicular to nadir

        custom_states = CustomStates(
            source_states=np.concatenate([source_pos, np.zeros((1, 1, 3))], axis=-1),
            target_states=np.concatenate([target_pos, np.zeros((1, 1, 3))], axis=-1),
        )

        constraint = AngleFromNadirConstraint(min_value=80, max_value=100)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert np.isclose(metric[0, 0, 0], 90.0, atol=0.1)
        assert access[0, 0, 0] == 1.0


class TestTimeWindowConstraints:
    """Test time window constraints."""

    def test_epoch_constraint_inside_window(self):
        """Test EpochConstraint with epoch inside window."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        end = Epoch(2024, 1, 1, 6, 0, 0)
        current = Epoch(2024, 1, 1, 3, 0, 0)

        custom_states = CustomStates(epochs=np.array([current]))

        constraint = EpochConstraint(start_epoch=start, end_epoch=end)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric[0] == 1.0  # Inside window
        assert access[0] == 1.0
        assert StateNeed.TIMELINE in constraint.needs

    def test_epoch_constraint_outside_window(self):
        """Test EpochConstraint with epoch outside window."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        end = Epoch(2024, 1, 1, 6, 0, 0)
        current = Epoch(2024, 1, 1, 12, 0, 0)  # After window

        custom_states = CustomStates(epochs=np.array([current]))

        constraint = EpochConstraint(start_epoch=start, end_epoch=end)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric[0] == 0.0  # Outside window
        assert access[0] == 0.0

    def test_epoch_windows_constraint_multiple_windows(self):
        """Test EpochWindowsConstraint with multiple windows."""
        windows = [
            (Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 6, 0, 0)),
            (Epoch(2024, 1, 1, 12, 0, 0), Epoch(2024, 1, 1, 18, 0, 0)),
        ]

        epochs = np.array([
            Epoch(2024, 1, 1, 3, 0, 0),   # Inside first window
            Epoch(2024, 1, 1, 9, 0, 0),   # Between windows
            Epoch(2024, 1, 1, 15, 0, 0),  # Inside second window
        ])

        custom_states = CustomStates(epochs=epochs)

        constraint = EpochWindowsConstraint(windows)
        inputs = resolve_inputs(EvalMode.CUSTOM, [constraint], custom_states=custom_states)

        metric, access = constraint.eval(inputs)

        assert metric[0] == 1.0  # Inside first window
        assert metric[1] == 0.0  # Between windows
        assert metric[2] == 1.0  # Inside second window


class TestConstraintIntegration:
    """Test constraint integration with batch evaluation."""

    def test_multiple_constraints_with_batch_eval(self):
        """Test using multiple constraints together."""
        from comet.access.core import evaluate_composite_access
        from comet.assets import Satellite
        from comet.assets import Groundstation
        from comet.state import Elements
        from comet.state import LLA
        from comet.time import TIMELINE, TimelineMode
        from comet.time import Duration

        # Set up timeline
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        # Create assets
        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='TestSat',
        )

        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='TestGS',
        )

        # Set up constraints: LOS and range
        constraints_per_source = {
            0: [
                EarthLineOfSightConstraint(),
                RangeConstraint(min_value=1000, max_value=10000),
            ],
        }

        # Evaluate
        composite = evaluate_composite_access(
            sources=[sat],
            targets=[gs],
            constraints_per_source=constraints_per_source,
            mode=EvalMode.BATCH,
            timeline=TIMELINE,
        )

        # Should have results for all time points
        assert composite.shape == (1, 1, 6)
        assert np.all((composite == 0.0) | (composite == 1.0))  # All boolean values
