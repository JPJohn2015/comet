"""Unit tests for access framework core classes and structures.

Tests the foundational abstractions:
- StateNeed and EvalMode enums
- CustomStates validation
- AccessInputs container
- AccessConstraint base class and metric-first contract
- Access window extraction
"""

# python imports
import pytest
import numpy as np

# COMET imports
from comet.access.core import (
    AccessConstraint,
    AccessInputs,
    CustomStates,
    EvalMode,
    StateNeed,
    extract_access_windows_indices,
    resolve_inputs,
    evaluate_composite_access,
)
from comet.time import Epoch
from comet.time import Duration
from comet.time import Timeline


class TestEnums:
    """Test StateNeed and EvalMode enums."""

    def test_state_need_values(self):
        """Test StateNeed enum has expected values."""
        assert StateNeed.SOURCE.value == "source"
        assert StateNeed.TARGET.value == "target"
        assert StateNeed.SUN.value == "sun"
        assert StateNeed.MOON.value == "moon"
        assert StateNeed.TIMELINE.value == "timeline"

    def test_eval_mode_values(self):
        """Test EvalMode enum has expected values."""
        assert EvalMode.CURRENT.value == "current"
        assert EvalMode.CUSTOM.value == "custom"
        assert EvalMode.BATCH.value == "batch"

    def test_state_need_in_set(self):
        """Test StateNeed can be used in sets."""
        needs = {StateNeed.SOURCE, StateNeed.TARGET, StateNeed.SUN}
        assert StateNeed.SOURCE in needs
        assert StateNeed.MOON not in needs
        assert len(needs) == 3


class TestCustomStates:
    """Test CustomStates validation."""

    def test_custom_states_empty(self):
        """Test empty CustomStates."""
        cs = CustomStates()
        assert cs.source_states is None
        assert cs.target_states is None
        assert cs.sun_position is None
        assert cs.moon_position is None
        assert cs.epochs is None

    def test_custom_states_validate_missing(self):
        """Test validation fails when required state is missing."""
        cs = CustomStates(source_states=np.zeros((1, 10, 6)))

        # Requires target but target_states is None
        with pytest.raises(ValueError, match="TARGET required"):
            cs.validate({StateNeed.SOURCE, StateNeed.TARGET})

    def test_custom_states_validate_shape_mismatch(self):
        """Test validation fails when time dimensions don't match."""
        cs = CustomStates(
            source_states=np.zeros((2, 10, 6)),  # T=10
            target_states=np.zeros((3, 15, 6)),  # T=15 (mismatch!)
        )

        with pytest.raises(ValueError, match="Time dimension mismatch"):
            cs.validate({StateNeed.SOURCE, StateNeed.TARGET})

    def test_custom_states_validate_success(self):
        """Test validation succeeds when all required states present and compatible."""
        cs = CustomStates(
            source_states=np.zeros((2, 10, 6)),
            target_states=np.zeros((3, 10, 6)),
            sun_position=np.zeros((10, 3)),
        )

        # Should not raise
        cs.validate({StateNeed.SOURCE, StateNeed.TARGET, StateNeed.SUN})

    def test_custom_states_validate_time_dims_match(self):
        """Test validation with all states having matching time dimensions."""
        T = 20
        cs = CustomStates(
            source_states=np.zeros((2, T, 6)),
            target_states=np.zeros((3, T, 6)),
            sun_position=np.zeros((T, 3)),
            moon_position=np.zeros((T, 3)),
            epochs=np.array([Epoch(2024, 1, 1, 0, 0, i) for i in range(T)]),
        )

        # Should not raise
        cs.validate({StateNeed.SOURCE, StateNeed.TARGET, StateNeed.SUN,
                     StateNeed.MOON, StateNeed.TIMELINE})


class TestAccessInputs:
    """Test AccessInputs container."""

    def test_access_inputs_construction(self):
        """Test constructing AccessInputs."""
        inputs = AccessInputs(
            mode=EvalMode.BATCH,
            source_positions=np.zeros((2, 10, 3)),
            target_positions=np.zeros((3, 10, 3)),
            time_count=10,
            source_count=2,
            target_count=3,
        )

        assert inputs.mode == EvalMode.BATCH
        assert inputs.source_positions.shape == (2, 10, 3)
        assert inputs.target_positions.shape == (3, 10, 3)
        assert inputs.time_count == 10
        assert inputs.source_count == 2
        assert inputs.target_count == 3

    def test_access_inputs_optional_fields(self):
        """Test AccessInputs with optional fields."""
        inputs = AccessInputs(mode=EvalMode.CURRENT)

        assert inputs.mode == EvalMode.CURRENT
        assert inputs.source_positions is None
        assert inputs.sun_position is None
        assert inputs.time_count == 0


class TestAccessConstraintBase:
    """Test AccessConstraint base class."""

    def test_constraint_abstract(self):
        """Test that AccessConstraint cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AccessConstraint()

    def test_constraint_repr(self):
        """Test constraint string representation."""
        # Create a concrete subclass for testing
        class TestConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.zeros((1, 1, 1))

        constraint = TestConstraint(
            min_value=10,
            max_value=90,
            needs={StateNeed.SOURCE, StateNeed.TARGET}
        )

        repr_str = repr(constraint)
        assert "TestConstraint" in repr_str
        assert "[10" in repr_str and "90]" in repr_str
        assert "source" in repr_str
        assert "target" in repr_str


class TestMetricFirstContract:
    """Test the metric-first contract implementation."""

    class SimpleConstraint(AccessConstraint):
        """Simple test constraint that returns a constant metric."""

        def _metric(self, inputs):
            return np.ones((2, 3, 5)) * 50  # Always returns 50

    def test_metric_passthrough(self):
        """Test that metric() passes through _metric() values."""
        constraint = self.SimpleConstraint()
        inputs = AccessInputs(mode=EvalMode.BATCH)

        metric = constraint.metric(inputs)

        assert metric.shape == (2, 3, 5)
        assert np.all(metric == 50)

    def test_access_within_bounds(self):
        """Test access() when metric is within bounds."""
        constraint = self.SimpleConstraint(min_value=0, max_value=100)
        inputs = AccessInputs(mode=EvalMode.BATCH)

        access = constraint.access(inputs)

        assert access.shape == (2, 3, 5)
        assert np.all(access == 1.0)  # All within bounds

    def test_access_below_min(self):
        """Test access() when metric is below minimum."""
        constraint = self.SimpleConstraint(min_value=60, max_value=100)
        inputs = AccessInputs(mode=EvalMode.BATCH)

        access = constraint.access(inputs)

        assert np.all(access == 0.0)  # All below min

    def test_access_above_max(self):
        """Test access() when metric is above maximum."""
        constraint = self.SimpleConstraint(min_value=0, max_value=40)
        inputs = AccessInputs(mode=EvalMode.BATCH)

        access = constraint.access(inputs)

        assert np.all(access == 0.0)  # All above max

    def test_access_partial_bounds(self):
        """Test access() with mixed values."""
        class VaryingConstraint(AccessConstraint):
            def _metric(self, inputs):
                # Return values 0, 25, 50, 75, 100
                return np.array([[[0, 25, 50, 75, 100]]])

        constraint = VaryingConstraint(min_value=20, max_value=80)
        inputs = AccessInputs(mode=EvalMode.BATCH)

        access = constraint.access(inputs)

        expected = np.array([[[0, 1, 1, 1, 0]]])  # Only 25, 50, 75 in bounds
        assert np.array_equal(access, expected.astype(float))

    def test_eval_computes_both(self):
        """Test eval() returns both metric and access."""
        constraint = self.SimpleConstraint(min_value=0, max_value=100)
        inputs = AccessInputs(mode=EvalMode.BATCH)

        metric, access = constraint.eval(inputs)

        assert metric.shape == (2, 3, 5)
        assert access.shape == (2, 3, 5)
        assert np.all(metric == 50)
        assert np.all(access == 1.0)

    def test_eval_single_computation(self):
        """Test that eval() computes metric only once."""
        call_count = 0

        class CountingConstraint(AccessConstraint):
            def _metric(self, inputs):
                nonlocal call_count
                call_count += 1
                return np.ones((1, 1, 1))

        constraint = CountingConstraint()
        inputs = AccessInputs(mode=EvalMode.BATCH)

        # Call eval once
        constraint.eval(inputs)

        # _metric should be called exactly once
        assert call_count == 1

    def test_boolean_constraint_pattern(self):
        """Test boolean-only constraints (no thresholding)."""
        class BooleanConstraint(AccessConstraint):
            def _metric(self, inputs):
                # Returns boolean as float (LOS pattern)
                return np.array([[[1.0, 0.0, 1.0]]])

            def access(self, inputs):
                # Boolean constraints return the metric directly (no thresholding)
                return self._metric(inputs)

        # Boolean constraints don't use bounds - they override access()
        constraint = BooleanConstraint()
        inputs = AccessInputs(mode=EvalMode.BATCH)

        metric, access = constraint.eval(inputs)

        # Metric is the boolean itself
        assert np.array_equal(metric, np.array([[[1.0, 0.0, 1.0]]]))
        # Access is also the boolean (no thresholding applied)
        assert np.array_equal(access, np.array([[[1.0, 0.0, 1.0]]]))


class TestAccessWindowExtraction:
    """Test access window extraction functions."""

    def test_extract_windows_single_window(self):
        """Test extracting a single continuous access window."""
        # Access from index 2 to 5
        mask = np.array([0, 0, 1, 1, 1, 1, 0, 0])

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 1
        assert windows[0] == (2, 5)

    def test_extract_windows_multiple_windows(self):
        """Test extracting multiple access windows."""
        # Two windows: 1-2 and 5-7
        mask = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 2
        assert windows[0] == (1, 2)
        assert windows[1] == (5, 7)

    def test_extract_windows_at_start(self):
        """Test window starting at index 0."""
        mask = np.array([1, 1, 0, 0, 1, 1])

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 2
        assert windows[0] == (0, 1)
        assert windows[1] == (4, 5)

    def test_extract_windows_at_end(self):
        """Test window ending at last index."""
        mask = np.array([0, 0, 1, 1, 1])

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 1
        assert windows[0] == (2, 4)

    def test_extract_windows_full_access(self):
        """Test when entire timeline has access."""
        mask = np.ones(10)

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 1
        assert windows[0] == (0, 9)

    def test_extract_windows_no_access(self):
        """Test when no access exists."""
        mask = np.zeros(10)

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 0

    def test_extract_windows_single_point(self):
        """Test window of single time point."""
        mask = np.array([0, 0, 1, 0, 0])

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 1
        assert windows[0] == (2, 2)

    def test_extract_windows_3d_mask(self):
        """Test extraction from 3D mask (N, K, T)."""
        # Create (2, 3, 10) mask
        mask = np.zeros((2, 3, 10))
        mask[1, 2, 3:7] = 1.0  # Access for source 1, target 2, times 3-6

        windows = extract_access_windows_indices(mask, source_idx=1, target_idx=2)

        assert len(windows) == 1
        assert windows[0] == (3, 6)

    def test_extract_windows_alternating(self):
        """Test extraction with alternating access."""
        # On/off/on/off pattern
        mask = np.array([1, 0, 1, 0, 1, 0])

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 3
        assert windows[0] == (0, 0)
        assert windows[1] == (2, 2)
        assert windows[2] == (4, 4)


class TestConstraintNeeds:
    """Test constraint state needs declaration."""

    def test_needs_default_empty(self):
        """Test default needs is empty set."""
        class MinimalConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.zeros((1, 1, 1))

        constraint = MinimalConstraint()
        assert constraint.needs == set()

    def test_needs_explicit_set(self):
        """Test setting explicit needs."""
        class SourceTargetConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.zeros((1, 1, 1))

        constraint = SourceTargetConstraint(
            needs={StateNeed.SOURCE, StateNeed.TARGET}
        )

        assert StateNeed.SOURCE in constraint.needs
        assert StateNeed.TARGET in constraint.needs
        assert StateNeed.SUN not in constraint.needs

    def test_needs_all_types(self):
        """Test constraint requiring all state types."""
        class FullConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.zeros((1, 1, 1))

        constraint = FullConstraint(
            needs={StateNeed.SOURCE, StateNeed.TARGET, StateNeed.SUN,
                   StateNeed.MOON, StateNeed.TIMELINE}
        )

        assert len(constraint.needs) == 5
        assert StateNeed.SOURCE in constraint.needs
        assert StateNeed.TARGET in constraint.needs
        assert StateNeed.SUN in constraint.needs
        assert StateNeed.MOON in constraint.needs
        assert StateNeed.TIMELINE in constraint.needs


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_constraint_inf_bounds(self):
        """Test constraint with infinite bounds."""
        class TestConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.array([[[np.inf, -np.inf, 0]]])

        constraint = TestConstraint(min_value=-np.inf, max_value=np.inf)
        inputs = AccessInputs(mode=EvalMode.BATCH)

        access = constraint.access(inputs)

        # Even infinite values are "within" infinite bounds
        assert np.all(access == 1.0)

    def test_constraint_nan_handling(self):
        """Test constraint behavior with NaN metrics."""
        class NaNConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.array([[[np.nan, 50, 100]]])

        constraint = NaNConstraint(min_value=0, max_value=100)
        inputs = AccessInputs(mode=EvalMode.BATCH)

        access = constraint.access(inputs)

        # NaN comparisons are False, so access should be 0 for NaN
        assert access[0, 0, 0] == 0.0
        assert access[0, 0, 1] == 1.0
        assert access[0, 0, 2] == 1.0

    def test_empty_mask_extraction(self):
        """Test window extraction from empty mask."""
        mask = np.array([])

        windows = extract_access_windows_indices(mask)

        assert len(windows) == 0


class TestResolveInputsCustomMode:
    """Test resolve_inputs in CUSTOM mode."""

    def test_custom_mode_with_source_and_target(self):
        """Test CUSTOM mode with source and target states."""
        source_states = np.random.randn(2, 10, 6)
        target_states = np.random.randn(3, 10, 6)

        custom_states = CustomStates(
            source_states=source_states,
            target_states=target_states,
        )

        class TestConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.ones((inputs.source_count, inputs.target_count, inputs.time_count))

        constraints = [TestConstraint(needs={StateNeed.SOURCE, StateNeed.TARGET})]

        inputs = resolve_inputs(
            mode=EvalMode.CUSTOM,
            constraints=constraints,
            custom_states=custom_states,
        )

        assert inputs.mode == EvalMode.CUSTOM
        assert inputs.source_positions.shape == (2, 10, 3)
        assert inputs.target_positions.shape == (3, 10, 3)
        assert inputs.source_count == 2
        assert inputs.target_count == 3
        assert inputs.time_count == 10

    def test_custom_mode_with_2d_inputs(self):
        """Test CUSTOM mode with 2D inputs (single source/target)."""
        source_states = np.random.randn(10, 6)  # Single source
        target_states = np.random.randn(10, 6)  # Single target

        custom_states = CustomStates(
            source_states=source_states,
            target_states=target_states,
        )

        class TestConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.ones((1, 1, inputs.time_count))

        constraints = [TestConstraint(needs={StateNeed.SOURCE, StateNeed.TARGET})]

        inputs = resolve_inputs(
            mode=EvalMode.CUSTOM,
            constraints=constraints,
            custom_states=custom_states,
        )

        # Should expand to 3D
        assert inputs.source_positions.shape == (1, 10, 3)
        assert inputs.target_positions.shape == (1, 10, 3)
        assert inputs.source_count == 1
        assert inputs.target_count == 1

    def test_custom_mode_with_sun_moon(self):
        """Test CUSTOM mode with sun and moon positions."""
        source_states = np.random.randn(1, 5, 6)
        sun_position = np.random.randn(5, 3)
        moon_position = np.random.randn(5, 3)

        custom_states = CustomStates(
            source_states=source_states,
            sun_position=sun_position,
            moon_position=moon_position,
        )

        class TestConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.ones((1, 1, inputs.time_count))

        constraints = [TestConstraint(needs={StateNeed.SOURCE, StateNeed.SUN, StateNeed.MOON})]

        inputs = resolve_inputs(
            mode=EvalMode.CUSTOM,
            constraints=constraints,
            custom_states=custom_states,
        )

        assert inputs.sun_position.shape == (5, 3)
        assert inputs.moon_position.shape == (5, 3)
        assert inputs.time_count == 5


class TestEvaluateCompositeAccess:
    """Test evaluate_composite_access batch engine."""

    def setup_method(self):
        """Set up timeline and assets for tests."""
        from comet.time import TIMELINE, TimelineMode
        from comet.time import Epoch
        from comet.time import Duration
        from comet.assets import Satellite
        from comet.assets import Groundstation
        from comet.state import Elements
        from comet.state import LLA

        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 5, 0),
            Duration(minutes=1),
        )

        self.timeline = TIMELINE

        # Create test assets
        self.sat1 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat1',
        )

        self.sat2 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7100, 0.001, 0, 0, 0, 0),
            name='Sat2',
        )

        self.gs1 = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS1',
        )

        self.gs2 = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(10, 15, 0),
            name='GS2',
        )

    def test_single_source_single_target_single_constraint(self):
        """Test basic case: one source, one target, one constraint."""
        class AlwaysAccessConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.ones((inputs.source_count, inputs.target_count, inputs.time_count))

        constraints_per_source = {
            0: [AlwaysAccessConstraint(min_value=0.5, needs={StateNeed.SOURCE, StateNeed.TARGET})],
        }

        composite = evaluate_composite_access(
            sources=[self.sat1],
            targets=[self.gs1],
            constraints_per_source=constraints_per_source,
            mode=EvalMode.BATCH,
            timeline=self.timeline,
        )

        assert composite.shape == (1, 1, 6)
        assert np.all(composite == 1.0)

    def test_multiple_sources_different_constraints(self):
        """Test multiple sources with different constraint sets."""
        class AlwaysAccessConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.ones((inputs.source_count, inputs.target_count, inputs.time_count))

        class NeverAccessConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.zeros((inputs.source_count, inputs.target_count, inputs.time_count))

        constraints_per_source = {
            0: [AlwaysAccessConstraint(min_value=0.5, needs={StateNeed.SOURCE, StateNeed.TARGET})],
            1: [NeverAccessConstraint(min_value=0.5, needs={StateNeed.SOURCE, StateNeed.TARGET})],
        }

        composite = evaluate_composite_access(
            sources=[self.sat1, self.sat2],
            targets=[self.gs1],
            constraints_per_source=constraints_per_source,
            mode=EvalMode.BATCH,
            timeline=self.timeline,
        )

        assert composite.shape == (2, 1, 6)
        assert np.all(composite[0, 0, :] == 1.0)  # Sat1 has access
        assert np.all(composite[1, 0, :] == 0.0)  # Sat2 no access

    def test_multiple_constraints_per_source_and_logic(self):
        """Test multiple constraints per source with AND logic."""
        class AlternatingConstraint(AccessConstraint):
            def _metric(self, inputs):
                result = np.ones((inputs.source_count, inputs.target_count, inputs.time_count))
                result[:, :, 1::2] = 0.0  # Every other time step is 0
                return result

        class OffsetAlternatingConstraint(AccessConstraint):
            def _metric(self, inputs):
                result = np.zeros((inputs.source_count, inputs.target_count, inputs.time_count))
                result[:, :, 1::2] = 1.0  # Every other time step is 1
                return result

        constraints_per_source = {
            0: [
                AlternatingConstraint(min_value=0.5, needs={StateNeed.SOURCE, StateNeed.TARGET}),
                OffsetAlternatingConstraint(min_value=0.5, needs={StateNeed.SOURCE, StateNeed.TARGET}),
            ],
        }

        composite = evaluate_composite_access(
            sources=[self.sat1],
            targets=[self.gs1],
            constraints_per_source=constraints_per_source,
            mode=EvalMode.BATCH,
            timeline=self.timeline,
        )

        # Non-overlapping alternating patterns should result in no access
        assert np.all(composite[0, 0, :] == 0.0)

    def test_empty_constraints(self):
        """Test with no constraints (should return all-access)."""
        constraints_per_source = {}

        composite = evaluate_composite_access(
            sources=[self.sat1],
            targets=[self.gs1],
            constraints_per_source=constraints_per_source,
            mode=EvalMode.BATCH,
            timeline=self.timeline,
        )

        assert composite.shape == (1, 1, 6)
        assert np.all(composite == 1.0)

    def test_multiple_targets(self):
        """Test with multiple targets."""
        class AlwaysAccessConstraint(AccessConstraint):
            def _metric(self, inputs):
                return np.ones((inputs.source_count, inputs.target_count, inputs.time_count))

        constraints_per_source = {
            0: [AlwaysAccessConstraint(min_value=0.5, needs={StateNeed.SOURCE, StateNeed.TARGET})],
        }

        composite = evaluate_composite_access(
            sources=[self.sat1],
            targets=[self.gs1, self.gs2],
            constraints_per_source=constraints_per_source,
            mode=EvalMode.BATCH,
            timeline=self.timeline,
        )

        assert composite.shape == (1, 2, 6)
        assert np.all(composite == 1.0)
