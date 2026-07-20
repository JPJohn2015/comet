"""Comprehensive test suite for the Timeline class.

This module tests all functionality of the Timeline class including:
- Basic construction and properties
- Dual-mode operation (BATCH and STEPPED)
- Mode switching and cursor management
- Cache invalidation via hash
- Serialization (to_dict/from_dict)
- Edge cases
"""

# python imports
import pytest
import numpy as np

# comet imports
from comet.time.timeline import Timeline, TimelineMode, get_time_deltas, get_epoch_list
from comet.time.epoch import Epoch
from comet.time.duration import Duration


class TestTimelineConstruction:
    """Tests for Timeline construction and basic properties."""

    def test_basic_construction(self):
        """Test basic Timeline construction."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 1, 1, 0, 0)
        step = Duration(seconds=60)

        tl = Timeline(start, stop, step)

        assert tl.start == start
        assert tl.stop == stop
        assert tl.step == step
        assert tl.get_mode() == TimelineMode.BATCH

    def test_construction_invalid_stop_before_start(self):
        """Test that stop before start raises ValueError."""
        start = Epoch(2024, 1, 1, 1, 0, 0)
        stop = Epoch(2024, 1, 1, 0, 0, 0)
        step = Duration(seconds=60)

        with pytest.raises(ValueError, match="stop Epoch must be before start Epoch"):
            Timeline(start, stop, step)

    def test_construction_invalid_zero_step(self):
        """Test that zero step raises ValueError."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 1, 1, 0, 0)
        step = Duration(seconds=0)

        with pytest.raises(ValueError, match="step Duration must be greater than zero"):
            Timeline(start, stop, step)

    def test_construction_invalid_negative_step(self):
        """Test that negative step raises ValueError."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 1, 1, 0, 0)
        step = Duration(seconds=-60)

        with pytest.raises(ValueError, match="step Duration must be greater than zero"):
            Timeline(start, stop, step)

    def test_initial_mode_is_batch(self):
        """Test that Timeline initializes in BATCH mode."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))
        assert tl.get_mode() == TimelineMode.BATCH

    def test_initial_cursor_at_start(self):
        """Test that cursor initializes at timeline start."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        tl = Timeline(start, Epoch(2024, 1, 2), Duration(seconds=60))

        assert tl.now_index() == 0
        assert tl.now() == start
        assert tl.now_time_delta() == 0.0

    def test_hash_generated_on_construction(self):
        """Test that hash is generated on construction."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))
        assert tl.get_hash() is not None
        assert isinstance(tl.get_hash(), str)
        assert len(tl.get_hash()) == 32  # MD5 hash length


class TestTimelineUpdate:
    """Tests for Timeline.update() method."""

    def test_update_start(self):
        """Test updating start epoch."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))
        new_start = Epoch(2024, 1, 1, 6, 0, 0)

        old_hash = tl.get_hash()
        tl.update(start=new_start)

        assert tl.start == new_start
        assert tl.get_hash() != old_hash

    def test_update_stop(self):
        """Test updating stop epoch."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))
        new_stop = Epoch(2024, 1, 3)

        old_hash = tl.get_hash()
        tl.update(stop=new_stop)

        assert tl.stop == new_stop
        assert tl.get_hash() != old_hash

    def test_update_step(self):
        """Test updating step duration."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))
        new_step = Duration(seconds=120)

        old_hash = tl.get_hash()
        tl.update(step=new_step)

        assert tl.step == new_step
        assert tl.get_hash() != old_hash

    def test_update_resets_cursor(self):
        """Test that update resets cursor to start."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        # Advance cursor
        tl.advance(5)
        assert tl.now_index() == 5

        # Update timeline
        tl.update(step=Duration(seconds=120))

        # Cursor should be reset
        assert tl.now_index() == 0
        assert tl.now_time_delta() == 0.0

    def test_update_none_preserves_values(self):
        """Test that passing None preserves existing values."""
        start = Epoch(2024, 1, 1)
        stop = Epoch(2024, 1, 2)
        step = Duration(seconds=60)

        tl = Timeline(start, stop, step)
        tl.update(start=None, stop=None, step=None)

        assert tl.start == start
        assert tl.stop == stop
        assert tl.step == step


class TestTimelineModeControl:
    """Tests for Timeline mode switching and control."""

    def test_set_mode_to_stepped(self):
        """Test switching to STEPPED mode."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        old_hash = tl.get_hash()
        tl.set_mode(TimelineMode.STEPPED)

        assert tl.get_mode() == TimelineMode.STEPPED
        assert tl.get_hash() != old_hash

    def test_set_mode_to_batch(self):
        """Test switching to BATCH mode."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        tl.set_mode(TimelineMode.STEPPED)
        old_hash = tl.get_hash()
        tl.set_mode(TimelineMode.BATCH)

        assert tl.get_mode() == TimelineMode.BATCH
        assert tl.get_hash() != old_hash

    def test_set_mode_same_mode_updates_hash(self):
        """Test that setting to same mode still updates hash."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        old_hash = tl.get_hash()
        tl.set_mode(TimelineMode.BATCH)

        # Hash should not change when setting to same mode
        assert tl.get_hash() == old_hash

    def test_set_mode_invalid_type(self):
        """Test that invalid mode type raises TypeError."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        with pytest.raises(TypeError, match="mode must be a TimelineMode enum"):
            tl.set_mode("batch")

    def test_mode_round_trip(self):
        """Test switching modes back and forth."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        # Start in BATCH
        assert tl.get_mode() == TimelineMode.BATCH

        # Switch to STEPPED
        tl.set_mode(TimelineMode.STEPPED)
        assert tl.get_mode() == TimelineMode.STEPPED

        # Switch back to BATCH
        tl.set_mode(TimelineMode.BATCH)
        assert tl.get_mode() == TimelineMode.BATCH


class TestTimelineCursorControl:
    """Tests for Timeline stepped-mode cursor control."""

    def test_advance_one_step(self):
        """Test advancing cursor by one step."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=600))

        initial_index = tl.now_index()
        tl.advance(1)

        assert tl.now_index() == initial_index + 1
        assert tl.now_time_delta() == 600.0

    def test_advance_multiple_steps(self):
        """Test advancing cursor by multiple steps."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        tl.advance(5)

        assert tl.now_index() == 5
        assert tl.now_time_delta() == 300.0

    def test_advance_negative_steps(self):
        """Test advancing cursor backward."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        tl.advance(10)
        assert tl.now_index() == 10

        tl.advance(-5)
        assert tl.now_index() == 5

    def test_advance_clamps_at_end(self):
        """Test that advancing past end clamps to last index."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=600))

        time_deltas = tl.get_time_deltas()
        max_index = len(time_deltas) - 1

        tl.advance(1000)

        assert tl.now_index() == max_index
        assert tl.now_time_delta() == time_deltas[max_index]

    def test_advance_clamps_at_start(self):
        """Test that advancing before start clamps to index 0."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        tl.advance(5)
        assert tl.now_index() == 5

        tl.advance(-1000)

        assert tl.now_index() == 0
        assert tl.now_time_delta() == 0.0

    def test_advance_updates_epoch(self):
        """Test that advance updates now correctly."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        tl = Timeline(start, Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=600))

        tl.advance(2)

        expected_epoch = start + Duration(seconds=1200.0)
        assert tl.now() == expected_epoch

    def test_advance_updates_hash(self):
        """Test that advance updates hash for cache invalidation."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        old_hash = tl.get_hash()
        tl.advance(1)

        assert tl.get_hash() != old_hash

    def test_reset_cursor(self):
        """Test resetting cursor to start."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        tl = Timeline(start, Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        # Advance cursor
        tl.advance(10)
        assert tl.now_index() != 0

        # Reset cursor
        old_hash = tl.get_hash()
        tl.reset()

        assert tl.now_index() == 0
        assert tl.now() == start
        assert tl.now_time_delta() == 0.0
        assert tl.get_hash() != old_hash


class TestTimelineHashInvalidation:
    """Tests for Timeline hash-based cache invalidation."""

    def test_hash_changes_on_mode_switch(self):
        """Test that hash changes when mode switches."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        hash_batch = tl.get_hash()
        tl.set_mode(TimelineMode.STEPPED)
        hash_stepped = tl.get_hash()

        assert hash_batch != hash_stepped

    def test_hash_changes_on_advance(self):
        """Test that hash changes when cursor advances."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        hash_0 = tl.get_hash()
        tl.advance(1)
        hash_1 = tl.get_hash()
        tl.advance(1)
        hash_2 = tl.get_hash()

        assert hash_0 != hash_1
        assert hash_1 != hash_2
        assert hash_0 != hash_2

    def test_hash_changes_on_reset_cursor(self):
        """Test that hash changes when cursor resets."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        tl.advance(5)
        hash_advanced = tl.get_hash()

        tl.reset()
        hash_reset = tl.get_hash()

        assert hash_advanced != hash_reset

    def test_hash_changes_on_update(self):
        """Test that hash changes on timeline update."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        hash_original = tl.get_hash()
        tl.update(step=Duration(seconds=120))
        hash_updated = tl.get_hash()

        assert hash_original != hash_updated

    def test_hash_consistent_for_same_state(self):
        """Test that hash is consistent for same timeline state."""
        tl1 = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))
        tl2 = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        assert tl1.get_hash() == tl2.get_hash()


class TestTimelineProperties:
    """Tests for Timeline property accessors."""

    def test_get_time_deltas(self):
        """Test get_time_deltas returns correct array."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 0, 5, 0), Duration(seconds=60))

        time_deltas = tl.get_time_deltas()

        assert isinstance(time_deltas, np.ndarray)
        assert len(time_deltas) == 6  # 0, 60, 120, 180, 240, 300
        assert time_deltas[0] == 0.0
        assert time_deltas[-1] == 300.0

    def test_get_epoch_list(self):
        """Test get_epoch_list returns correct array."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        tl = Timeline(start, Epoch(2024, 1, 1, 0, 5, 0), Duration(seconds=60))

        epochs = tl.get_epoch_list()

        assert isinstance(epochs, np.ndarray)
        assert len(epochs) == 6
        assert epochs[0] == start

    def test_get_relative_time_deltas(self):
        """Test get_relative_time_deltas returns correct array."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 0, 5, 0), Duration(seconds=60))

        relative_deltas = tl.get_relative_time_deltas()

        assert isinstance(relative_deltas, np.ndarray)
        assert len(relative_deltas) == 6
        assert relative_deltas[0] == 0.0
        assert all(relative_deltas[1:] == 60.0)

    def test_get_duration_list(self):
        """Test get_duration_list returns correct array."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        tl = Timeline(start, Epoch(2024, 1, 1, 0, 5, 0), Duration(seconds=60))

        durations = tl.get_duration_list()

        assert isinstance(durations, np.ndarray)
        assert len(durations) == 6
        assert durations[0] == Duration(seconds=0)

    def test_get_julian_date_list(self):
        """Test get_julian_date_list returns correct array."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 0, 5, 0), Duration(seconds=60))

        jds = tl.get_julian_date_list()

        assert isinstance(jds, np.ndarray)
        assert len(jds) == 6

    def test_properties_memoized(self):
        """Test that property arrays are memoized."""
        tl = Timeline(Epoch(2024, 1, 1), Epoch(2024, 1, 2), Duration(seconds=60))

        # Call twice and check same object returned
        time_deltas_1 = tl.get_time_deltas()
        time_deltas_2 = tl.get_time_deltas()

        assert time_deltas_1 is time_deltas_2


class TestTimelineSerialization:
    """Tests for Timeline serialization (to_dict/from_dict)."""

    def test_to_dict_basic(self):
        """Test basic to_dict serialization."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 2, 0, 0, 0)
        step = Duration(seconds=60)

        tl = Timeline(start, stop, step)
        d = tl.to_dict()

        assert d["type"] == "Timeline"
        assert "start" in d
        assert "stop" in d
        assert "step" in d
        assert d["mode"] == "batch"
        assert d["index_now"] == 0
        assert d["dt_now"] == 0.0

    def test_to_dict_with_stepped_mode(self):
        """Test to_dict preserves STEPPED mode state."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        tl.set_mode(TimelineMode.STEPPED)
        tl.advance(5)

        d = tl.to_dict()

        assert d["mode"] == "stepped"
        assert d["index_now"] == 5
        assert d["dt_now"] == 300.0

    def test_from_dict_basic(self):
        """Test basic from_dict deserialization."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 2, 0, 0, 0)
        step = Duration(seconds=60)

        tl1 = Timeline(start, stop, step)
        d = tl1.to_dict()

        tl2 = Timeline.from_dict(d)

        assert tl2.start == start
        assert tl2.stop == stop
        assert tl2.step == step
        assert tl2.get_mode() == TimelineMode.BATCH

    def test_from_dict_preserves_mode_and_cursor(self):
        """Test from_dict preserves mode and cursor state."""
        tl1 = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        tl1.set_mode(TimelineMode.STEPPED)
        tl1.advance(5)

        d = tl1.to_dict()
        tl2 = Timeline.from_dict(d)

        assert tl2.get_mode() == TimelineMode.STEPPED
        assert tl2.now_index() == 5
        assert tl2.now_time_delta() == 300.0
        assert tl2.now() == tl1.now()

    def test_from_dict_round_trip(self):
        """Test complete round trip preserves all state."""
        tl1 = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 1, 0, 0), Duration(seconds=60))

        tl1.set_mode(TimelineMode.STEPPED)
        tl1.advance(3)

        d = tl1.to_dict()
        tl2 = Timeline.from_dict(d)

        assert tl2.get_mode() == tl1.get_mode()
        assert tl2.now_index() == tl1.now_index()
        assert tl2.now() == tl1.now()
        assert tl2.now_time_delta() == tl1.now_time_delta()

    def test_from_dict_backward_compatibility(self):
        """Test from_dict handles old format without mode/cursor fields."""
        d = {
            "type": "Timeline",
            "start": Epoch(2024, 1, 1).to_dict(),
            "stop": Epoch(2024, 1, 2).to_dict(),
            "step": Duration(seconds=60).to_dict(),
        }

        tl = Timeline.from_dict(d)

        assert tl.get_mode() == TimelineMode.BATCH
        assert tl.now_index() == 0

    def test_from_dict_invalid_type(self):
        """Test from_dict raises ValueError for invalid type."""
        d = {
            "type": "NotATimeline",
            "start": Epoch(2024, 1, 1).to_dict(),
            "stop": Epoch(2024, 1, 2).to_dict(),
            "step": Duration(seconds=60).to_dict(),
        }

        with pytest.raises(ValueError, match="Invalid construction dictionary"):
            Timeline.from_dict(d)


class TestTimelineModuleFunctions:
    """Tests for module-level timeline functions."""

    def test_get_time_deltas_function(self):
        """Test get_time_deltas module function."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 1, 0, 5, 0)
        step = Duration(seconds=60)

        time_deltas = get_time_deltas(start, stop, step)

        assert isinstance(time_deltas, np.ndarray)
        assert len(time_deltas) == 6
        assert time_deltas[0] == 0.0
        assert time_deltas[-1] == 300.0

    def test_get_epoch_list_function(self):
        """Test get_epoch_list module function."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 1, 0, 5, 0)
        step = Duration(seconds=60)

        epochs, time_deltas = get_epoch_list(start, stop, step)

        assert isinstance(epochs, np.ndarray)
        assert isinstance(time_deltas, np.ndarray)
        assert len(epochs) == len(time_deltas)
        assert epochs[0] == start


class TestTimelineEdgeCases:
    """Tests for Timeline edge cases and boundary conditions."""

    def test_timeline_with_single_step(self):
        """Test timeline with only one time step."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 1, 0, 1, 0)
        step = Duration(seconds=120)

        tl = Timeline(start, stop, step)
        time_deltas = tl.get_time_deltas()

        assert len(time_deltas) == 1
        assert time_deltas[0] == 0.0

    def test_advance_on_single_step_timeline(self):
        """Test advancing cursor on single-step timeline."""
        tl = Timeline(Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 0, 1, 0), Duration(seconds=120))

        tl.advance(10)

        # Should clamp to index 0 (the only valid index)
        assert tl.now_index() == 0

    def test_timeline_with_very_small_step(self):
        """Test timeline with very small step size."""
        start = Epoch(2024, 1, 1, 0, 0, 0)
        stop = Epoch(2024, 1, 1, 0, 0, 1)
        step = Duration(seconds=0.1)

        tl = Timeline(start, stop, step)
        time_deltas = tl.get_time_deltas()

        assert len(time_deltas) == 11  # 0.0, 0.1, ..., 1.0

    def test_timeline_with_large_range(self):
        """Test timeline with large time range."""
        start = Epoch(2024, 1, 1)
        stop = Epoch(2025, 1, 1)
        step = Duration(days=1)

        tl = Timeline(start, stop, step)
        time_deltas = tl.get_time_deltas()

        # 367 time points (0 to 366 days inclusive) for 2024 (leap year)
        assert len(time_deltas) == 367
