"""Tests for Asset access integration (Phase 6).

Tests the integration of the access framework with the Asset class hierarchy,
covering range computation, access evaluation, and window extraction in all
timeline modes.
"""

# python imports
import pytest
import numpy as np

# COMET imports
from comet.assets.satellite import Satellite
from comet.assets.groundstation import Groundstation
from comet.state.elements import Elements
from comet.state.lla import LLA
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.time.timeline import TIMELINE, TimelineMode
from comet.access.constraints import (
    RangeConstraint,
    EarthLineOfSightConstraint,
)
from comet.access.core import EvalMode


class TestAssetGetRangeTo:
    """Test Asset.get_range_to() in all timeline modes."""

    def test_range_to_single_target_batch_mode(self):
        """Test range to single target in BATCH mode."""
        # Set up timeline
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        # Create satellites
        sat1 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat1',
        )
        sat2 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7500, 0.001, 0, 0, 0, 90),
            name='Sat2',
        )

        # Get range
        ranges = sat1.get_range_to(sat2)

        # Check shape: (T,) for single target in BATCH mode
        assert ranges.ndim == 1
        assert ranges.shape[0] == 6  # 6 time points

        # Check values are positive
        assert np.all(ranges > 0)

    def test_range_to_multiple_targets_batch_mode(self):
        """Test range to multiple targets in BATCH mode."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat1 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat1',
        )
        sat2 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7500, 0.001, 0, 0, 0, 90),
            name='Sat2',
        )
        sat3 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(8000, 0.001, 0, 0, 0, 180),
            name='Sat3',
        )

        # Get range to multiple targets
        ranges = sat1.get_range_to([sat2, sat3])

        # Check shape: (K, T) for multiple targets in BATCH mode
        assert ranges.shape == (2, 6)  # 2 targets, 6 time points
        assert np.all(ranges > 0)

    def test_range_to_stepped_mode(self):
        """Test range in STEPPED mode."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat1 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat1',
        )
        sat2 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7500, 0.001, 0, 0, 0, 90),
            name='Sat2',
        )

        # Switch to STEPPED mode
        TIMELINE.set_mode(TimelineMode.STEPPED)
        TIMELINE.reset()

        # Single target: should return scalar
        range_val = sat1.get_range_to(sat2)
        assert isinstance(range_val, (float, np.floating))
        assert range_val > 0

        # Multiple targets: should return (K,)
        sat3 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(8000, 0.001, 0, 0, 0, 180),
            name='Sat3',
        )
        ranges = sat1.get_range_to([sat2, sat3])
        assert ranges.shape == (2,)
        assert np.all(ranges > 0)

    def test_range_cache_invalidation_on_mode_switch(self):
        """Test that range recomputes after mode switch."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat1 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat1',
        )
        sat2 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7500, 0.001, 0, 0, 0, 90),
            name='Sat2',
        )

        # Get range in BATCH mode
        ranges_batch = sat1.get_range_to(sat2)
        assert ranges_batch.shape == (6,)

        # Switch to STEPPED
        TIMELINE.set_mode(TimelineMode.STEPPED)
        range_stepped = sat1.get_range_to(sat2)
        assert isinstance(range_stepped, (float, np.floating))

        # First value should match
        TIMELINE.reset()
        range_stepped_first = sat1.get_range_to(sat2)
        assert np.isclose(range_stepped_first, ranges_batch[0], rtol=1e-10)


class TestAssetGetAccess:
    """Test Asset.get_access() with constraints in all modes."""

    def test_access_no_constraints_batch_mode(self):
        """Test access with no constraints returns all-pass."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # No constraints - should return all 1.0
        access = sat.get_access(gs, constraints=None)

        assert access.shape == (6,)  # Single target, 6 time points
        assert np.all(access == 1.0)

    def test_access_with_range_constraint(self):
        """Test access with range constraint."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # Range constraint
        constraint = RangeConstraint(min_value=0, max_value=3000)
        access = sat.get_access(gs, constraints=constraint)

        # Should have shape (6,)
        assert access.shape == (6,)

        # Values should be 0.0 or 1.0
        assert np.all((access == 0.0) | (access == 1.0))

    def test_access_multiple_constraints(self):
        """Test access with multiple constraints ANDed together."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 0, 0),
            Duration(minutes=10),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 98, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # Multiple constraints
        constraints = [
            EarthLineOfSightConstraint(),
            RangeConstraint(min_value=0, max_value=3000),
        ]
        access = sat.get_access(gs, constraints=constraints)

        # Should be boolean
        assert np.all((access == 0.0) | (access == 1.0))

    def test_access_stepped_mode(self):
        """Test access in STEPPED mode."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # Get batch access first
        constraint = RangeConstraint(min_value=0, max_value=3000)
        access_batch = sat.get_access(gs, constraints=constraint)

        # Switch to STEPPED
        TIMELINE.set_mode(TimelineMode.STEPPED)
        TIMELINE.reset()

        # Single target should return scalar or 0-d array
        access_stepped = sat.get_access(gs, constraints=constraint)
        # Convert to scalar if it's a 0-d array
        if isinstance(access_stepped, np.ndarray):
            access_stepped = access_stepped.item()
        assert isinstance(access_stepped, (float, np.floating))
        assert access_stepped in [0.0, 1.0]

        # Should match first time point
        assert access_stepped == access_batch[0]

    def test_access_multiple_targets(self):
        """Test access to multiple targets."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 98, 0, 0, 0),
            name='Sat',
        )
        gs1 = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS1',
        )
        gs2 = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(45, 0, 0),
            name='GS2',
        )

        # Multiple targets
        constraint = EarthLineOfSightConstraint()
        access = sat.get_access([gs1, gs2], constraints=constraint)

        # Should have shape (K, T)
        assert access.shape == (2, 6)
        assert np.all((access == 0.0) | (access == 1.0))


class TestAccessWindowExtraction:
    """Test access window extraction methods."""

    def test_get_access_windows_single_window(self):
        """Test extraction of single continuous access window."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # Create synthetic access mask with one window
        access_mask = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

        # Extract windows
        windows = sat.get_access_windows(access_mask)

        # Should have one window
        assert len(windows) == 1
        start, end = windows[0]
        assert isinstance(start, Epoch)
        assert isinstance(end, Epoch)

    def test_get_access_windows_multiple_windows(self):
        """Test extraction of multiple access windows."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=1),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # Create synthetic access mask with two windows
        access_mask = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        # Extract windows
        windows = sat.get_access_windows(access_mask)

        # Should have two windows
        assert len(windows) == 2

    def test_get_access_windows_indices(self):
        """Test extraction of access window indices."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # Create synthetic access mask
        access_mask = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])

        # Extract window indices
        windows = sat.get_access_windows_indices(access_mask)

        # Should have two windows
        assert len(windows) == 2
        assert windows[0] == (1, 2)
        assert windows[1] == (5, 5)

    def test_get_access_windows_requires_batch_mode(self):
        """Test that window extraction requires BATCH mode."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )

        # Switch to STEPPED mode
        TIMELINE.set_mode(TimelineMode.STEPPED)

        # Should raise error
        access_mask = np.array([1.0])
        with pytest.raises(ValueError, match="requires BATCH mode"):
            sat.get_access_windows(access_mask)

    def test_window_extraction_with_multiple_targets(self):
        """Test window extraction from multi-target access mask."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 10, 0),
            Duration(minutes=2),
        )

        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat',
        )
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='GS',
        )

        # Create multi-target access mask (K=2, T=6)
        access_mask = np.array([
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],  # Target 0
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],  # Target 1
        ])

        # Extract for target 0
        windows_t0 = sat.get_access_windows(access_mask, target_idx=0)
        assert len(windows_t0) == 2

        # Extract for target 1
        windows_t1 = sat.get_access_windows(access_mask, target_idx=1)
        assert len(windows_t1) == 1


class TestAssetIntegration:
    """Test integration with actual Satellite and Groundstation assets."""

    def test_satellite_to_groundstation_access(self):
        """Test satellite-to-ground-station access computation."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 1, 30, 0),
            Duration(minutes=10),
        )

        # LEO satellite
        sat = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 98, 0, 0, 0),
            name='LEO',
        )

        # Ground station
        gs = Groundstation.create_groundstation(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            lla=LLA(0, 0, 0),
            name='Equator GS',
        )

        # LOS constraint
        constraint = EarthLineOfSightConstraint()
        access = sat.get_access(gs, constraints=constraint)

        # Should have correct shape and boolean values
        assert access.shape == (10,)  # 10 time points
        assert np.all((access == 0.0) | (access == 1.0))  # Boolean mask

    def test_satellite_to_satellite_range(self):
        """Test inter-satellite range computation."""
        TIMELINE.set_mode(TimelineMode.BATCH)
        TIMELINE.update(
            Epoch(2024, 1, 1, 0, 0, 0),
            Epoch(2024, 1, 1, 0, 30, 0),
            Duration(minutes=5),
        )

        sat1 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 0),
            name='Sat1',
        )
        sat2 = Satellite.create_satellite(
            epoch=Epoch(2024, 1, 1, 0, 0, 0),
            elements=Elements(7000, 0.001, 0, 0, 0, 180),
            name='Sat2',
        )

        ranges = sat1.get_range_to(sat2)

        # Should have expected shape
        assert ranges.shape == (7,)  # 7 time points

        # Range should be reasonable (similar orbits, 180 deg separation)
        assert np.all(ranges > 10000)  # At least 10,000 km apart
        assert np.all(ranges < 15000)  # Less than 15,000 km
