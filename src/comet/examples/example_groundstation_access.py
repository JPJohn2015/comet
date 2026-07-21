#!/usr/bin/env python
"""Example: Groundstation Access Analysis

This example demonstrates how to:
1. Create multiple satellites in different orbits
2. Create ground stations at various locations
3. Set up a timeline for batch analysis
4. Compute access windows between satellites and ground stations
5. Apply constraints (line-of-sight, elevation angle, range)
6. Extract and display access windows

The scenario simulates a constellation of 3 satellites (LEO, MEO, and polar orbit)
communicating with a network of 4 ground stations around the globe.
"""

import numpy as np

# COMET imports
from comet.time import Epoch, Duration, TIMELINE, TimelineMode
from comet.state import Elements, LLA
from comet.assets import Satellite, Groundstation
from comet.access.constraints import (
    EarthLineOfSightConstraint,
    RangeConstraint,
    AngleFromNadirConstraint,
)
from comet.access.core import extract_access_windows


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{text}")
    print("-" * len(text))


def create_satellites():
    """Create a constellation of three satellites with different orbital parameters.

    Returns:
        list[Satellite]: List of three satellites
    """
    print_subheader("Creating Satellite Constellation")

    epoch = Epoch(2024, 1, 1, 0, 0, 0)

    # Satellite 1: Low Earth Orbit (LEO) - ISS-like orbit
    sat1 = Satellite.create_satellite(
        epoch=epoch,
        elements=Elements(
            6778,  # a: Semi-major axis (km) - ~400 km altitude
            0.0001,  # e: Very low eccentricity (nearly circular)
            np.radians(51.6),  # i: Inclination (rad) - ISS inclination
            np.radians(0),  # Ω: Right ascension of ascending node (rad)
            np.radians(0),  # ω: Argument of perigee (rad)
            np.radians(0),  # ν: True anomaly (rad)
        ),
        name="LEO-1",
    )
    print(f"  ✓ {sat1.name}: LEO orbit at ~400 km altitude, i=51.6°")

    # Satellite 2: Medium Earth Orbit (MEO) - GPS-like orbit
    sat2 = Satellite.create_satellite(
        epoch=epoch,
        elements=Elements(
            26560,  # ~20,200 km altitude (GPS orbit)
            0.01,  # Small eccentricity
            np.radians(55),  # GPS-like inclination
            np.radians(90),  # Offset RAAN for coverage diversity
            np.radians(0),
            np.radians(90),  # Offset in orbit
        ),
        name="MEO-1",
    )
    print(f"  ✓ {sat2.name}: MEO orbit at ~20,200 km altitude, i=55°")

    # Satellite 3: Polar orbit for global coverage
    sat3 = Satellite.create_satellite(
        epoch=epoch,
        elements=Elements(
            7078,  # ~700 km altitude
            0.001,  # Nearly circular
            np.radians(98),  # Sun-synchronous polar orbit
            np.radians(180),  # Different RAAN for coverage
            np.radians(0),
            np.radians(180),  # Offset in orbit
        ),
        name="POLAR-1",
    )
    print(f"  ✓ {sat3.name}: Polar orbit at ~700 km altitude, i=98° (sun-sync)")

    return [sat1, sat2, sat3]


def create_groundstations():
    """Create a network of ground stations at strategic global locations.

    Returns:
        list[Groundstation]: List of four ground stations
    """
    print_subheader("Creating Ground Station Network")

    epoch = Epoch(2024, 1, 1, 0, 0, 0)

    # Ground Station 1: Cape Canaveral, Florida, USA
    gs1 = Groundstation.create_groundstation(
        epoch=epoch,
        lla=LLA(
            28.5,  # Latitude (degrees)
            -80.6,  # Longitude (degrees)
            0.01,  # Altitude (km) - sea level
        ),
        name="Cape Canaveral",
    )
    print(f"  ✓ {gs1.name}: Florida, USA (28.5°N, 80.6°W)")

    # Ground Station 2: Svalbard, Norway (high latitude)
    gs2 = Groundstation.create_groundstation(
        epoch=epoch,
        lla=LLA(
            78.2,  # High latitude for polar orbit coverage
            15.4,
            0.05,
        ),
        name="Svalbard",
    )
    print(f"  ✓ {gs2.name}: Norway (78.2°N, 15.4°E) - polar coverage")

    # Ground Station 3: Singapore (equatorial)
    gs3 = Groundstation.create_groundstation(
        epoch=epoch,
        lla=LLA(
            1.35,
            103.8,
            0.01,
        ),
        name="Singapore",
    )
    print(f"  ✓ {gs3.name}: Singapore (1.35°N, 103.8°E) - equatorial")

    # Ground Station 4: Santiago, Chile (southern hemisphere)
    gs4 = Groundstation.create_groundstation(
        epoch=epoch,
        lla=LLA(
            -33.4,
            -70.7,
            0.6,  # ~600m elevation
        ),
        name="Santiago",
    )
    print(f"  ✓ {gs4.name}: Chile (33.4°S, 70.7°W) - southern hemisphere")

    return [gs1, gs2, gs3, gs4]


def setup_timeline():
    """Configure the timeline for batch analysis over a 24-hour period.

    The timeline is set to BATCH mode with 5-minute intervals, giving us
    288 time points over the full day.
    """
    print_subheader("Configuring Timeline")

    # Set timeline to BATCH mode for access analysis
    TIMELINE.set_mode(TimelineMode.BATCH)

    # Set up timeline: 24 hours with 5-minute steps
    start_epoch = Epoch(2024, 1, 1, 0, 0, 0)
    end_epoch = Epoch(2024, 1, 2, 0, 0, 0)  # 24 hours later
    step = Duration(minutes=5)

    TIMELINE.update(start_epoch, end_epoch, step)

    # Get epoch list to calculate number of points
    epochs = TIMELINE.get_epoch_list()
    n_points = len(epochs)
    duration_hours = (end_epoch - start_epoch).total_seconds() / 3600

    print(f"  Mode: BATCH")
    print(f"  Start: {start_epoch}")
    print(f"  End: {end_epoch}")
    print(f"  Step: {step}")
    print(f"  Time points: {n_points}")
    print(f"  Duration: {duration_hours:.1f} hours")


def define_constraints():
    """Define access constraints for the analysis.

    Returns:
        list: List of constraint objects
    """
    print_subheader("Defining Access Constraints")

    constraints = [
        # Line of sight: Earth must not block the path
        EarthLineOfSightConstraint(),
        # Minimum elevation angle: >10° above horizon (typical for ground stations)
        # This is expressed as max angle from nadir for the satellite
        AngleFromNadirConstraint(min_value=0, max_value=80),  # 90-10=80° max nadir angle
        # Maximum range: 3000 km (typical for LEO communications)
        RangeConstraint(min_value=0, max_value=3000),
    ]

    print(f"  ✓ Earth Line of Sight (no blockage)")
    print(f"  ✓ Elevation angle >10° above horizon")
    print(f"  ✓ Range <3000 km")

    return constraints


def analyze_access(satellites, groundstations, constraints):
    """Compute access between all satellite-groundstation pairs.

    Args:
        satellites: List of Satellite objects
        groundstations: List of Groundstation objects
        constraints: List of constraint objects

    Returns:
        dict: Access results keyed by (sat_name, gs_name)
    """
    print_subheader("Computing Access Windows")

    access_results = {}

    for sat in satellites:
        for gs in groundstations:
            # Compute access (returns binary array: 1.0 = access, 0.0 = no access)
            access = sat.get_access(gs, constraints=constraints)

            # Extract continuous access windows (pass TIMELINE object)
            windows = extract_access_windows(access, TIMELINE)

            access_results[(sat.name, gs.name)] = {"access_mask": access, "windows": windows}

            # Calculate statistics
            total_points = len(access)
            access_points = int(np.sum(access))
            access_percent = 100 * access_points / total_points if total_points > 0 else 0
            n_windows = len(windows)

            print(
                f"  {sat.name:12s} → {gs.name:18s}: "
                f"{access_percent:5.1f}% coverage, {n_windows:2d} windows"
            )

    return access_results


def display_detailed_windows(access_results, max_windows=3):
    """Display detailed information about access windows.

    Args:
        access_results: Dictionary of access results
        max_windows: Maximum number of windows to display per pair
    """
    print_subheader("Detailed Access Windows (first 3 per link)")

    for (sat_name, gs_name), result in access_results.items():
        windows = result["windows"]

        if len(windows) == 0:
            print(f"\n{sat_name} → {gs_name}: No access windows")
            continue

        print(f"\n{sat_name} → {gs_name}:")

        for i, (start, end) in enumerate(windows[:max_windows]):
            duration = (end - start).total_seconds() / 60  # minutes
            print(f"  Window {i + 1}: {start} to {end} (duration: {duration:.1f} min)")

        if len(windows) > max_windows:
            print(f"  ... and {len(windows) - max_windows} more windows")


def generate_summary(access_results, satellites, groundstations):
    """Generate a summary of access analysis results.

    Args:
        access_results: Dictionary of access results
        satellites: List of satellites
        groundstations: List of ground stations
    """
    print_header("ACCESS ANALYSIS SUMMARY")

    # Calculate total coverage time for each satellite
    print_subheader("Satellite Coverage Statistics")

    for sat in satellites:
        total_coverage = 0
        total_windows = 0

        for gs in groundstations:
            result = access_results[(sat.name, gs.name)]
            total_coverage += np.sum(result["access_mask"])
            total_windows += len(result["windows"])

        # Calculate percentage (each time point is 5 minutes)
        total_points = len(TIMELINE.get_epoch_list()) * len(groundstations)
        coverage_percent = 100 * total_coverage / total_points if total_points > 0 else 0

        print(
            f"  {sat.name:12s}: {coverage_percent:5.1f}% coverage of network, "
            f"{total_windows:3d} total windows"
        )

    # Calculate coverage for each ground station
    print_subheader("Ground Station Coverage Statistics")

    for gs in groundstations:
        total_coverage = 0
        total_windows = 0

        for sat in satellites:
            result = access_results[(sat.name, gs.name)]
            total_coverage += np.sum(result["access_mask"])
            total_windows += len(result["windows"])

        total_points = len(TIMELINE.get_epoch_list()) * len(satellites)
        coverage_percent = 100 * total_coverage / total_points if total_points > 0 else 0

        print(
            f"  {gs.name:18s}: {coverage_percent:5.1f}% coverage by constellation, "
            f"{total_windows:3d} total passes"
        )

    # Find best and worst links
    print_subheader("Best and Worst Links")

    coverage_by_link = []
    epochs = TIMELINE.get_epoch_list()
    for (sat_name, gs_name), result in access_results.items():
        total_points = len(epochs) if len(epochs) > 0 else 1
        coverage = 100 * np.sum(result["access_mask"]) / total_points
        coverage_by_link.append((sat_name, gs_name, coverage))

    coverage_by_link.sort(key=lambda x: x[2], reverse=True)

    print(
        f"  Best link:  {coverage_by_link[0][0]} → {coverage_by_link[0][1]}: "
        f"{coverage_by_link[0][2]:.1f}% coverage"
    )
    print(
        f"  Worst link: {coverage_by_link[-1][0]} → {coverage_by_link[-1][1]}: "
        f"{coverage_by_link[-1][2]:.1f}% coverage"
    )


def main():
    """Main execution function."""
    print_header("COMET Groundstation Access Analysis Example")

    print("\nThis example demonstrates satellite-to-groundstation access analysis")
    print("using the COMET astrodynamics library. We'll create a small constellation")
    print("and ground station network, then compute when satellites can communicate")
    print("with each station subject to realistic constraints.")

    # Step 1: Create assets
    satellites = create_satellites()
    groundstations = create_groundstations()

    # Step 2: Set up timeline
    setup_timeline()

    # Step 3: Define constraints
    constraints = define_constraints()

    # Step 4: Compute access
    access_results = analyze_access(satellites, groundstations, constraints)

    # Step 5: Display detailed windows
    display_detailed_windows(access_results)

    # Step 6: Generate summary
    generate_summary(access_results, satellites, groundstations)

    print_header("Analysis Complete!")
    print("\nNext steps you could try:")
    print("  • Modify orbital parameters to see how coverage changes")
    print("  • Add more ground stations to improve coverage")
    print("  • Change constraints (e.g., minimum elevation angle)")
    print("  • Extend timeline to analyze longer periods")
    print("  • Export access windows for mission planning\n")


if __name__ == "__main__":
    main()
