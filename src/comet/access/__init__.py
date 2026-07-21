"""COMET Access & Constraints Framework.

A vectorized framework for computing line-of-sight, geometric access, and constraints
between platforms (satellites, ground stations, celestial bodies) and their payloads.

Design Principles:
    1. Metric-first: Every constraint computes a physical value and thresholds it
    2. Single-pass batch evaluation: No per-source/target Python loops
    3. Flat function module: Pure math in functions.py, constraints in constraints.py
    4. Three evaluation modes: CURRENT, CUSTOM, BATCH

Typical Usage:
    From functions.py (pure math):
        >>> from comet.access.functions import compute_solar_phase_ang_deg
        >>> phase_angle = compute_solar_phase_ang_deg(source_pos, target_pos, sun_pos)

    From constraints.py (thresholded access):
        >>> from comet.access.constraints import SolarPhaseAngleConstraint
        >>> constraint = SolarPhaseAngleConstraint(min_value=0, max_value=90)
        >>> metric, access = constraint.eval(inputs)

Module Structure:
    functions.py    - Vectorized geometric functions (no thresholding)
    core.py         - Base classes, enums, resolver, batch engine
    constraints.py  - Concrete constraints (to be implemented in Phase 3)
"""

from comet.access.core import (
    AccessConstraint,
    AccessInputs,
    CustomStates,
    EvalMode,
    StateNeed,
    resolve_inputs,
    evaluate_composite_access,
    extract_access_windows,
    extract_access_windows_indices,
)

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

__all__ = [
    # Core classes and enums
    "AccessConstraint",
    "AccessInputs",
    "CustomStates",
    "EvalMode",
    "StateNeed",
    # Evaluation functions
    "resolve_inputs",
    "evaluate_composite_access",
    "extract_access_windows",
    "extract_access_windows_indices",
    # Line-of-sight and illumination constraints
    "EarthLineOfSightConstraint",
    "LunarLineOfSightConstraint",
    "TargetIsSunlitConstraint",
    "SelfIsSunlitConstraint",
    "SelfIsEclipseConstraint",
    # Solar and lunar geometry constraints
    "SolarPhaseAngleConstraint",
    "InPlaneSolarPhaseAngleConstraint",
    "SolarExclusionAngleConstraint",
    "LunarExclusionAngleConstraint",
    "EarthLimbExclusionAngleConstraint",
    # Range and rate constraints
    "RangeConstraint",
    "RangeRateConstraint",
    "AngularRateConstraint",
    # Pointing constraints
    "AngleFromNadirConstraint",
    # Time window constraints
    "EpochConstraint",
    "EpochWindowsConstraint",
]
