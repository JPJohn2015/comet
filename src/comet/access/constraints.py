"""Concrete access constraints for mission analysis.

This module provides all concrete constraint implementations for the COMET access
framework. Each constraint wraps a function from functions.py and applies
min/max bounds to produce access masks.

Users import constraints from this single file:
    from comet.access.constraints import RangeConstraint, SolarPhaseAngleConstraint

All constraints follow the metric-first contract:
    - _metric(inputs) → raw physical values
    - metric(inputs) → public passthrough to raw values
    - access(inputs) → boolean mask after applying bounds
    - eval(inputs) → (metric, access) in one pass

References:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.)
    Wertz, J. R., & Larson, W. J. (1999). Space Mission Analysis and Design (3rd ed.)
"""

# python imports
import numpy as np
import numpy.typing as npt

# COMET imports
from comet.access.core import AccessConstraint, AccessInputs, StateNeed
from comet.access import functions


# ============================================================================
# Line-of-Sight and Illumination Constraints
# ============================================================================


class EarthLineOfSightConstraint(AccessConstraint):
    """Earth line-of-sight constraint between source and target.

    Checks whether a direct line-of-sight path exists between source and target
    without Earth obstruction. Returns 1.0 when LOS is clear, 0.0 when blocked.

    This is a boolean constraint - the metric is the access itself.

    Reference:
        Vallado (2013), Algorithm 35: Line-of-Sight Check, pg. 308-310

    Args:
        body_radius (float, optional): Obscuring body radius in km.
            Defaults to Earth radius (6378.137 km).

    Example:
        >>> constraint = EarthLineOfSightConstraint()
        >>> metric, access = constraint.eval(inputs)
        >>> # metric and access are identical (boolean as float)
    """

    def __init__(self, body_radius: float = None):
        """Initialize Earth line-of-sight constraint.

        Args:
            body_radius (float, optional): Body radius in km. Defaults to Earth radius.
        """
        super().__init__(
            min_value=-np.inf,
            max_value=np.inf,
            needs={StateNeed.SOURCE, StateNeed.TARGET},
        )
        self.body_radius = body_radius

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute Earth LOS access (boolean as float).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: LOS access, shape (N, K, T). 1.0 = clear, 0.0 = blocked.
        """
        kwargs = {}
        if self.body_radius is not None:
            kwargs['body_radius'] = self.body_radius

        return functions.compute_los_access(
            inputs.source_positions,
            inputs.target_positions,
            **kwargs
        )

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Return LOS access directly (no thresholding).

        For boolean constraints, access is the metric itself.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (N, K, T).
        """
        return self._metric(inputs)


class LunarLineOfSightConstraint(AccessConstraint):
    """Lunar line-of-sight constraint between source and target.

    Checks whether the Moon obstructs the line-of-sight path between source
    and target. Returns 1.0 when LOS is clear, 0.0 when Moon blocks.

    This is a boolean constraint - the metric is the access itself.

    Reference:
        Vallado (2013), Algorithm 35: Line-of-Sight Check, pg. 308-310

    Args:
        moon_radius (float, optional): Moon radius in km. Defaults to 1737.4 km.

    Example:
        >>> constraint = LunarLineOfSightConstraint()
        >>> metric, access = constraint.eval(inputs)
    """

    def __init__(self, moon_radius: float = None):
        """Initialize lunar line-of-sight constraint.

        Args:
            moon_radius (float, optional): Moon radius in km. Defaults to 1737.4 km.
        """
        super().__init__(
            min_value=-np.inf,
            max_value=np.inf,
            needs={StateNeed.SOURCE, StateNeed.TARGET, StateNeed.MOON},
        )
        self.moon_radius = moon_radius

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute lunar LOS access (boolean as float).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: LOS access, shape (N, K, T). 1.0 = clear, 0.0 = blocked.
        """
        kwargs = {}
        if self.moon_radius is not None:
            kwargs['moon_radius'] = self.moon_radius

        return functions.compute_los_access_lunar(
            inputs.source_positions,
            inputs.target_positions,
            inputs.moon_position,
            **kwargs
        )

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Return lunar LOS access directly (no thresholding).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (N, K, T).
        """
        return self._metric(inputs)


class TargetIsSunlitConstraint(AccessConstraint):
    """Constraint requiring target to be sunlit (not in Earth shadow).

    Checks whether the target has direct line-of-sight to the Sun (not eclipsed
    by Earth). Returns 1.0 when target is sunlit, 0.0 when in shadow.

    This is a boolean constraint - the metric is the access itself.

    Reference:
        Vallado (2013), Section 5.4: Visibility and Access

    Example:
        >>> constraint = TargetIsSunlitConstraint()
        >>> metric, access = constraint.eval(inputs)
    """

    def __init__(self):
        """Initialize target-sunlit constraint."""
        super().__init__(
            min_value=-np.inf,
            max_value=np.inf,
            needs={StateNeed.TARGET, StateNeed.SUN},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute target sunlit status (boolean as float).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Sunlit status, shape (K, T). 1.0 = sunlit, 0.0 = eclipse.
        """
        # Check LOS from sun to target
        # Sun position is (T, 3), target is (K, T, 3)
        # We need to compute LOS for each target
        los = functions.compute_los_access(
            inputs.sun_position,  # (T, 3) - will be broadcast
            inputs.target_positions,  # (K, T, 3)
        )
        return los

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Return sunlit access directly (no thresholding).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (K, T) or (1, K, T).
        """
        return self._metric(inputs)


class SelfIsSunlitConstraint(AccessConstraint):
    """Constraint requiring source to be sunlit (not in Earth shadow).

    Checks whether the source has direct line-of-sight to the Sun (not eclipsed
    by Earth). Returns 1.0 when source is sunlit, 0.0 when in shadow.

    This is a boolean constraint - the metric is the access itself.

    Reference:
        Vallado (2013), Section 5.4: Visibility and Access

    Example:
        >>> constraint = SelfIsSunlitConstraint()
        >>> metric, access = constraint.eval(inputs)
    """

    def __init__(self):
        """Initialize self-sunlit constraint."""
        super().__init__(
            min_value=-np.inf,
            max_value=np.inf,
            needs={StateNeed.SOURCE, StateNeed.SUN},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute source sunlit status (boolean as float).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Sunlit status, shape (N, T). 1.0 = sunlit, 0.0 = eclipse.
        """
        # Check LOS from sun to source
        los = functions.compute_los_access(
            inputs.sun_position,  # (T, 3)
            inputs.source_positions,  # (N, T, 3)
        )
        return los

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Return sunlit access directly (no thresholding).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (N, T) or (1, N, T).
        """
        return self._metric(inputs)


class SelfIsEclipseConstraint(AccessConstraint):
    """Constraint requiring source to be in eclipse (Earth shadow).

    Inverse of SelfIsSunlitConstraint. Returns 1.0 when source is in shadow,
    0.0 when sunlit.

    This is a boolean constraint - the metric is the access itself.

    Reference:
        Vallado (2013), Section 5.4: Visibility and Access

    Example:
        >>> constraint = SelfIsEclipseConstraint()
        >>> metric, access = constraint.eval(inputs)
    """

    def __init__(self):
        """Initialize self-eclipse constraint."""
        super().__init__(
            min_value=-np.inf,
            max_value=np.inf,
            needs={StateNeed.SOURCE, StateNeed.SUN},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute source eclipse status (boolean as float).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Eclipse status, shape (N, T). 1.0 = eclipse, 0.0 = sunlit.
        """
        # Check LOS from sun to source, then invert
        los = functions.compute_los_access(
            inputs.sun_position,
            inputs.source_positions,
        )
        # Invert: eclipse when NO line-of-sight to sun
        return 1.0 - los

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Return eclipse access directly (no thresholding).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (N, T) or (1, N, T).
        """
        return self._metric(inputs)


# ============================================================================
# Solar and Lunar Geometry Constraints
# ============================================================================


class SolarPhaseAngleConstraint(AccessConstraint):
    """Solar phase angle constraint.

    Constrains the angle at the target between the Sun direction and the source
    direction. 0° = source viewing fully illuminated side, 180° = viewing dark side.

    Reference:
        Wertz & Larson (1999), Section 11.1: Solar Phase Angle

    Args:
        min_value (float, optional): Minimum phase angle in degrees. Defaults to -inf.
        max_value (float, optional): Maximum phase angle in degrees. Defaults to +inf.

    Example:
        >>> # Require source to view illuminated side (< 90°)
        >>> constraint = SolarPhaseAngleConstraint(min_value=0, max_value=90)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize solar phase angle constraint.

        Args:
            min_value (float): Minimum acceptable phase angle (degrees).
            max_value (float): Maximum acceptable phase angle (degrees).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET, StateNeed.SUN},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute solar phase angle in degrees.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Phase angles in degrees, shape (N, K, T). Range [0, 180].
        """
        return functions.compute_solar_phase_ang_deg(
            inputs.source_positions,
            inputs.target_positions,
            inputs.sun_position,
        )


class InPlaneSolarPhaseAngleConstraint(AccessConstraint):
    """In-plane solar phase angle constraint.

    Signed in-plane projection of solar phase angle relative to target's orbit plane.
    Positive = Sun leading target.

    Reference:
        Mission-specific geometry, common in formation flying

    Args:
        min_value (float, optional): Minimum in-plane angle in degrees. Defaults to -inf.
        max_value (float, optional): Maximum in-plane angle in degrees. Defaults to +inf.

    Example:
        >>> # Require Sun to be trailing (negative angle)
        >>> constraint = InPlaneSolarPhaseAngleConstraint(min_value=-180, max_value=0)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize in-plane solar phase angle constraint.

        Args:
            min_value (float): Minimum acceptable in-plane angle (degrees).
            max_value (float): Maximum acceptable in-plane angle (degrees).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET, StateNeed.SUN},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute in-plane solar phase angle in degrees.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: In-plane angles in degrees, shape (N, K, T). Range [-180, 180].
        """
        return functions.compute_in_plane_solar_phase_ang_deg(
            inputs.source_positions,
            inputs.target_positions,
            inputs.sun_position,
        )


class SolarExclusionAngleConstraint(AccessConstraint):
    """Solar exclusion angle constraint.

    Angular separation between line-of-sight to target and line-of-sight to Sun,
    as seen from the source. Used to avoid pointing antenna or sensor at Sun.

    Reference:
        Wertz & Larson (1999), Section 5.3: Angular Separation

    Args:
        min_value (float, optional): Minimum exclusion angle in degrees.
            Typical: 10-30° to avoid Sun contamination. Defaults to -inf.
        max_value (float, optional): Maximum exclusion angle in degrees. Defaults to +inf.

    Example:
        >>> # Require at least 20° separation from Sun
        >>> constraint = SolarExclusionAngleConstraint(min_value=20)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize solar exclusion angle constraint.

        Args:
            min_value (float): Minimum acceptable exclusion angle (degrees).
            max_value (float): Maximum acceptable exclusion angle (degrees).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET, StateNeed.SUN},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute solar exclusion angle in degrees.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Exclusion angles in degrees, shape (N, K, T). Range [0, 180].
        """
        return functions.compute_exclusion_angle_deg(
            inputs.source_positions,
            inputs.target_positions,
            inputs.sun_position,
        )


class LunarExclusionAngleConstraint(AccessConstraint):
    """Lunar exclusion angle constraint.

    Angular separation between line-of-sight to target and line-of-sight to Moon,
    as seen from the source. Used for optical observations requiring lunar avoidance.

    Reference:
        Wertz & Larson (1999), Section 5.3: Angular Separation

    Args:
        min_value (float, optional): Minimum exclusion angle in degrees. Defaults to -inf.
        max_value (float, optional): Maximum exclusion angle in degrees. Defaults to +inf.

    Example:
        >>> # Require at least 15° separation from Moon
        >>> constraint = LunarExclusionAngleConstraint(min_value=15)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize lunar exclusion angle constraint.

        Args:
            min_value (float): Minimum acceptable exclusion angle (degrees).
            max_value (float): Maximum acceptable exclusion angle (degrees).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET, StateNeed.MOON},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute lunar exclusion angle in degrees.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Exclusion angles in degrees, shape (N, K, T). Range [0, 180].
        """
        return functions.compute_exclusion_angle_deg(
            inputs.source_positions,
            inputs.target_positions,
            inputs.moon_position,
        )


class EarthLimbExclusionAngleConstraint(AccessConstraint):
    """Earth limb angle constraint (limb clearance).

    Angular separation between line-of-sight to target and Earth limb (tangent to
    Earth surface) as seen from source. Positive = target above limb (clear),
    negative = target below limb (obstructed).

    Reference:
        Vallado (2013), Section 5.4: Visibility and Access

    Args:
        min_value (float, optional): Minimum limb clearance in degrees.
            Typical: 0° for horizon, positive for clear views. Defaults to -inf.
        max_value (float, optional): Maximum limb clearance in degrees. Defaults to +inf.

    Example:
        >>> # Require target to be at least 5° above horizon
        >>> constraint = EarthLimbExclusionAngleConstraint(min_value=5)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize Earth limb exclusion angle constraint.

        Args:
            min_value (float): Minimum acceptable limb clearance (degrees).
            max_value (float): Maximum acceptable limb clearance (degrees).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute Earth limb clearance angle in degrees.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Limb clearance in degrees, shape (N, K, T).
                Positive = above limb, negative = below limb.
        """
        return functions.compute_earth_limb_angle_deg(
            inputs.source_positions,
            inputs.target_positions,
        )


# ============================================================================
# Range and Rate Constraints
# ============================================================================


class RangeConstraint(AccessConstraint):
    """Range (distance) constraint between source and target.

    Constrains the Euclidean distance between source and target positions.

    Args:
        min_value (float, optional): Minimum acceptable range in km. Defaults to -inf.
        max_value (float, optional): Maximum acceptable range in km. Defaults to +inf.

    Example:
        >>> # Require range between 1000 and 5000 km
        >>> constraint = RangeConstraint(min_value=1000, max_value=5000)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize range constraint.

        Args:
            min_value (float): Minimum acceptable range (km).
            max_value (float): Maximum acceptable range (km).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute range in kilometers.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Ranges in km, shape (N, K, T).
        """
        return functions.compute_range_km(
            inputs.source_positions,
            inputs.target_positions,
        )


class RangeRateConstraint(AccessConstraint):
    """Range rate (radial velocity) constraint.

    Constrains the rate of change of range along the line-of-sight direction.
    Positive = separating (range increasing), negative = approaching.

    Reference:
        Vallado (2013), Section 5.3: Relative Motion

    Args:
        min_value (float, optional): Minimum range rate in km/s. Defaults to -inf.
        max_value (float, optional): Maximum range rate in km/s. Defaults to +inf.

    Example:
        >>> # Require approaching (negative range rate)
        >>> constraint = RangeRateConstraint(min_value=-10, max_value=0)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize range rate constraint.

        Args:
            min_value (float): Minimum acceptable range rate (km/s).
            max_value (float): Maximum acceptable range rate (km/s).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute range rate in km/s.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Range rates in km/s, shape (N, K, T).
                Positive = separating, negative = approaching.
        """
        return functions.compute_range_rate(
            inputs.source_states,
            inputs.target_states,
        )


class AngularRateConstraint(AccessConstraint):
    """Angular rate constraint.

    Constrains the rate of change of the line-of-sight angle as seen from source.
    Useful for tracking requirements and sensor agility constraints.

    Reference:
        Wertz & Larson (1999), Section 5.3.4: Angular Rates

    Args:
        min_value (float, optional): Minimum angular rate in deg/s. Defaults to -inf.
        max_value (float, optional): Maximum angular rate in deg/s. Defaults to +inf.

    Example:
        >>> # Require angular rate below 0.1 deg/s for tracking
        >>> constraint = AngularRateConstraint(max_value=0.1)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize angular rate constraint.

        Args:
            min_value (float): Minimum acceptable angular rate (deg/s).
            max_value (float): Maximum acceptable angular rate (deg/s).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute angular rate in degrees per second.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Angular rates in deg/s, shape (N, K, T).
        """
        return functions.compute_angular_rate(
            inputs.source_states,
            inputs.target_states,
        )


# ============================================================================
# Pointing Constraints
# ============================================================================


class AngleFromNadirConstraint(AccessConstraint):
    """Off-nadir angle constraint.

    Constrains the angle between source's nadir direction (toward Earth center)
    and the line-of-sight to target. 0° = looking straight down, 90° = horizon.

    Reference:
        Wertz & Larson (1999), Section 5.3: Nadir Pointing

    Args:
        min_value (float, optional): Minimum off-nadir angle in degrees. Defaults to -inf.
        max_value (float, optional): Maximum off-nadir angle in degrees. Defaults to +inf.

    Example:
        >>> # Require near-nadir pointing (within 30° of nadir)
        >>> constraint = AngleFromNadirConstraint(max_value=30)
    """

    def __init__(self, min_value: float = -np.inf, max_value: float = np.inf):
        """Initialize off-nadir angle constraint.

        Args:
            min_value (float): Minimum acceptable off-nadir angle (degrees).
            max_value (float): Maximum acceptable off-nadir angle (degrees).
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            needs={StateNeed.SOURCE, StateNeed.TARGET},
        )

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute off-nadir angle in degrees.

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Off-nadir angles in degrees, shape (N, K, T). Range [0, 180].
        """
        return functions.compute_angle_from_nadir_deg(
            inputs.source_positions,
            inputs.target_positions,
        )


# ============================================================================
# Time Window Constraints
# ============================================================================


class EpochConstraint(AccessConstraint):
    """Single epoch window constraint.

    Returns 1.0 (access) when the current epoch is within the specified window,
    0.0 otherwise. This is a boolean constraint operating on timeline epochs.

    Args:
        start_epoch: Start of access window (Epoch object).
        end_epoch: End of access window (Epoch object).

    Example:
        >>> from comet.time import Epoch
        >>> constraint = EpochConstraint(
        ...     start_epoch=Epoch(2024, 1, 1, 0, 0, 0),
        ...     end_epoch=Epoch(2024, 1, 1, 6, 0, 0),
        ... )
    """

    def __init__(self, start_epoch, end_epoch):
        """Initialize epoch window constraint.

        Args:
            start_epoch: Start of window (Epoch).
            end_epoch: End of window (Epoch).
        """
        super().__init__(
            min_value=-np.inf,
            max_value=np.inf,
            needs={StateNeed.TIMELINE},
        )
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute epoch window access (boolean as float).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (T,). 1.0 = in window, 0.0 = outside.
        """
        # Compare epochs
        epochs = inputs.epochs
        in_window = np.array([
            (self.start_epoch <= epoch <= self.end_epoch)
            for epoch in epochs
        ], dtype=float)

        return in_window

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Return epoch window access directly (no thresholding).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (T,).
        """
        return self._metric(inputs)


class EpochWindowsConstraint(AccessConstraint):
    """Multiple epoch windows constraint.

    Returns 1.0 (access) when the current epoch falls within ANY of the specified
    windows, 0.0 otherwise. This is a boolean constraint operating on timeline epochs.

    Args:
        windows: List of (start_epoch, end_epoch) tuples.

    Example:
        >>> from comet.time import Epoch
        >>> windows = [
        ...     (Epoch(2024, 1, 1, 0, 0, 0), Epoch(2024, 1, 1, 6, 0, 0)),
        ...     (Epoch(2024, 1, 1, 12, 0, 0), Epoch(2024, 1, 1, 18, 0, 0)),
        ... ]
        >>> constraint = EpochWindowsConstraint(windows)
    """

    def __init__(self, windows):
        """Initialize multi-window epoch constraint.

        Args:
            windows: List of (start_epoch, end_epoch) tuples.
        """
        super().__init__(
            min_value=-np.inf,
            max_value=np.inf,
            needs={StateNeed.TIMELINE},
        )
        self.windows = windows

    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute multi-window epoch access (boolean as float).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (T,). 1.0 = in any window, 0.0 = outside all.
        """
        epochs = inputs.epochs
        in_any_window = np.zeros(len(epochs), dtype=float)

        # Check each epoch against all windows
        for i, epoch in enumerate(epochs):
            for start, end in self.windows:
                if start <= epoch <= end:
                    in_any_window[i] = 1.0
                    break  # Found a matching window, no need to check others

        return in_any_window

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Return multi-window epoch access directly (no thresholding).

        Args:
            inputs (AccessInputs): Resolved state inputs

        Returns:
            npt.NDArray: Access mask, shape (T,).
        """
        return self._metric(inputs)
