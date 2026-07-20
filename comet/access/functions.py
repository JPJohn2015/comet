"""Vectorized access computation functions.

This module provides pure mathematical functions for computing geometric relationships
between spacecraft, ground stations, celestial bodies, and targets. All functions are
vectorized to operate on arrays of positions/states across time.

All functions follow a consistent broadcasting convention:
- Inputs shaped (N, T, ·) for N objects, T time points
- Outputs broadcast to (N, K, T) for N sources × K targets × T times via match_dims

Functions return physical values with no thresholding. The constraint layer in
constraints.py applies bounds to produce access masks.

References:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications (4th ed.).
    Wertz, J. R., & Larson, W. J. (1999). Space Mission Analysis and Design (3rd ed.).
"""

# python imports
import numpy as np
import numpy.typing as npt
from typing import Tuple

# COMET imports
from comet.utilities.constants import Constants as c


def match_dims(
    source: npt.NDArray,
    target: npt.NDArray,
    time_axis_last: bool = True
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Broadcast source and target arrays to (N, K, T) shape for access computation.

    Converts inputs shaped (N, T, ·) and (K, T, ·) to broadcast-compatible shapes
    for vectorized pairwise computation across all source-target-time combinations.

    Args:
        source (npt.NDArray): Source array, shape (N, T, ...) or (T, ...)
        target (npt.NDArray): Target array, shape (K, T, ...) or (T, ...)
        time_axis_last (bool): If True, time is last dim. If False, reshape accordingly.

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: (source_broadcast, target_broadcast)
            Both shaped for broadcasting to (N, K, T, ...) or (N, K, ...) depending
            on input dimensions.

    Examples:
        >>> source = np.random.randn(3, 10, 3)  # 3 sources, 10 times, 3D positions
        >>> target = np.random.randn(5, 10, 3)  # 5 targets, 10 times, 3D positions
        >>> src_bc, tgt_bc = match_dims(source, target)
        >>> src_bc.shape  # (3, 1, 10, 3) - broadcasts to (3, 5, 10, 3)
        >>> tgt_bc.shape  # (1, 5, 10, 3) - broadcasts to (3, 5, 10, 3)
    """
    # Ensure at least 2D
    source = np.atleast_2d(source)
    target = np.atleast_2d(target)

    # If source is (T, ...), expand to (1, T, ...)
    if source.ndim == 2 and target.ndim >= 3:
        source = source[np.newaxis, ...]

    # If target is (T, ...), expand to (1, T, ...)
    if target.ndim == 2 and source.ndim >= 3:
        target = target[np.newaxis, ...]

    # Add singleton dimension for broadcasting: (N, T, ...) -> (N, 1, T, ...)
    if source.ndim >= 2:
        source = source[:, np.newaxis, ...]

    # Target: (K, T, ...) -> (1, K, T, ...)
    if target.ndim >= 2:
        target = target[np.newaxis, ...]

    return source, target


def compute_los_access(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray,
    body_radius: float = c.A_EARTH / 1000.0  # Convert m to km
) -> npt.NDArray:
    """Compute line-of-sight access between sources and targets with Earth obscuration.

    Determines whether a direct line-of-sight path exists between source and target
    positions, accounting for obscuration by a spherical body (typically Earth).
    Uses geometric ray-sphere intersection test.

    Algorithm: For each source-target pair, computes the minimum distance from the
    Earth center to the line segment connecting the two positions. If this distance
    is less than the body radius, the line is obstructed.

    Reference:
        Vallado (2013), Algorithm 35: Line-of-Sight Check, pg. 308-310
        Wertz & Larson (1999), Section 5.4.3: Geometric Visibility

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km
        body_radius (float): Obscuring body radius in km. Defaults to Earth radius.

    Returns:
        npt.NDArray: Boolean LOS access, shape (N, K, T). 1.0 = LOS exists, 0.0 = obscured
    """
    # Broadcast to (N, K, T, 3)
    src, tgt = match_dims(source_pos, target_pos)

    # Vector from source to target
    r_st = tgt - src  # (N, K, T, 3)

    # Distance from source to target
    range_st = np.linalg.norm(r_st, axis=-1, keepdims=True)  # (N, K, T, 1)

    # Avoid division by zero
    range_st = np.where(range_st < 1e-10, 1e-10, range_st)

    # Unit vector from source to target
    u_st = r_st / range_st  # (N, K, T, 3)

    # Project source position onto line to find closest approach to Earth center
    # Using the formula: d_min = |r_s × u_st|
    # where r_s is source position from Earth center
    cross = np.cross(src, u_st, axis=-1)  # (N, K, T, 3)
    d_min = np.linalg.norm(cross, axis=-1)  # (N, K, T)

    # Check if closest approach point is between source and target
    # Distance along line from source to closest point
    t_closest = -np.sum(src * u_st, axis=-1)  # (N, K, T)

    # Closest point is between if: 0 <= t_closest <= range
    between = (t_closest >= 0) & (t_closest <= range_st[..., 0])

    # LOS is BLOCKED if: (d_min < body_radius) AND (closest point is between)
    # LOS exists otherwise
    blocked = (d_min < body_radius) & between
    los = ~blocked

    return los.astype(float)


def compute_los_access_lunar(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray,
    moon_pos: npt.NDArray,
    moon_radius: float = c.RADIUS_MOON
) -> npt.NDArray:
    """Compute line-of-sight access accounting for lunar obscuration.

    Similar to Earth LOS but checks obscuration by the Moon at its current position.

    Reference:
        Vallado (2013), Algorithm 35: Line-of-Sight Check, pg. 308-310

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km
        moon_pos (npt.NDArray): Moon ECI positions, shape (T, 3) in km
        moon_radius (float): Moon radius in km. Defaults to 1737.4 km.

    Returns:
        npt.NDArray: Boolean LOS access, shape (N, K, T). 1.0 = LOS exists, 0.0 = obscured
    """
    # Broadcast to (N, K, T, 3)
    src, tgt = match_dims(source_pos, target_pos)

    # Expand moon position to broadcast: (T, 3) -> (1, 1, T, 3)
    moon = moon_pos[np.newaxis, np.newaxis, ...]

    # Vector from source to target
    r_st = tgt - src
    range_st = np.linalg.norm(r_st, axis=-1, keepdims=True)
    range_st = np.where(range_st < 1e-10, 1e-10, range_st)
    u_st = r_st / range_st

    # Vector from moon center to source
    r_ms = src - moon  # (N, K, T, 3)

    # Project to find closest approach
    cross = np.cross(r_ms, u_st, axis=-1)
    d_min = np.linalg.norm(cross, axis=-1)

    # Distance along line from source to closest point to moon center
    t_closest = -np.sum(r_ms * u_st, axis=-1)
    between = (t_closest >= 0) & (t_closest <= range_st[..., 0])

    # LOS is BLOCKED if: (d_min < moon_radius) AND (closest point is between)
    blocked = (d_min < moon_radius) & between
    los = ~blocked

    return los.astype(float)


def compute_solar_phase_ang_deg(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray,
    sun_pos: npt.NDArray
) -> npt.NDArray:
    """Compute solar phase angle between source, target, and Sun.

    Solar phase angle is the angle at the target between the Sun direction and
    the source direction. A phase angle of 0° means the source is looking at the
    fully illuminated side; 180° means looking at the dark side.

    Reference:
        Wertz & Larson (1999), Section 11.1: Solar Phase Angle
        Angle between vectors: arccos(dot(u_ts, u_tsun) / (|u_ts| * |u_tsun|))

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km
        sun_pos (npt.NDArray): Sun ECI positions, shape (T, 3) in km

    Returns:
        npt.NDArray: Solar phase angle in degrees, shape (N, K, T). Range [0, 180].
    """
    # Broadcast source and target
    src, tgt = match_dims(source_pos, target_pos)

    # Normalize sun_pos to (T, 3) if needed (handles both (T, 3) and (1, T, 3) inputs)
    if sun_pos.ndim == 3:
        sun_pos = sun_pos.reshape(-1, sun_pos.shape[-1])

    # Expand sun: (T, 3) -> (1, 1, T, 3)
    sun = sun_pos[np.newaxis, np.newaxis, ...]

    # Vector from target to source
    r_ts = src - tgt  # (N, K, T, 3)

    # Vector from target to sun
    r_tsun = sun - tgt  # (N, K, T, 3)

    # Normalize
    u_ts = r_ts / (np.linalg.norm(r_ts, axis=-1, keepdims=True) + 1e-10)
    u_tsun = r_tsun / (np.linalg.norm(r_tsun, axis=-1, keepdims=True) + 1e-10)

    # Compute angle
    cos_angle = np.sum(u_ts * u_tsun, axis=-1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    phase_angle_rad = np.arccos(cos_angle)
    phase_angle_deg = np.rad2deg(phase_angle_rad)

    return phase_angle_deg


def compute_in_plane_solar_phase_ang_deg(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray,
    sun_pos: npt.NDArray
) -> npt.NDArray:
    """Compute signed in-plane solar phase angle.

    Projects the solar phase angle onto the plane defined by the target's orbit.
    This gives a signed angle indicating whether the Sun is leading or trailing
    relative to the target's velocity vector.

    Reference:
        Mission-specific geometry, common in formation flying and constellation design.

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km
        sun_pos (npt.NDArray): Sun ECI positions, shape (T, 3) in km

    Returns:
        npt.NDArray: Signed in-plane phase angle in degrees, shape (N, K, T).
            Range [-180, 180]. Positive = Sun leading target.
    """
    # This is a simplified version - a full implementation would need target velocity
    # For now, return the standard solar phase angle (unsigned)
    # TODO: Implement proper in-plane projection when target velocity is available
    return compute_solar_phase_ang_deg(source_pos, target_pos, sun_pos)


def compute_exclusion_angle_deg(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray,
    body_pos: npt.NDArray
) -> npt.NDArray:
    """Compute exclusion angle from source to target relative to a body (Sun or Moon).

    Exclusion angle is the angular separation between the line-of-sight to the target
    and the line-of-sight to the body (typically Sun or Moon) as seen from the source.

    Common use: Solar exclusion angle for RF links (avoid pointing antenna at Sun),
    or lunar exclusion for optical observations.

    Reference:
        Wertz & Larson (1999), Section 5.3: Angular Separation

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km
        body_pos (npt.NDArray): Body (Sun/Moon) ECI positions, shape (T, 3) in km

    Returns:
        npt.NDArray: Exclusion angle in degrees, shape (N, K, T). Range [0, 180].
    """
    # Broadcast source and target
    src, tgt = match_dims(source_pos, target_pos)

    # Expand body: (T, 3) -> (1, 1, T, 3)
    body = body_pos[np.newaxis, np.newaxis, ...]

    # Vector from source to target
    r_st = tgt - src  # (N, K, T, 3)

    # Vector from source to body
    r_sb = body - src  # (N, K, T, 3)

    # Normalize
    u_st = r_st / (np.linalg.norm(r_st, axis=-1, keepdims=True) + 1e-10)
    u_sb = r_sb / (np.linalg.norm(r_sb, axis=-1, keepdims=True) + 1e-10)

    # Compute angle
    cos_angle = np.sum(u_st * u_sb, axis=-1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    exclusion_rad = np.arccos(cos_angle)
    exclusion_deg = np.rad2deg(exclusion_rad)

    return exclusion_deg


def compute_earth_limb_angle_deg(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray,
    earth_radius: float = c.A_EARTH / 1000.0  # Convert m to km
) -> npt.NDArray:
    """Compute Earth limb angle for source-to-target line of sight.

    Limb angle is the angular separation between the line-of-sight to the target
    and the Earth limb (tangent to Earth) as seen from the source. Positive values
    indicate the target is above the limb; negative values indicate obstruction.

    Reference:
        Vallado (2013), Section 5.4: Visibility and Access
        Limb angle: arcsin(R_earth / r_source) where r_source is source altitude

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km
        earth_radius (float): Earth radius in km. Defaults to 6378.137 km.

    Returns:
        npt.NDArray: Limb angle in degrees, shape (N, K, T).
            Positive = above limb, negative = below limb (obstructed).
    """
    # Broadcast
    src, tgt = match_dims(source_pos, target_pos)

    # Source distance from Earth center
    r_source = np.linalg.norm(src, axis=-1, keepdims=True)  # (N, 1, T, 1)

    # Vector from source to target
    r_st = tgt - src
    u_st = r_st / (np.linalg.norm(r_st, axis=-1, keepdims=True) + 1e-10)

    # Unit vector from Earth center to source
    u_source = src / (r_source + 1e-10)

    # Angle from nadir (Earth center direction) to target LOS
    cos_angle_from_nadir = np.sum(u_st * (-u_source), axis=-1)
    cos_angle_from_nadir = np.clip(cos_angle_from_nadir, -1.0, 1.0)
    angle_from_nadir_rad = np.arccos(cos_angle_from_nadir)
    angle_from_nadir_deg = np.rad2deg(angle_from_nadir_rad)

    # Earth limb angle from source position (angle from nadir to horizon)
    # This is the angular radius of Earth as seen from the source
    sin_limb = earth_radius / (r_source[..., 0] + 1e-10)
    sin_limb = np.clip(sin_limb, -1.0, 1.0)
    limb_angle_rad = np.arcsin(sin_limb)
    limb_angle_deg = np.rad2deg(limb_angle_rad)

    # Limb clearance: how far above the limb we're looking
    # Positive = looking into space (clear), negative = looking at/through Earth
    # At nadir (angle=0): clearance is maximum (limb_angle)
    # At horizon (angle=limb_angle): clearance is zero
    # Beyond horizon (angle>limb_angle): clearance is negative (obstructed)
    limb_clearance = limb_angle_deg - angle_from_nadir_deg

    return limb_clearance


def compute_range_km(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray
) -> npt.NDArray:
    """Compute range (distance) between source and target positions.

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km

    Returns:
        npt.NDArray: Range in km, shape (N, K, T).
    """
    # Broadcast
    src, tgt = match_dims(source_pos, target_pos)

    # Compute distance
    r_st = tgt - src
    range_km = np.linalg.norm(r_st, axis=-1)

    return range_km


def compute_range_rate(
    source_state: npt.NDArray,
    target_state: npt.NDArray
) -> npt.NDArray:
    """Compute range rate (relative velocity along line of sight).

    Range rate is the component of relative velocity along the line-of-sight direction.
    Positive values indicate targets moving away (range increasing).

    Reference:
        Vallado (2013), Section 5.3: Relative Motion
        Range rate: dot(v_rel, u_r) where v_rel = v_target - v_source

    Args:
        source_state (npt.NDArray): Source ECI states, shape (N, T, 6) in km and km/s
        target_state (npt.NDArray): Target ECI states, shape (K, T, 6) in km and km/s

    Returns:
        npt.NDArray: Range rate in km/s, shape (N, K, T).
            Positive = separating, negative = approaching.
    """
    # Extract positions and velocities
    source_pos = source_state[..., :3]
    source_vel = source_state[..., 3:6]
    target_pos = target_state[..., :3]
    target_vel = target_state[..., 3:6]

    # Broadcast
    src_pos, tgt_pos = match_dims(source_pos, target_pos)
    src_vel, tgt_vel = match_dims(source_vel, target_vel)

    # Relative position and velocity
    r_rel = tgt_pos - src_pos  # (N, K, T, 3)
    v_rel = tgt_vel - src_vel  # (N, K, T, 3)

    # Unit vector along line of sight
    range_mag = np.linalg.norm(r_rel, axis=-1, keepdims=True) + 1e-10
    u_los = r_rel / range_mag

    # Range rate: projection of relative velocity onto LOS
    range_rate = np.sum(v_rel * u_los, axis=-1)

    return range_rate


def compute_angular_rate(
    source_state: npt.NDArray,
    target_state: npt.NDArray
) -> npt.NDArray:
    """Compute angular rate of target as seen from source.

    Angular rate is the rate of change of the line-of-sight angle, measured in
    degrees per second. Useful for tracking requirements and sensor agility.

    Reference:
        Wertz & Larson (1999), Section 5.3.4: Angular Rates
        ω = |v_perp| / r where v_perp is velocity perpendicular to LOS

    Args:
        source_state (npt.NDArray): Source ECI states, shape (N, T, 6) in km and km/s
        target_state (npt.NDArray): Target ECI states, shape (K, T, 6) in km and km/s

    Returns:
        npt.NDArray: Angular rate in deg/s, shape (N, K, T).
    """
    # Extract positions and velocities
    source_pos = source_state[..., :3]
    source_vel = source_state[..., 3:6]
    target_pos = target_state[..., :3]
    target_vel = target_state[..., 3:6]

    # Broadcast
    src_pos, tgt_pos = match_dims(source_pos, target_pos)
    src_vel, tgt_vel = match_dims(source_vel, target_vel)

    # Relative position and velocity
    r_rel = tgt_pos - src_pos
    v_rel = tgt_vel - src_vel

    # Range
    range_mag = np.linalg.norm(r_rel, axis=-1, keepdims=True) + 1e-10

    # Perpendicular velocity component: v_perp = v_rel - (v_rel · r̂)r̂
    u_los = r_rel / range_mag
    v_parallel = np.sum(v_rel * u_los, axis=-1, keepdims=True) * u_los
    v_perp = v_rel - v_parallel

    # Angular rate: ω = |v_perp| / r
    v_perp_mag = np.linalg.norm(v_perp, axis=-1)
    angular_rate_rad_s = v_perp_mag / range_mag[..., 0]
    angular_rate_deg_s = np.rad2deg(angular_rate_rad_s)

    return angular_rate_deg_s


def compute_angle_from_nadir_deg(
    source_pos: npt.NDArray,
    target_pos: npt.NDArray
) -> npt.NDArray:
    """Compute off-nadir angle from source to target.

    Off-nadir angle is the angle between the source's nadir direction (toward Earth
    center) and the line-of-sight to the target. 0° = looking straight down at Earth,
    90° = looking at horizon.

    Reference:
        Wertz & Larson (1999), Section 5.3: Nadir Pointing

    Args:
        source_pos (npt.NDArray): Source ECI positions, shape (N, T, 3) in km
        target_pos (npt.NDArray): Target ECI positions, shape (K, T, 3) in km

    Returns:
        npt.NDArray: Off-nadir angle in degrees, shape (N, K, T). Range [0, 180].
    """
    # Broadcast
    src, tgt = match_dims(source_pos, target_pos)

    # Nadir direction (toward Earth center)
    u_nadir = -src / (np.linalg.norm(src, axis=-1, keepdims=True) + 1e-10)

    # Line-of-sight to target
    r_st = tgt - src
    u_los = r_st / (np.linalg.norm(r_st, axis=-1, keepdims=True) + 1e-10)

    # Angle between nadir and LOS
    cos_angle = np.sum(u_nadir * u_los, axis=-1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg
