# python imports
import numpy as np
import numpy.typing as npt
import warnings


def unit(vector: npt.ArrayLike, axis: int = -1) -> npt.ArrayLike:
    """Calculates the unit vector of the provided vector.

    Handles zero-magnitude vectors by returning zeros without raising division errors.

    Args:
        vector (npt.ArrayLike): Vector array.
        axis (int, optional): Axis specifying vector. Defaults to -1.

    Returns:
        unit_vector (npt.ArrayLike): Unit vector array. Zero vectors return zero vectors.
    """
    magnitude = np.linalg.norm(vector, axis=axis, keepdims=True)

    # Handle zero-magnitude vectors to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vec = np.divide(vector, magnitude)
        # Replace NaN/Inf with zeros (occurs when magnitude is zero)
        unit_vec = np.where(np.isfinite(unit_vec), unit_vec, 0.0)

    return unit_vec


def angle_between(vec1: npt.ArrayLike, vec2: npt.ArrayLike, axis: int = -1) -> npt.ArrayLike:
    """Calculates the angle between two vectors.

    Args:
        vec1 (npt.ArrayLike): Vector Array.
        vec2 (npt.ArrayLike): Vector Array.
        axis (int, optional): Axis specifying Vector. Defaults to -1.

    Returns:
        angle_between (npt.ArrayLike): Angle Between Vectors.
    """
    # Find Unit Vectors for each vector
    unit1 = unit(vec1, axis=axis)
    unit2 = unit(vec2, axis=axis)

    # Calculate Dot Product
    dot_product = np.sum(unit1 * unit2, axis=axis)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    return np.arccos(dot_product)


def phase_angle_between(vec1: npt.ArrayLike, vec2: npt.ArrayLike, axis: int = -1) -> npt.ArrayLike:
    """Calculates the phase angle between two vectors following right hand rule.
    If the Z component of the cross product of the two vectors is negative, the
    angle is 2pi - angle.

    Args:
        vec1 (npt.ArrayLike): Vector Array.
        vec2 (npt.ArrayLike): Vector Array.
        axis (int, optional): Axis specifying Vector. Defaults to -1.

    Returns:
        phase_angle (npt.ArrayLike): Phase Angle Between Vectors.
    """
    angle = angle_between(vec1, vec2, axis=axis)
    direction = np.cross(vec1, vec2, axis=axis)[..., 2]

    return np.where(direction >= 0, angle, 2 * np.pi - angle)


def dot(vec1: npt.ArrayLike, vec2: npt.ArrayLike, axis: int = -1) -> npt.ArrayLike:
    """Dot product of two vectors.

    Args:
        vec1 (npt.ArrayLike): Vector array.
        vec2 (npt.ArrayLike): Vector array.
        axis (int, optional): Axis specifying vector. Defaults to -1.

    Returns:
        dot_product (npt.NDArray): Dot product of the vectors.
    """
    return np.sum(vec1 * vec2, axis=axis)


def projection(vec1: npt.ArrayLike, vec2: npt.ArrayLike, axis: int = -1) -> npt.ArrayLike:
    """Calculates the vector projection of vec1 onto vec2.

    Returns the component of vec1 in the direction of vec2.

    Reference:
        proj_vec2(vec1) = (vec1 · vec2 / |vec2|²) * vec2

    Args:
        vec1 (npt.ArrayLike): Vector to project.
        vec2 (npt.ArrayLike): Vector to project onto.
        axis (int, optional): Axis specifying vector. Defaults to -1.

    Returns:
        projection (npt.ArrayLike): Vector projection of vec1 onto vec2.
    """
    vec2_arr = np.asarray(vec2)
    dot_product = dot(vec1, vec2, axis=axis)
    vec2_mag_sq = dot(vec2, vec2, axis=axis)

    # Avoid division by zero for zero-magnitude vec2
    with np.errstate(divide='ignore', invalid='ignore'):
        scalar = np.divide(dot_product, vec2_mag_sq)
        scalar = np.where(np.isfinite(scalar), scalar, 0.0)

    # Expand dims to broadcast correctly
    if axis == -1:
        scalar = np.expand_dims(scalar, axis=-1)
    else:
        scalar = np.expand_dims(scalar, axis=axis)

    return scalar * vec2_arr


def rejection(vec1: npt.ArrayLike, vec2: npt.ArrayLike, axis: int = -1) -> npt.ArrayLike:
    """Calculates the vector rejection of vec1 from vec2.

    Returns the component of vec1 perpendicular to vec2.

    Reference:
        rej_vec2(vec1) = vec1 - proj_vec2(vec1)

    Args:
        vec1 (npt.ArrayLike): Vector to reject.
        vec2 (npt.ArrayLike): Vector to reject from.
        axis (int, optional): Axis specifying vector. Defaults to -1.

    Returns:
        rejection (npt.ArrayLike): Vector rejection of vec1 from vec2.
    """
    return vec1 - projection(vec1, vec2, axis=axis)


if __name__ == "__main__":
    vec1 = np.array([[1, 0, 0], [1, 0, 0]])
    vec2 = np.array([[1, 0, 0], [0, -1, 0]])

    phase_angle_between(vec1, vec2)
