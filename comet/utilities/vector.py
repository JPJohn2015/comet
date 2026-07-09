# python imports
import numpy as np
import numpy.typing as npt


def unit(vector: npt.ArrayLike, axis: int = -1) -> npt.ArrayLike:
    """Calculates the unit vector of the provided vector.

    Args:
        vector (npt.ArrayLike): Vector Array.
        axis (int, optional): Axis specifying Vector. Defaults to -1.

    Returns:
        unit_vector (npt.ArrayLike): Unit Vector Array.
    """
    magnitude = np.linalg.norm(vector, axis=axis, keepdims=True)

    return np.divide(vector, magnitude)


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
    """Dot product of two vectors

    Args:
        vec1 (npt.ArrayLike): Vector Array.
        vec2 (npt.ArrayLike): Vector Array.
        axis (int, optional): Axis specifying Vector. Defaults to -1.

    Returns:
        dot_product (npt.NDArray): Dot Product of the Vectors.
    """
    return np.sum(vec1 * vec2, axis=axis)


if __name__ == "__main__":
    vec1 = np.array([[1, 0, 0], [1, 0, 0]])
    vec2 = np.array([[1, 0, 0], [0, -1, 0]])

    phase_angle_between(vec1, vec2)
