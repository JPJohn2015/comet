# python imports
import numpy as np
import numpy.typing as npt


def rot1(angle: npt.ArrayLike) -> npt.NDArray:
    """Returns the rotation matrix around the 1st axis (X-axis) for the specified angle.

    Rotation matrix for right-handed rotation about the X-axis.

    Args:
        angle (npt.ArrayLike): Angle to be rotated in rad. Can be scalar or array.

    Returns:
        rot (npt.NDArray): 3x3 rotation matrix, or (N, 3, 3) array for N angles
    """
    angle = np.atleast_1d(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    # Construct rotation matrices for batch processing
    N = len(angle)
    rot = np.zeros((N, 3, 3))
    rot[:, 0, 0] = 1.0
    rot[:, 1, 1] = c
    rot[:, 1, 2] = s
    rot[:, 2, 1] = -s
    rot[:, 2, 2] = c

    # Return scalar matrix if input was scalar
    return rot[0] if N == 1 else rot


def rot2(angle: npt.ArrayLike) -> npt.NDArray:
    """Returns the rotation matrix around the 2nd axis (Y-axis) for the specified angle.

    Rotation matrix for right-handed rotation about the Y-axis.

    Args:
        angle (npt.ArrayLike): Angle to be rotated in rad. Can be scalar or array.

    Returns:
        rot (npt.NDArray): 3x3 rotation matrix, or (N, 3, 3) array for N angles
    """
    angle = np.atleast_1d(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    # Construct rotation matrices for batch processing
    N = len(angle)
    rot = np.zeros((N, 3, 3))
    rot[:, 0, 0] = c
    rot[:, 0, 2] = -s
    rot[:, 1, 1] = 1.0
    rot[:, 2, 0] = s
    rot[:, 2, 2] = c

    # Return scalar matrix if input was scalar
    return rot[0] if N == 1 else rot


def rot3(angle: npt.ArrayLike) -> npt.NDArray:
    """Returns the rotation matrix around the 3rd axis (Z-axis) for the specified angle.

    Rotation matrix for right-handed rotation about the Z-axis.

    Args:
        angle (npt.ArrayLike): Angle to be rotated in rad. Can be scalar or array.

    Returns:
        rot (npt.NDArray): 3x3 rotation matrix, or (N, 3, 3) array for N angles
    """
    angle = np.atleast_1d(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    # Construct rotation matrices for batch processing
    N = len(angle)
    rot = np.zeros((N, 3, 3))
    rot[:, 0, 0] = c
    rot[:, 0, 1] = s
    rot[:, 1, 0] = -s
    rot[:, 1, 1] = c
    rot[:, 2, 2] = 1.0

    # Return scalar matrix if input was scalar
    return rot[0] if N == 1 else rot


def rot313(angle1: float, angle2: float, angle3: float) -> npt.NDArray:
    """Returns the composed 3-1-3 Euler angle rotation matrix.

    Applies rotations in order: Z-axis (angle1), X-axis (angle2), Z-axis (angle3).
    This sequence is commonly used for classical Euler angles.

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., pg. 162

    Args:
        angle1 (float): First rotation angle about Z-axis in rad
        angle2 (float): Second rotation angle about X-axis in rad
        angle3 (float): Third rotation angle about Z-axis in rad

    Returns:
        rot (npt.NDArray): 3x3 composed rotation matrix
    """
    return rot3(angle3) @ rot1(angle2) @ rot3(angle1)


def rot321(angle1: float, angle2: float, angle3: float) -> npt.NDArray:
    """Returns the composed 3-2-1 Euler angle rotation matrix.

    Applies rotations in order: Z-axis (angle1), Y-axis (angle2), X-axis (angle3).
    This sequence is commonly used for aerospace applications (yaw-pitch-roll).

    Reference:
        Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., pg. 162

    Args:
        angle1 (float): First rotation angle about Z-axis in rad (yaw)
        angle2 (float): Second rotation angle about Y-axis in rad (pitch)
        angle3 (float): Third rotation angle about X-axis in rad (roll)

    Returns:
        rot (npt.NDArray): 3x3 composed rotation matrix
    """
    return rot1(angle3) @ rot2(angle2) @ rot3(angle1)
