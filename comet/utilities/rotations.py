# python imports
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------------
def rot1(angle: float|int):
    """Returns the rotation matrix around the 1st Axis for the specified angle.

    Args:
        angle (float|int): Amgle to be rotated in rad.

    Returns:
        rot (np.ndarray): 3x3 Rotation Matrix
    """
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), np.sin(angle)],
        [0.0, -np.sin(angle), np.cos(angle)]
    ])
    return rot

# ---------------------------------------------------------------------------------------------------------------------------
def rot2(angle: float|int):
    """Returns the rotation matrix around the 2nd Axis for the specified angle.

    Args:
        angle (float|int): Amgle to be rotated in rad.

    Returns:
        rot (np.ndarray): 3x3 Rotation Matrix
    """
    rot = np.array([
        [np.cos(angle), 0.0, -np.sin(angle)],
        [0.0, 1.0, 0.0],
        [np.sin(angle), 0.0, np.cos(angle)]
    ])
    return rot

# ---------------------------------------------------------------------------------------------------------------------------
def rot3(angle: float|int):
    """Returns the rotation matrix around the 3rd Axis for the specified angle.

    Args:
        angle (float|int): Amgle to be rotated in rad.

    Returns:
        rot (np.ndarray): 3x3 Rotation Matrix
    """
    rot = np.array([
        [np.cos(angle), np.sin(angle), 0.0],
        [-np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return rot