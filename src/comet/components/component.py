# python imports
import numpy as np
import itertools
from typing import Optional

# COMET imports
from comet.time import Epoch
from comet.frames import StateFrame


class Component:
    """Base Class that defines any component that can be attached to an Asset."""

    # Component ID Counter
    id_counter = itertools.count()

    def __init__(
        self,
        mass: float = 0.0,
        body_vector: np.ndarray = [1, 0, 0],
        name: str = None,
        parent_to_component_rot: Optional[np.ndarray] = None,
    ):
        """Construct Base Component.

        Args:
            mass (float, optional): Component mass in kg. Defaults to 0.0.
            body_vector (np.ndarray, optional): Pointing Vector in Body Frame. Defaults to [1, 0, 0].
            name (str, optional): Component name. Defaults to None.
            parent_to_component_rot (np.ndarray, optional): 3x3 rotation matrix from parent to component frame.
                Defaults to identity matrix.
        """
        # Assign Class attributes
        self.mass = mass
        self.body_vector = np.array(body_vector)

        # Assign Component ID and name
        self.id = next(Component.id_counter)
        self.name = name if name is not None else f"Component_{self.id}"

        # Parent asset reference
        self.parent = None

        # Initialize rotation matrix
        self._parent_to_component_rot = None
        if parent_to_component_rot is not None:
            self.set_parent_to_component_rot(parent_to_component_rot)
        else:
            # Default to identity rotation
            self._parent_to_component_rot = np.eye(3)

    def get_mass(self):
        """Returns Component mass in kg.

        Returns:
            mass (float): Component mass in kg.
        """
        return self.mass

    def get_body_vector(self):
        """Returns the pointing vector of the component in the Body Frame.

        Returns:
            body_vector (np.ndarray): Pointing Vector in Body Frame.
        """
        return self.body_vector

    def get_id(self):
        """Returns the Component ID.

        Returns:
            id (int): Component ID.
        """
        return self.id

    def get_name(self):
        """Returns the Component name.

        Returns:
            name (str): Component name.
        """
        return self.name

    def set_parent_to_component_rot(self, rotation: np.ndarray):
        """Set the rotation matrix from parent frame to component frame.

        Args:
            rotation (np.ndarray): 3x3 rotation matrix.

        Raises:
            ValueError: If rotation is not a valid 3x3 rotation matrix.
        """
        rotation = np.array(rotation)

        # Validate shape
        if rotation.shape != (3, 3):
            raise ValueError(
                f"Component.set_parent_to_component_rot(): rotation must be 3x3, got {rotation.shape}"
            )

        # Validate orthogonality (R @ R.T should be identity)
        identity_check = rotation @ rotation.T
        if not np.allclose(identity_check, np.eye(3), atol=1e-6):
            raise ValueError(
                "Component.set_parent_to_component_rot(): rotation must be orthogonal (R @ R.T = I)"
            )

        # Validate determinant is +1 (proper rotation, not reflection)
        det = np.linalg.det(rotation)
        if not np.isclose(det, 1.0, atol=1e-6):
            raise ValueError(
                f"Component.set_parent_to_component_rot(): rotation determinant must be +1, got {det:.6f}"
            )

        self._parent_to_component_rot = rotation

    def get_parent_to_component_rot(self) -> np.ndarray:
        """Get the rotation matrix from parent frame to component frame.

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        return self._parent_to_component_rot.copy()

    # Mode-transparent state pass-through methods
    def get_state(self, frame: str | StateFrame = StateFrame.ECI):
        """Get component state by deferring to parent asset.

        Mode-transparent: returns 2D array in BATCH mode, 1D array in STEPPED mode,
        automatically based on parent's timeline mode.

        Args:
            frame (str|StateFrame, optional): StateFrame to return state. Defaults to StateFrame.ECI.

        Returns:
            np.ndarray: State data (position and velocity).
                BATCH mode: (n_time, 6) array
                STEPPED mode: (6,) array

        Raises:
            ValueError: If component has no parent.
        """
        if self.parent is None:
            raise ValueError("Component.get_state(): Component has no parent asset")

        return self.parent.get_state(frame)

    def get_position(self, frame: str | StateFrame = StateFrame.ECI):
        """Get component position by deferring to parent asset.

        Mode-transparent: returns 2D array in BATCH mode, 1D array in STEPPED mode.

        Args:
            frame (str|StateFrame, optional): StateFrame to return position. Defaults to StateFrame.ECI.

        Returns:
            np.ndarray: Position data.
                BATCH mode: (n_time, 3) array
                STEPPED mode: (3,) array

        Raises:
            ValueError: If component has no parent.
        """
        if self.parent is None:
            raise ValueError("Component.get_position(): Component has no parent asset")

        return self.parent.get_position(frame)

    def get_velocity(self, frame: str | StateFrame = StateFrame.ECI):
        """Get component velocity by deferring to parent asset.

        Mode-transparent: returns 2D array in BATCH mode, 1D array in STEPPED mode.

        Args:
            frame (str|StateFrame, optional): StateFrame to return velocity. Defaults to StateFrame.ECI.

        Returns:
            np.ndarray: Velocity data.
                BATCH mode: (n_time, 3) array
                STEPPED mode: (3,) array

        Raises:
            ValueError: If component has no parent.
        """
        if self.parent is None:
            raise ValueError("Component.get_velocity(): Component has no parent asset")

        return self.parent.get_velocity(frame)

    def get_access(
        self,
        targets,
        parent_constraints=None,
        component_constraints=None
    ):
        """Get composite access (parent-access × component-access).

        Evaluates composite access by ANDing parent geometric access with
        component-specific constraints. This allows components (e.g., sensors)
        to gate their parent's access with additional constraints.

        Args:
            targets: Single Asset or list of Assets
            parent_constraints: Constraints for parent geometric access (optional)
            component_constraints: Additional constraints specific to component (optional)

        Returns:
            np.ndarray: Composite access mask (0.0 or 1.0). Shape depends on mode:
                - BATCH, multiple: (n_targets, n_times)
                - BATCH, single: (n_times,)
                - STEPPED, multiple: (n_targets,)
                - STEPPED, single: scalar float

        Raises:
            ValueError: If component has no parent.
        """
        if self.parent is None:
            raise ValueError("Component.get_access(): Component has no parent asset")

        # Get parent's geometric access
        parent_access = self.parent.get_access(targets, parent_constraints)

        # If no component constraints, return parent access
        if component_constraints is None:
            return parent_access

        # Evaluate component constraints
        component_access = self.parent.get_access(targets, component_constraints)

        # Return composite: parent AND component
        return parent_access * component_access

    def to_dict(self):
        """Serialize Component to dictionary.

        Returns:
            dict: Dictionary representation of Component.
        """
        return {
            "type": "Component",
            "id": self.id,
            "name": self.name,
            "mass": self.mass,
            "body_vector": self.body_vector.tolist() if isinstance(self.body_vector, np.ndarray) else self.body_vector,
            "parent_to_component_rot": self._parent_to_component_rot.tolist() if self._parent_to_component_rot is not None else None,
        }

    @staticmethod
    def from_dict(d: dict):
        """Deserialize Component from dictionary.

        Args:
            d (dict): Dictionary representation.

        Returns:
            Component: Reconstructed Component instance.
        """
        if d["type"] != "Component":
            raise ValueError("Component.from_dict(): Invalid construction dictionary type")

        # Reconstruct rotation matrix if present
        rot = None
        if "parent_to_component_rot" in d and d["parent_to_component_rot"] is not None:
            rot = np.array(d["parent_to_component_rot"])

        # Create component
        component = Component(
            mass=d.get("mass", 0.0),
            body_vector=np.array(d.get("body_vector", [1, 0, 0])),
            name=d.get("name"),
            parent_to_component_rot=rot
        )

        # Restore ID (override the auto-generated one)
        component.id = d["id"]

        return component
