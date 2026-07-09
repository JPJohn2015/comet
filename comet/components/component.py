# python imports
import numpy as np
import itertools

# COMET imports
from comet.time.epoch import Epoch


class Component:
    """Base Class that defines any component that can be attached to a Satellite."""

    # Component ID Counter
    id_counter = itertools.count()

    def __init__(self, mass: float = 0.0, body_vector: np.ndarray = [1, 0, 0]):
        """Construct Base Component.

        Args:
            mass (float, optional): Component mass in kg. Defaults to 0.0.
            body_vector (np.ndarray, optional): Pointing Vector in Body Frame. Defaults to [1, 0, 0].
        """
        # Assign Class attributes
        self.mass = mass
        self.body_vector = body_vector

        # Assign Component ID
        self.id = next(Component.id_counter)

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
