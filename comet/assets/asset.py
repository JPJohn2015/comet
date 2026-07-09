# python imports
import numpy as np
import itertools

# COMET imports
from comet.utilities.constants import Constants as c
from comet.state.state import State
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.propagation.propagator import Propagator
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.time.timeline import TIMELINE
from comet.frames.transformations import eci_to_ecef, ecef_to_lla
from comet.frames.frame import StateFrame

# --------------------------------------------------------------------------------------------------------------------------
class Asset:
    """Base Asset Class.
    """
    # Asset ID Counter
    id_counter = itertools.count()

    # ----------------------------------------------------------------------------------------------------------------------
    # Class Construction
    # ----------------------------------------------------------------------------------------------------------------------
    def __init__(self, 
                 propagator: Propagator = None,
                 **kwargs):
        
        # Assign TIMELINE
        self.timeline = TIMELINE

        # Assign Asset ID
        self.id = next(Asset.id_counter)

        # Assign Propagator
        self._propagator = propagator

        # Initialize state data attributes
        self._eci_state = None
        self._ecef_state = None

    # ----------------------------------------------------------------------------------------------------------------------
    def get_state(self, frame: str|StateFrame = StateFrame.ECI):
        """Method that gets the full state data in the specified frame.

        Args:
            frame (str|StateFrame, optional): StateFrame to return state data. Defaults to StateFrame.ECI.

        Returns:
            state_data (np.ndarray): Nx6 Array of state data.
        """
        # Determine which frame to return state data 
        match frame:
            case StateFrame.ECI | 'ECI' | 'eci':
                # ECI is requested
                if self._eci_state is None:
                    self._sample_eci_state()
                return np.copy(self._eci_state)
            case StateFrame.ECEF | 'ECEF' | 'ecef':
                # ECEF is requested
                if self._ecef_state is None:
                    self._sample_ecef_state()
                return np.copy(self._ecef_state)
            case StateFrame.LLA | 'LLA' | 'lla':
                # LLA is requested
                if self._ecef_state is None:
                    self._sample_ecef_state()
                return ecef_to_lla(self._ecef_state)
            case _:
                raise ValueError('Asset(): Invalid StateFrame')
            
    # ----------------------------------------------------------------------------------------------------------------------
    def get_position(self, frame: str|StateFrame = StateFrame.ECI):
        """Method that gets the position state data in the specified frame.

        Args:
            frame (str|StateFrame, optional): StateFrame to return position. Defaults to StateFrame.ECI.

        Returns:
            position (np.ndarray): Nx3 Array of position in km.
        """
        state_data = self.get_state(frame)

        return state_data[...,0:3]
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_velocity(self, frame: str|StateFrame = StateFrame.ECI):
        """Method that gets the velocity state data in the specified frame.

        Args:
            frame (str|StateFrame, optional): StateFrame to return velocity. Defaults to StateFrame.ECI.

        Returns:
            velocity (np.ndarray): Nx3 Array of Velocity in km/s.
        """
        state_data = self.get_state(frame)

        return state_data[...,3:6]
            
    # ----------------------------------------------------------------------------------------------------------------------
    def _sample_eci_state(self):
        """Method samples the ECI state from the given propagator over the global TIMELINE.
        """
        # If the Asset has a propagator, propagate timeline
        if self._propagator is not None:
            self._eci_state = self._propagator.project_to_epoch(self.timeline.stop, self.timeline.step)

    # ----------------------------------------------------------------------------------------------------------------------
    def _sample_ecef_state(self):
        """Method samples the ECEF state from the given propagator over the global TIMELINE.
        If the ECI state is not sampled, it will also sample that state data.
        """
        # If the Asset has a propagator, propagate timeline
        if self._propagator is not None:
            # Check if eci state has been propagated
            if self._eci_state is None:
                self._eci_state = self._propagator.project_to_epoch(self.timeline.stop, self.timeline.step)
        
        # Calculate ecef states
        self._ecef_state = eci_to_ecef(self.timeline.epochs, self._eci_state)

    # ----------------------------------------------------------------------------------------------------------------------
    def _sample_state(self):
        """Method samples both the ECI and ECEF state from the given propagator over the global TIMELINE.
        """
        # Samples all states
        self._sample_eci_state()
        self._sample_ecef_state()

# Testing
if __name__ == "__main__":
    # Parse Timeline
    TIMELINE.update(Epoch(2024,1,1,0,0,0), Epoch(2024,1,1,1,0,0), Duration(seconds=60))

    # Create Satellite Properties for Propagator
    start_elements = Elements(6378 + 780, 0, np.pi/4, 0, 0, 0)

    propagator = Propagator(epoch=TIMELINE.start, state=start_elements)
    asset = Asset(propagator=propagator)

    asset._sample_state()
