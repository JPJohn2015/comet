# python imports
from __future__ import annotations
import numpy as np
import itertools
from typing import Dict, List, Optional, Union

# COMET imports
from comet.assets import Asset, AssetArray
from comet.utilities.constants import Constants as c
from comet.state import State
from comet.state import Elements
from comet.time import Epoch
from comet.time import Duration
from comet.propagation.propagator import Propagator, SpacePropagator
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.time import TIMELINE
from comet.frames.transformations import eci_to_ecef, ecef_to_lla
from comet.frames import StateFrame


class Satellite(Asset):
    """Class that defines a Satellite Asset.

    Extends Asset with satellite-specific functionality including classical orbital
    element accessors and builder classmethods for common satellite creation patterns.
    """

    def __init__(self, propagator: Propagator = None, **kwargs):
        """Initialize Satellite.

        Args:
            propagator (Propagator, optional): Propagator for state evolution. Defaults to None.
            **kwargs: Additional keyword arguments passed to Asset.
        """
        # Initialize superclass constructor
        super().__init__(propagator, **kwargs)

    @classmethod
    def create_satellite(
        cls,
        epoch: Epoch,
        elements: Elements,
        name: str = None,
        force_model: ForceModel = None,
        sat_properties: SatelliteProperties = None,
    ) -> "Satellite":
        """Create a satellite from orbital elements.

        Builder classmethod for creating a satellite with a SpacePropagator
        from classical orbital elements at a given epoch.

        Args:
            epoch (Epoch): Initial epoch for propagation.
            elements (Elements): Classical orbital elements at epoch.
            name (str, optional): Satellite name. Defaults to None.
            force_model (ForceModel, optional): Force model for propagation. Defaults to None.
            sat_properties (SatelliteProperties, optional): Satellite physical properties. Defaults to None.

        Returns:
            Satellite: Configured satellite instance.
        """
        # Use default ForceModel and SatelliteProperties if not provided
        if force_model is None:
            force_model = ForceModel()
        if sat_properties is None:
            sat_properties = SatelliteProperties()

        propagator = SpacePropagator(
            epoch=epoch,
            state=elements,
            force_model=force_model,
            sat_properties=sat_properties,
        )
        return cls(propagator=propagator, name=name)

    @classmethod
    def create_from_state(
        cls,
        epoch: Epoch,
        state: State,
        name: str = None,
        force_model: ForceModel = None,
        sat_properties: SatelliteProperties = None,
    ) -> "Satellite":
        """Create a satellite from a Cartesian state vector.

        Builder classmethod for creating a satellite with a SpacePropagator
        from a Cartesian state at a given epoch.

        Args:
            epoch (Epoch): Initial epoch for propagation.
            state (State): Cartesian state [position, velocity] at epoch.
            name (str, optional): Satellite name. Defaults to None.
            force_model (ForceModel, optional): Force model for propagation. Defaults to None.
            sat_properties (SatelliteProperties, optional): Satellite physical properties. Defaults to None.

        Returns:
            Satellite: Configured satellite instance.
        """
        # Use default ForceModel and SatelliteProperties if not provided
        if force_model is None:
            force_model = ForceModel()
        if sat_properties is None:
            sat_properties = SatelliteProperties()

        propagator = SpacePropagator(
            epoch=epoch,
            state=state,
            force_model=force_model,
            sat_properties=sat_properties,
        )
        return cls(propagator=propagator, name=name)

    def to_dict(self):
        """Serialize Satellite to dictionary.

        Returns:
            dict: Dictionary representation of Satellite.
        """
        d = super().to_dict()
        d["type"] = "Satellite"
        return d

    @staticmethod
    def from_dict(d: dict):
        """Deserialize Satellite from dictionary.

        Args:
            d (dict): Dictionary representation.

        Returns:
            Satellite: Reconstructed Satellite instance.
        """
        if d["type"] != "Satellite":
            raise ValueError("Satellite.from_dict(): Invalid construction dictionary type")

        # Reconstruct propagator if present
        propagator = None
        if d.get("propagator") is not None:
            from comet.propagation.propagator import Propagator

            propagator = Propagator.from_dict(d["propagator"])

        # Create satellite
        satellite = Satellite(propagator=propagator, name=d.get("name"))

        # Restore ID
        satellite.id = d["id"]

        # Reconstruct components if present
        if "components" in d and d["components"]:
            from comet.components import Component

            for comp_dict in d["components"]:
                comp = Component.from_dict(comp_dict)
                satellite.add_component(comp)

        return satellite

    def get_coe(self) -> np.ndarray:
        """Get classical orbital elements (COE).

        Mode-aware: returns 2D array (n_time, 6) in BATCH mode,
        1D array (6,) in STEPPED mode.

        Returns:
            np.ndarray: Orbital elements [a, e, i, Ω, ω, ν]
                - a: Semi-major axis [km]
                - e: Eccentricity [-]
                - i: Inclination [rad]
                - Ω: Right ascension of ascending node [rad]
                - ω: Argument of perigee [rad]
                - ν: True anomaly [rad]

        Raises:
            ValueError: If satellite has no propagator or propagator is not orbit-based.
        """
        from comet.state.state_conversions import cartesian_to_elements

        if self._propagator is None:
            raise ValueError("Satellite.get_coe(): Cannot compute COE without a propagator")

        # Get ECI state (already mode-aware)
        eci_state = self.get_state(StateFrame.ECI)

        # Convert to COE
        coe = cartesian_to_elements(eci_state, mu=c.MU_EARTH)

        return coe

    def get_a(self) -> np.ndarray:
        """Get semi-major axis.

        Mode-aware: returns 1D array (n_time,) in BATCH mode,
        scalar in STEPPED mode.

        Returns:
            np.ndarray: Semi-major axis [km]
        """
        coe = self.get_coe()
        return coe[..., 0]

    def get_e(self) -> np.ndarray:
        """Get eccentricity.

        Mode-aware: returns 1D array (n_time,) in BATCH mode,
        scalar in STEPPED mode.

        Returns:
            np.ndarray: Eccentricity [-]
        """
        coe = self.get_coe()
        return coe[..., 1]

    def get_i(self) -> np.ndarray:
        """Get inclination.

        Mode-aware: returns 1D array (n_time,) in BATCH mode,
        scalar in STEPPED mode.

        Returns:
            np.ndarray: Inclination [rad]
        """
        coe = self.get_coe()
        return coe[..., 2]

    def get_raan(self) -> np.ndarray:
        """Get right ascension of ascending node (RAAN).

        Mode-aware: returns 1D array (n_time,) in BATCH mode,
        scalar in STEPPED mode.

        Returns:
            np.ndarray: RAAN [rad]
        """
        coe = self.get_coe()
        return coe[..., 3]

    def get_omega(self) -> np.ndarray:
        """Get argument of perigee.

        Mode-aware: returns 1D array (n_time,) in BATCH mode,
        scalar in STEPPED mode.

        Returns:
            np.ndarray: Argument of perigee [rad]
        """
        coe = self.get_coe()
        return coe[..., 4]

    def get_nu(self) -> np.ndarray:
        """Get true anomaly.

        Mode-aware: returns 1D array (n_time,) in BATCH mode,
        scalar in STEPPED mode.

        Returns:
            np.ndarray: True anomaly [rad]
        """
        coe = self.get_coe()
        return coe[..., 5]


class SatelliteArray(AssetArray):
    """Typed array enforcing Satellite-only elements.

    Extends AssetArray with Satellite-specific methods.
    """

    def __init__(self, assets: List):
        """Initialize SatelliteArray.

        Args:
            assets (List): List of Satellite instances.

        Raises:
            TypeError: If any element is not a Satellite instance.
        """
        # Import here to avoid circular imports
        from comet.assets import Satellite

        # Validate all elements are Satellites
        for i, asset in enumerate(assets):
            if not isinstance(asset, Satellite):
                raise TypeError(
                    f"SatelliteArray: Element {i} is not a Satellite instance (got {type(asset).__name__})"
                )

        super().__init__(assets)

    def copy(self) -> SatelliteArray:
        """Create a copy of the array.

        Returns:
            SatelliteArray: New array with same assets.
        """
        return SatelliteArray(self._assets.copy())
