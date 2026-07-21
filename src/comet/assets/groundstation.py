# python imports
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Union

# COMET imports
from comet.assets import Asset, AssetArray
from comet.state import LLA
from comet.time import Epoch
from comet.time import Duration
from comet.propagation.propagator import Propagator, GroundPropagator
from comet.time import TIMELINE


class Groundstation(Asset):
    """Class that defines a Groundstation Asset.

    Extends Asset with ground station-specific functionality including LLA accessors
    and builder classmethods. Explicitly disables orbit-only methods that don't apply
    to ground-based assets.
    """

    def __init__(self, propagator: Propagator = None, **kwargs):
        """Initialize Groundstation.

        Args:
            propagator (Propagator, optional): Propagator for state evolution.
                Typically a GroundPropagator. Defaults to None.
            **kwargs: Additional keyword arguments passed to Asset.
        """
        # Initialize superclass constructor
        super().__init__(propagator, **kwargs)

    @classmethod
    def create_groundstation(
        cls,
        epoch: Epoch,
        lla: LLA,
        name: str = None,
    ) -> "Groundstation":
        """Create a ground station from LLA coordinates.

        Builder classmethod for creating a ground station with a GroundPropagator
        from latitude, longitude, and altitude.

        Args:
            epoch (Epoch): Reference epoch for the ground station.
            lla (LLA): Latitude, Longitude, Altitude coordinates.
            name (str, optional): Ground station name. Defaults to None.

        Returns:
            Groundstation: Configured ground station instance.
        """
        propagator = GroundPropagator(epoch=epoch, state=lla)
        return cls(propagator=propagator, name=name)

    def get_lla(self) -> np.ndarray:
        """Get ground station LLA (Latitude, Longitude, Altitude) coordinates.

        Mode-aware: returns 2D array (n_time, 3) in BATCH mode with constant values,
        1D array (3,) in STEPPED mode.

        Returns:
            np.ndarray: LLA coordinates [lat, lon, alt]
                - lat: Latitude [rad]
                - lon: Longitude [rad]
                - alt: Altitude [km]

        Raises:
            ValueError: If ground station has no propagator.
        """
        from comet.frames import StateFrame
        from comet.time import TimelineMode

        if self._propagator is None:
            raise ValueError("Groundstation.get_lla(): Cannot get LLA without a propagator")

        # Get LLA state (already mode-aware via Asset.get_state)
        lla = self.get_state(StateFrame.LLA)

        return lla

    # Orbit-only methods - disabled for ground stations
    def get_coe(self) -> np.ndarray:
        """Get classical orbital elements.

        Raises:
            NotImplementedError: Orbital elements don't apply to ground-based assets.
        """
        raise NotImplementedError(
            "Groundstation.get_coe(): Classical orbital elements are not applicable to ground-based assets. "
            "Use get_lla() to get ground station position."
        )

    def get_a(self) -> np.ndarray:
        """Get semi-major axis.

        Raises:
            NotImplementedError: Orbital elements don't apply to ground-based assets.
        """
        raise NotImplementedError(
            "Groundstation.get_a(): Semi-major axis is not applicable to ground-based assets."
        )

    def get_e(self) -> np.ndarray:
        """Get eccentricity.

        Raises:
            NotImplementedError: Orbital elements don't apply to ground-based assets.
        """
        raise NotImplementedError(
            "Groundstation.get_e(): Eccentricity is not applicable to ground-based assets."
        )

    def get_i(self) -> np.ndarray:
        """Get inclination.

        Raises:
            NotImplementedError: Orbital elements don't apply to ground-based assets.
        """
        raise NotImplementedError(
            "Groundstation.get_i(): Inclination is not applicable to ground-based assets."
        )

    def get_raan(self) -> np.ndarray:
        """Get right ascension of ascending node (RAAN).

        Raises:
            NotImplementedError: Orbital elements don't apply to ground-based assets.
        """
        raise NotImplementedError(
            "Groundstation.get_raan(): RAAN is not applicable to ground-based assets."
        )

    def get_omega(self) -> np.ndarray:
        """Get argument of perigee.

        Raises:
            NotImplementedError: Orbital elements don't apply to ground-based assets.
        """
        raise NotImplementedError(
            "Groundstation.get_omega(): Argument of perigee is not applicable to ground-based assets."
        )

    def get_nu(self) -> np.ndarray:
        """Get true anomaly.

        Raises:
            NotImplementedError: Orbital elements don't apply to ground-based assets.
        """
        raise NotImplementedError(
            "Groundstation.get_nu(): True anomaly is not applicable to ground-based assets."
        )

    def to_dict(self):
        """Serialize Groundstation to dictionary.

        Returns:
            dict: Dictionary representation of Groundstation.
        """
        d = super().to_dict()
        d["type"] = "Groundstation"
        return d

    @staticmethod
    def from_dict(d: dict):
        """Deserialize Groundstation from dictionary.

        Args:
            d (dict): Dictionary representation.

        Returns:
            Groundstation: Reconstructed Groundstation instance.
        """
        if d["type"] != "Groundstation":
            raise ValueError("Groundstation.from_dict(): Invalid construction dictionary type")

        # Reconstruct propagator if present
        propagator = None
        if d.get("propagator") is not None:
            from comet.propagation.propagator import Propagator

            propagator = Propagator.from_dict(d["propagator"])

        # Create groundstation
        groundstation = Groundstation(propagator=propagator, name=d.get("name"))

        # Restore ID
        groundstation.id = d["id"]

        # Reconstruct components if present
        if "components" in d and d["components"]:
            from comet.components import Component

            for comp_dict in d["components"]:
                comp = Component.from_dict(comp_dict)
                groundstation.add_component(comp)

        return groundstation


class GroundstationArray(AssetArray):
    """Typed array enforcing Groundstation-only elements.

    Extends AssetArray with Groundstation-specific methods.
    """

    def __init__(self, assets: List):
        """Initialize GroundstationArray.

        Args:
            assets (List): List of Groundstation instances.

        Raises:
            TypeError: If any element is not a Groundstation instance.
        """
        # Import here to avoid circular imports
        from comet.assets import Groundstation

        # Validate all elements are Groundstations
        for i, asset in enumerate(assets):
            if not isinstance(asset, Groundstation):
                raise TypeError(
                    f"GroundstationArray: Element {i} is not a Groundstation instance (got {type(asset).__name__})"
                )

        super().__init__(assets)

    def copy(self) -> GroundstationArray:
        """Create a copy of the array.

        Returns:
            GroundstationArray: New array with same assets.
        """
        return GroundstationArray(self._assets.copy())
