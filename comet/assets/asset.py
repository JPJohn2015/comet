# python imports
from __future__ import annotations
import numpy as np
import itertools
from typing import Dict, List, Optional, Union

# COMET imports
from comet.utilities.constants import Constants as c
from comet.state.state import State
from comet.state.elements import Elements
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.propagation.propagator import (
    Propagator,
    PropagatorCategory,
    SpacePropagator,
    GroundPropagator,
    SunPropagator,
    MoonPropagator,
    TLEPropagator,
)
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.time.timeline import TIMELINE, TimelineMode
from comet.frames.transformations import eci_to_ecef, ecef_to_lla
from comet.frames.frame import StateFrame


class Asset:
    """Base Asset Class.

    Provides mode-aware state propagation, component management, and cache invalidation.
    Assets can operate in BATCH mode (returning 2D arrays over timeline) or STEPPED mode
    (returning 1D vectors at current timeline cursor position).
    """

    # Asset ID Counter
    id_counter = itertools.count()

    def __init__(self, propagator: Propagator = None, name: str = None, **kwargs):
        """Initialize Asset.

        Args:
            propagator (Propagator, optional): Propagator for state evolution. Defaults to None.
            name (str, optional): Asset name. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # Assign TIMELINE
        self.timeline = TIMELINE

        # Assign Asset ID and name
        self.id = next(Asset.id_counter)
        self.name = name if name is not None else f"Asset_{self.id}"

        # Initialize propagator via property setter
        self._propagator = None
        self._propagator_category = None
        if propagator is not None:
            self.set_propagator(propagator)

        # Initialize state data attributes
        self._eci_state = None
        self._ecef_state = None

        # Initialize timeline hash for cache invalidation
        self._valid_timeline_hash = None

        # Initialize component management
        self._components: List = []
        self._component_name_to_id: Dict[str, int] = {}

    def set_propagator(self, propagator: Propagator):
        """Set the propagator and update category.

        Args:
            propagator (Propagator): New propagator.
        """
        if propagator is not None and not isinstance(propagator, Propagator):
            raise TypeError("Propagator must be an instance of Propagator or its subclasses")

        self._propagator = propagator

        # Map propagator to category
        if isinstance(propagator, (SpacePropagator, TLEPropagator)):
            self._propagator_category = PropagatorCategory.SPACE
        elif isinstance(propagator, GroundPropagator):
            self._propagator_category = PropagatorCategory.GROUND
        elif isinstance(propagator, (SunPropagator, MoonPropagator)):
            self._propagator_category = PropagatorCategory.CELESTIAL
        elif isinstance(propagator, Propagator):
            self._propagator_category = PropagatorCategory.GENERIC
        else:
            self._propagator_category = PropagatorCategory.GENERIC

        # Clear cached states
        self._clear_state_cache()

    def get_propagator(self) -> Propagator:
        """Get the current propagator.

        Returns:
            Propagator: Current propagator.
        """
        return self._propagator

    def get_propagator_category(self) -> PropagatorCategory | None:
        """Get the propagator category.

        Returns:
            PropagatorCategory | None: Category of the current propagator, or None if no propagator set.
        """
        return self._propagator_category

    def _clear_state_cache(self):
        """Clear all cached state data."""
        self._eci_state = None
        self._ecef_state = None
        self._valid_timeline_hash = None

    def _check_timeline_or_resample_state(self):
        """Check if timeline has changed and invalidate cache if needed."""
        current_hash = self.timeline.get_hash()
        if self._valid_timeline_hash != current_hash:
            self._clear_state_cache()
            self._valid_timeline_hash = current_hash

    def get_state(self, frame: str | StateFrame = StateFrame.ECI):
        """Get the full state data in the specified frame.

        Mode-aware: returns 2D array (n_time, 6) in BATCH mode,
        1D array (6,) in STEPPED mode.

        Args:
            frame (str|StateFrame, optional): StateFrame to return state data. Defaults to StateFrame.ECI.

        Returns:
            state_data (np.ndarray): State data in km and km/s.
                BATCH mode: (n_time, 6) array
                STEPPED mode: (6,) array
        """
        # Check timeline and resample if needed
        self._check_timeline_or_resample_state()

        # Determine which frame to return state data
        match frame:
            case StateFrame.ECI | "ECI" | "eci":
                # ECI is requested
                if self._eci_state is None:
                    self._sample_eci_state()
                return np.copy(self._eci_state)
            case StateFrame.ECEF | "ECEF" | "ecef":
                # ECEF is requested
                if self._ecef_state is None:
                    self._sample_ecef_state()
                return np.copy(self._ecef_state)
            case StateFrame.LLA | "LLA" | "lla":
                # LLA is requested
                if self._ecef_state is None:
                    self._sample_ecef_state()
                return ecef_to_lla(self._ecef_state)
            case _:
                raise ValueError("Asset.get_state(): Invalid StateFrame")

    def get_position(self, frame: str | StateFrame = StateFrame.ECI):
        """Method that gets the position state data in the specified frame.

        Args:
            frame (str|StateFrame, optional): StateFrame to return position. Defaults to StateFrame.ECI.

        Returns:
            position (np.ndarray): Nx3 Array of position in km.
        """
        state_data = self.get_state(frame)

        return state_data[..., 0:3]

    def get_velocity(self, frame: str | StateFrame = StateFrame.ECI):
        """Method that gets the velocity state data in the specified frame.

        Args:
            frame (str|StateFrame, optional): StateFrame to return velocity. Defaults to StateFrame.ECI.

        Returns:
            velocity (np.ndarray): Nx3 Array of Velocity in km/s.
        """
        state_data = self.get_state(frame)

        return state_data[..., 3:6]

    def _sample_eci_state(self):
        """Sample the ECI state from propagator.

        Mode-aware:
            BATCH: Projects across full timeline, returns (n_time, 6) array
            STEPPED: Evaluates at timeline.now(), returns (6,) array
        """
        if self._propagator is None:
            return

        mode = self.timeline.get_mode()

        if mode == TimelineMode.BATCH:
            # BATCH mode: propagate over full timeline
            self._eci_state = self._propagator.project_to_epoch(
                self.timeline.stop, self.timeline.step
            )
        else:
            # STEPPED mode: project to current epoch only
            current_epoch = self.timeline.now()
            duration = current_epoch - self._propagator.epoch
            self._eci_state = self._propagator.project_to_duration(duration, step=None)

    def _sample_ecef_state(self):
        """Sample the ECEF state from ECI state.

        Mode-aware:
            BATCH: Transforms full timeline ECI states, returns (n_time, 6) array
            STEPPED: Transforms current ECI state, returns (6,) array

        If ECI state is not sampled, it will also sample that state data.
        """
        # Ensure ECI state is sampled
        if self._eci_state is None:
            self._sample_eci_state()

        if self._eci_state is None:
            return

        mode = self.timeline.get_mode()

        if mode == TimelineMode.BATCH:
            # BATCH mode: transform all epochs
            epochs = self.timeline.get_epoch_list()
            self._ecef_state = eci_to_ecef(epochs, self._eci_state)
        else:
            # STEPPED mode: transform single epoch
            current_epoch = self.timeline.now()
            self._ecef_state = eci_to_ecef(current_epoch, self._eci_state)

    def _sample_state(self):
        """Sample both the ECI and ECEF state from the propagator."""
        # Samples all states
        self._sample_eci_state()
        self._sample_ecef_state()

    # Component Management
    def add_component(self, component):
        """Add a component to this asset.

        Args:
            component: Component to add. Must have 'id' and 'name' attributes.
        """
        if not hasattr(component, "id"):
            raise ValueError("Component must have an 'id' attribute")

        # Set parent reference on component
        if hasattr(component, "parent"):
            component.parent = self

        # Add to components list
        self._components.append(component)

        # Add to name mapping if component has a name
        if hasattr(component, "name") and component.name is not None:
            self._component_name_to_id[component.name] = component.id

    def remove_component(self, component_id: int):
        """Remove a component by ID.

        Args:
            component_id (int): ID of component to remove.

        Returns:
            bool: True if component was removed, False if not found.
        """
        for i, comp in enumerate(self._components):
            if comp.id == component_id:
                # Remove from name mapping
                if hasattr(comp, "name") and comp.name in self._component_name_to_id:
                    del self._component_name_to_id[comp.name]

                # Remove parent reference
                if hasattr(comp, "parent"):
                    comp.parent = None

                # Remove from list
                self._components.pop(i)
                return True
        return False

    def get_component(self, component_id: int = None, component_name: str = None):
        """Get a component by ID or name.

        Args:
            component_id (int, optional): Component ID. Defaults to None.
            component_name (str, optional): Component name. Defaults to None.

        Returns:
            Component or None: The component if found, None otherwise.
        """
        if component_id is not None:
            for comp in self._components:
                if comp.id == component_id:
                    return comp
        elif component_name is not None:
            comp_id = self._component_name_to_id.get(component_name)
            if comp_id is not None:
                return self.get_component(component_id=comp_id)
        return None

    def get_components(self) -> List:
        """Get all components.

        Returns:
            List: List of all components.
        """
        return self._components.copy()

    def clear_components(self):
        """Remove all components."""
        for comp in self._components:
            if hasattr(comp, "parent"):
                comp.parent = None
        self._components.clear()
        self._component_name_to_id.clear()

    # Access and Range Stubs
    def get_access(self, *args, **kwargs):
        """Get access to/from this asset.

        Raises:
            NotImplementedError: Will be implemented in Phase 6 with comet.access.
        """
        raise NotImplementedError(
            "Asset.get_access() will be implemented in Phase 6 when comet.access module exists"
        )

    def get_range_to(self, *args, **kwargs):
        """Get range to another asset.

        Raises:
            NotImplementedError: Will be implemented in Phase 6 with comet.access.
        """
        raise NotImplementedError(
            "Asset.get_range_to() will be implemented in Phase 6 when comet.access module exists"
        )

    # Serialization
    def to_dict(self):
        """Serialize Asset to dictionary.

        Returns:
            dict: Dictionary representation of Asset.
        """
        return {
            "type": "Asset",
            "id": self.id,
            "name": self.name,
            "propagator": self._propagator.to_dict() if self._propagator is not None else None,
            "propagator_category": self._propagator_category.value
            if self._propagator_category is not None
            else None,
            "components": [comp.to_dict() for comp in self._components if hasattr(comp, "to_dict")],
        }

    @staticmethod
    def from_dict(d: dict):
        """Deserialize Asset from dictionary.

        Args:
            d (dict): Dictionary representation.

        Returns:
            Asset: Reconstructed Asset instance.
        """
        if d["type"] != "Asset":
            raise ValueError("Asset.from_dict(): Invalid construction dictionary type")

        # Reconstruct propagator if present
        propagator = None
        if d.get("propagator") is not None:
            # Import required for reconstruction
            from comet.propagation.propagator import Propagator

            propagator = Propagator.from_dict(d["propagator"])

        # Create asset
        asset = Asset(propagator=propagator, name=d.get("name"))

        # Restore ID (override the auto-generated one)
        asset.id = d["id"]

        # Reconstruct components if present
        if "components" in d and d["components"]:
            from comet.components.component import Component

            for comp_dict in d["components"]:
                comp = Component.from_dict(comp_dict)
                asset.add_component(comp)

        return asset


class AssetArray:
    """Base array class for holding multiple Assets.

    AssetArray is permissive - it holds any mix of Asset subclasses and guarantees
    only the base Asset surface. Mode-aware stacked getters automatically return
    correct shapes based on timeline mode.
    """

    def __init__(self, assets: List[Asset]):
        """Initialize AssetArray.

        Args:
            assets (List[Asset]): List of Asset instances.

        Raises:
            TypeError: If any element is not an Asset instance.
        """
        # Validate all elements are Assets
        for i, asset in enumerate(assets):
            if not isinstance(asset, Asset):
                raise TypeError(
                    f"AssetArray: Element {i} is not an Asset instance (got {type(asset).__name__})"
                )

        self._assets = list(assets)
        self.timeline = TIMELINE

    def __len__(self) -> int:
        """Get number of assets in array.

        Returns:
            int: Number of assets.
        """
        return len(self._assets)

    def __getitem__(self, index: int) -> Asset:
        """Get asset by index.

        Args:
            index (int): Index of asset.

        Returns:
            Asset: Asset at index.
        """
        return self._assets[index]

    def __iter__(self):
        """Iterate over assets.

        Yields:
            Asset: Each asset in the array.
        """
        return iter(self._assets)

    # Mode-aware stacked getters
    def get_state(self, frame: str | StateFrame = StateFrame.ECI) -> np.ndarray:
        """Get stacked states for all assets.

        Mode-aware stacking:
            BATCH mode: (n_asset, n_time, 6) array
            STEPPED mode: (n_asset, 6) array

        Args:
            frame (str|StateFrame, optional): StateFrame to return. Defaults to StateFrame.ECI.

        Returns:
            np.ndarray: Stacked state data.
        """
        if len(self._assets) == 0:
            return np.array([])

        states = [asset.get_state(frame) for asset in self._assets]
        return np.stack(states, axis=0)

    def get_position(self, frame: str | StateFrame = StateFrame.ECI) -> np.ndarray:
        """Get stacked positions for all assets.

        Mode-aware stacking:
            BATCH mode: (n_asset, n_time, 3) array
            STEPPED mode: (n_asset, 3) array

        Args:
            frame (str|StateFrame, optional): StateFrame to return. Defaults to StateFrame.ECI.

        Returns:
            np.ndarray: Stacked position data.
        """
        if len(self._assets) == 0:
            return np.array([])

        positions = [asset.get_position(frame) for asset in self._assets]
        return np.stack(positions, axis=0)

    def get_velocity(self, frame: str | StateFrame = StateFrame.ECI) -> np.ndarray:
        """Get stacked velocities for all assets.

        Mode-aware stacking:
            BATCH mode: (n_asset, n_time, 3) array
            STEPPED mode: (n_asset, 3) array

        Args:
            frame (str|StateFrame, optional): StateFrame to return. Defaults to StateFrame.ECI.

        Returns:
            np.ndarray: Stacked velocity data.
        """
        if len(self._assets) == 0:
            return np.array([])

        velocities = [asset.get_velocity(frame) for asset in self._assets]
        return np.stack(velocities, axis=0)

    # Asset retrieval methods
    def get_asset_by_id(self, asset_id: int) -> Optional[Asset]:
        """Get asset by ID.

        Args:
            asset_id (int): Asset ID to search for.

        Returns:
            Optional[Asset]: Asset with matching ID, or None if not found.
        """
        for asset in self._assets:
            if asset.id == asset_id:
                return asset
        return None

    def get_asset_by_name(self, name: str) -> Optional[Asset]:
        """Get asset by name.

        Args:
            name (str): Asset name to search for.

        Returns:
            Optional[Asset]: Asset with matching name, or None if not found.
        """
        for asset in self._assets:
            if asset.name == name:
                return asset
        return None

    def get_asset_by_index(self, index: int) -> Asset:
        """Get asset by array index.

        Args:
            index (int): Array index.

        Returns:
            Asset: Asset at index.
        """
        return self._assets[index]

    def get_assets_by_category(self, category: PropagatorCategory) -> AssetArray:
        """Filter assets by propagator category.

        Args:
            category (PropagatorCategory): Propagator category to filter by.

        Returns:
            AssetArray: New array containing only assets with matching category.
        """
        filtered = [asset for asset in self._assets if asset.get_propagator_category() == category]
        return AssetArray(filtered)

    # Type checking and filtering
    def is_homogeneous(self, cls_type: Optional[type] = None) -> bool:
        """Check if array is homogeneous.

        Args:
            cls_type (type, optional): If provided, checks if all elements are this type (or subclass).
                                       If None, checks if all elements share one concrete type.

        Returns:
            bool: True if array is homogeneous, False otherwise.
        """
        if len(self._assets) == 0:
            return True

        if cls_type is not None:
            # Check if all elements are instances of cls_type
            return all(isinstance(asset, cls_type) for asset in self._assets)
        else:
            # Check if all elements share one concrete type
            first_type = type(self._assets[0])
            return all(type(asset) is first_type for asset in self._assets)

    def filter_by_type(self, cls_type: type) -> AssetArray:
        """Filter assets by type.

        Returns the narrowest possible array type:
        - All Satellite -> SatelliteArray
        - All Groundstation -> GroundstationArray
        - Otherwise -> AssetArray

        Args:
            cls_type (type): Type to filter by.

        Returns:
            AssetArray: New array containing only assets of specified type.
        """
        filtered = [asset for asset in self._assets if isinstance(asset, cls_type)]
        return _narrowest_common_array(filtered)

    # Concatenation
    def __add__(self, other: Union[AssetArray, Asset]) -> AssetArray:
        """Concatenate arrays or add single asset.

        Narrowing rule:
        - Same typed array + same typed array = same typed array
        - Mixed types = AssetArray

        Args:
            other (AssetArray | Asset): Array or asset to concatenate.

        Returns:
            AssetArray: Concatenated array with narrowest appropriate type.
        """
        if isinstance(other, Asset):
            # Asset + AssetArray
            combined = self._assets + [other]
        elif isinstance(other, AssetArray):
            # AssetArray + AssetArray
            combined = self._assets + other._assets
        else:
            raise TypeError(f"Cannot concatenate AssetArray with {type(other).__name__}")

        return _narrowest_common_array(combined)

    def __radd__(self, other: Union[AssetArray, Asset]) -> AssetArray:
        """Right-side concatenation.

        Args:
            other (AssetArray | Asset): Array or asset to concatenate.

        Returns:
            AssetArray: Concatenated array with narrowest appropriate type.
        """
        if isinstance(other, Asset):
            # Asset + AssetArray
            combined = [other] + self._assets
        elif isinstance(other, AssetArray):
            # AssetArray + AssetArray
            combined = other._assets + self._assets
        else:
            raise TypeError(f"Cannot concatenate {type(other).__name__} with AssetArray")

        return _narrowest_common_array(combined)

    def copy(self) -> AssetArray:
        """Create a copy of the array.

        Returns:
            AssetArray: New array with same assets (shallow copy of asset list).
        """
        return AssetArray(self._assets.copy())

    # Access and range stubs
    def get_access(self, *args, **kwargs):
        """Get access for all assets.

        Raises:
            NotImplementedError: Will be implemented in Phase 6 with comet.access.
        """
        raise NotImplementedError(
            "AssetArray.get_access() will be implemented in Phase 6 when comet.access module exists"
        )

    def get_range_to(self, *args, **kwargs):
        """Get range to another asset/array.

        Raises:
            NotImplementedError: Will be implemented in Phase 6 with comet.access.
        """
        raise NotImplementedError(
            "AssetArray.get_range_to() will be implemented in Phase 6 when comet.access module exists"
        )

    # Serialization
    def to_dict(self) -> dict:
        """Serialize AssetArray to dictionary.

        Returns:
            dict: Dictionary representation.
        """
        return {
            "type": self.__class__.__name__,
            "assets": [asset.to_dict() for asset in self._assets],
        }

    @staticmethod
    def from_dict(d: dict) -> AssetArray:
        """Deserialize AssetArray from dictionary.

        Args:
            d (dict): Dictionary representation.

        Returns:
            AssetArray: Reconstructed array.
        """
        # Import here to avoid circular imports
        from comet.assets.satellite import Satellite
        from comet.assets.groundstation import Groundstation

        # Reconstruct assets
        assets = []
        for asset_dict in d["assets"]:
            asset_type = asset_dict.get("type", "Asset")
            if asset_type == "Satellite":
                assets.append(Satellite.from_dict(asset_dict))
            elif asset_type == "Groundstation":
                assets.append(Groundstation.from_dict(asset_dict))
            else:
                assets.append(Asset.from_dict(asset_dict))

        # Construct appropriate array type
        array_type = d.get("type", "AssetArray")
        if array_type == "SatelliteArray":
            from comet.assets.satellite import SatelliteArray

            return SatelliteArray(assets)
        elif array_type == "GroundstationArray":
            from comet.assets.groundstation import GroundstationArray

            return GroundstationArray(assets)
        else:
            return AssetArray(assets)


def _narrowest_common_array(assets: List[Asset]) -> AssetArray:
    """Create the narrowest appropriate array type for the given assets.

    Args:
        assets (List[Asset]): List of assets.

    Returns:
        AssetArray: SatelliteArray, GroundstationArray, or AssetArray depending on element types.
    """
    if len(assets) == 0:
        return AssetArray([])

    # Import here to avoid circular imports
    from comet.assets.satellite import Satellite
    from comet.assets.groundstation import Groundstation

    # Check if all are Satellites
    if all(isinstance(asset, Satellite) for asset in assets):
        from comet.assets.satellite import SatelliteArray

        return SatelliteArray(assets)

    # Check if all are Groundstations
    if all(isinstance(asset, Groundstation) for asset in assets):
        from comet.assets.groundstation import GroundstationArray

        return GroundstationArray(assets)

    # Mixed or base Assets
    return AssetArray(assets)
