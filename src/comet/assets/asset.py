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
    def get_access(
        self,
        targets: Union['Asset', List['Asset']],
        constraints: Optional[Union['AccessConstraint', List['AccessConstraint']]] = None
    ) -> np.ndarray:
        """Get access to target asset(s) subject to constraints.

        Evaluates composite access from this asset to target(s) by ANDing all
        provided constraints. Mode-aware based on timeline state.

        Args:
            targets (Asset | List[Asset]): Single target or list of targets
            constraints (AccessConstraint | List[AccessConstraint] | None): Constraints
                to evaluate. If None, returns all-pass (all 1.0).

        Returns:
            np.ndarray: Composite access mask (0.0 or 1.0). Shape depends on mode:
                - BATCH, multiple: (n_targets, n_times)
                - BATCH, single: (n_times,)
                - STEPPED, multiple: (n_targets,)
                - STEPPED, single: scalar float
        """
        from comet.access.core import evaluate_composite_access, EvalMode

        # Normalize targets to list
        if not isinstance(targets, list):
            targets = [targets]
            single_target = True
        else:
            single_target = False

        # Get mode and convert to EvalMode
        mode = self.timeline.get_mode()
        if mode == TimelineMode.BATCH:
            eval_mode = EvalMode.BATCH
        else:
            # STEPPED mode maps to CURRENT
            eval_mode = EvalMode.CURRENT

        # Handle no constraints case - return all-pass
        if constraints is None:
            if mode == TimelineMode.BATCH:
                n_times = len(self.timeline.get_epoch_list())
                n_targets = len(targets)
                access = np.ones((1, n_targets, n_times))
            else:
                n_targets = len(targets)
                access = np.ones((1, n_targets, 1))
        else:
            # Normalize constraints to list
            if not isinstance(constraints, list):
                constraints = [constraints]

            # Build constraints_per_source dict (self is source 0)
            constraints_per_source = {0: constraints}

            # Evaluate composite access
            access = evaluate_composite_access(
                sources=[self],
                targets=targets,
                constraints_per_source=constraints_per_source,
                mode=eval_mode,
                timeline=self.timeline,
            )

        # Remove source dimension (always 1 source)
        access = access[0, :, :]  # (K, T)

        # Handle single target
        if single_target:
            access = access[0, :]  # (T,) or (1,)

        # Handle STEPPED mode - squeeze time dimension
        if mode != TimelineMode.BATCH:
            access = access.squeeze()

        return access

    def get_access_windows(
        self,
        access_mask: np.ndarray,
        target_idx: int = 0
    ) -> List[Tuple[Epoch, Epoch]]:
        """Extract continuous access windows from an access mask.

        Identifies contiguous blocks of access and returns them as
        (start_epoch, end_epoch) tuples.

        Args:
            access_mask (np.ndarray): Access mask from get_access(). Shape (T,) for
                single target or (K, T) for multiple targets.
            target_idx (int): Target index to extract if multiple targets. Defaults to 0.

        Returns:
            List[Tuple[Epoch, Epoch]]: List of (start, end) epoch pairs for each window

        Raises:
            ValueError: If not in BATCH mode (requires timeline with multiple epochs)
        """
        from comet.access.core import extract_access_windows

        # Check mode
        if self.timeline.get_mode() != TimelineMode.BATCH:
            raise ValueError(
                "get_access_windows() requires BATCH mode. "
                "Use timeline.set_mode(TimelineMode.BATCH) first."
            )

        # Ensure access_mask has shape (1, K, T) or (1, T) for extraction function
        if access_mask.ndim == 1:
            # Single target: (T,) -> (1, 1, T)
            access_mask = access_mask[np.newaxis, np.newaxis, :]
        elif access_mask.ndim == 2:
            # Multiple targets: (K, T) -> (1, K, T)
            access_mask = access_mask[np.newaxis, :, :]
        else:
            raise ValueError(f"Unexpected access_mask shape: {access_mask.shape}")

        return extract_access_windows(access_mask, self.timeline, source_idx=0, target_idx=target_idx)

    def get_access_windows_indices(
        self,
        access_mask: np.ndarray,
        target_idx: int = 0
    ) -> List[Tuple[int, int]]:
        """Extract continuous access windows as timeline index ranges.

        Similar to get_access_windows but returns integer indices instead of epochs.

        Args:
            access_mask (np.ndarray): Access mask from get_access(). Shape (T,) for
                single target or (K, T) for multiple targets.
            target_idx (int): Target index to extract if multiple targets. Defaults to 0.

        Returns:
            List[Tuple[int, int]]: List of (start_idx, end_idx) pairs for each window.
                Indices are inclusive: [start_idx, end_idx].

        Raises:
            ValueError: If not in BATCH mode (requires timeline with multiple epochs)
        """
        from comet.access.core import extract_access_windows_indices

        # Check mode
        if self.timeline.get_mode() != TimelineMode.BATCH:
            raise ValueError(
                "get_access_windows_indices() requires BATCH mode. "
                "Use timeline.set_mode(TimelineMode.BATCH) first."
            )

        # Ensure access_mask has shape (1, K, T) or (1, T) for extraction function
        if access_mask.ndim == 1:
            # Single target: (T,) -> (1, 1, T)
            access_mask = access_mask[np.newaxis, np.newaxis, :]
        elif access_mask.ndim == 2:
            # Multiple targets: (K, T) -> (1, K, T)
            access_mask = access_mask[np.newaxis, :, :]
        else:
            raise ValueError(f"Unexpected access_mask shape: {access_mask.shape}")

        return extract_access_windows_indices(access_mask, source_idx=0, target_idx=target_idx)

    def get_range_to(self, targets: Union['Asset', List['Asset']]) -> np.ndarray:
        """Get range to target asset(s).

        Mode-aware computation using timeline mode:
        - BATCH mode: Returns (n_targets, n_times) array
        - STEPPED/CURRENT mode: Returns (n_targets,) array
        - Single target: First dimension is squeezed out

        Args:
            targets (Asset | List[Asset]): Single target Asset or list of target Assets

        Returns:
            np.ndarray: Range in km. Shape depends on mode and number of targets:
                - BATCH, multiple: (n_targets, n_times)
                - BATCH, single: (n_times,)
                - STEPPED, multiple: (n_targets,)
                - STEPPED, single: scalar float
        """
        from comet.access.functions import compute_range_km

        # Normalize to list
        if not isinstance(targets, list):
            targets = [targets]
            single_target = True
        else:
            single_target = False

        # Get source position (self)
        source_pos = self.get_position(StateFrame.ECI)  # (T, 3) in BATCH or (3,) in STEPPED

        # Get mode
        mode = self.timeline.get_mode()

        # Collect target positions
        target_positions = []
        for target in targets:
            target_pos = target.get_position(StateFrame.ECI)
            target_positions.append(target_pos)

        if mode == TimelineMode.BATCH:
            # BATCH mode: positions are (T, 3)
            # Reshape for compute_range_km: needs (N, T, 3) and (K, T, 3)
            source_pos = source_pos[np.newaxis, :, :]  # (1, T, 3)

            # Stack targets along first axis: (K, T, 3)
            target_pos_stacked = np.stack(target_positions, axis=0)

            # Compute range: returns (1, K, T)
            ranges = compute_range_km(source_pos, target_pos_stacked)

            # Remove source dimension: (K, T)
            ranges = ranges[0, :, :]

        else:
            # STEPPED mode: positions are (3,)
            # Reshape to (1, 1, 3) for compute_range_km
            source_pos = source_pos[np.newaxis, np.newaxis, :]  # (1, 1, 3)

            # Stack targets and add time dimension: (K, 1, 3)
            target_pos_stacked = np.stack(target_positions, axis=0)[:, np.newaxis, :]

            # Compute range: returns (1, K, 1)
            ranges = compute_range_km(source_pos, target_pos_stacked)

            # Squeeze to (K,)
            ranges = ranges[0, :, 0]

        # Handle single target case
        if single_target:
            if mode == TimelineMode.BATCH:
                ranges = ranges[0, :]  # (T,)
            else:
                ranges = ranges[0]  # scalar

        return ranges

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
    def get_access(
        self,
        targets: Union[Asset, List[Asset]],
        constraints: Optional[Union['AccessConstraint', List['AccessConstraint']]] = None,
        constraints_per_source: Optional[Dict[int, List['AccessConstraint']]] = None
    ) -> np.ndarray:
        """Get access from all assets in array to target(s).

        Stacks per-asset access computations along leading axis. Supports either
        uniform constraints for all assets or per-source constraint specification.

        Args:
            targets (Asset | List[Asset]): Single target or list of targets
            constraints (AccessConstraint | List | None): Constraints applied to all
                assets. Mutually exclusive with constraints_per_source.
            constraints_per_source (Dict[int, List[AccessConstraint]] | None): Per-asset
                constraints indexed by asset position in array. Mutually exclusive with
                constraints.

        Returns:
            np.ndarray: Stacked access masks (0.0 or 1.0). Shape depends on mode:
                - BATCH, multiple targets: (n_assets, n_targets, n_times)
                - BATCH, single target: (n_assets, n_times)
                - STEPPED, multiple targets: (n_assets, n_targets)
                - STEPPED, single target: (n_assets,)

        Raises:
            ValueError: If both constraints and constraints_per_source are provided.
        """
        if constraints is not None and constraints_per_source is not None:
            raise ValueError(
                "Cannot specify both 'constraints' and 'constraints_per_source'. "
                "Use 'constraints' for uniform constraints or 'constraints_per_source' "
                "for per-asset constraints."
            )

        # Collect access from each asset
        access_results = []
        for idx, asset in enumerate(self._assets):
            # Determine which constraints to use
            if constraints_per_source is not None:
                asset_constraints = constraints_per_source.get(idx, None)
            else:
                asset_constraints = constraints

            # Evaluate access for this asset
            asset_access = asset.get_access(targets, asset_constraints)
            access_results.append(asset_access)

        # Stack along leading axis
        return np.stack(access_results, axis=0)

    def get_range_to(self, targets: Union[Asset, List[Asset]]) -> np.ndarray:
        """Get range from all assets in array to target(s).

        Stacks per-asset range computations along leading axis. Mode-aware based
        on timeline state.

        Args:
            targets (Asset | List[Asset]): Single target or list of targets

        Returns:
            np.ndarray: Stacked ranges in km. Shape depends on mode and targets:
                - BATCH, multiple targets: (n_assets, n_targets, n_times)
                - BATCH, single target: (n_assets, n_times)
                - STEPPED, multiple targets: (n_assets, n_targets)
                - STEPPED, single target: (n_assets,)
        """
        # Collect ranges from each asset
        ranges = []
        for asset in self._assets:
            asset_range = asset.get_range_to(targets)
            ranges.append(asset_range)

        # Stack along leading axis
        return np.stack(ranges, axis=0)

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
