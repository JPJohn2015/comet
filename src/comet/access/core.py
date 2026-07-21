"""Core access framework: base classes, enums, and batch evaluation engine.

This module provides the foundational abstractions for the access constraint system:
- AccessConstraint ABC implementing the metric-first contract
- Enums for evaluation modes and state requirements
- Data containers for inputs and custom states
- Batch evaluation engine for vectorized multi-constraint evaluation
- Access window extraction functions

The metric-first contract:
    Every constraint computes a physical metric and derives access by thresholding:
    - _metric(inputs) → raw values (abstract, subclass implements)
    - metric(inputs) → raw values (public passthrough)
    - access(inputs) → boolean mask (base applies bounds)
    - eval(inputs) → (metric, access) (computed once)
"""

# python imports
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import numpy.typing as npt

# COMET imports
from comet.time import Timeline, TimelineMode
from comet.time import Epoch
from comet.frames import StateFrame


class StateNeed(Enum):
    """Enumeration of state data requirements for access constraints.

    Constraints declare which bodies/platforms they need states for, allowing
    the resolver to fetch only required data and validate presence before evaluation.
    """

    SOURCE = "source"  # Source platform(s) state required
    TARGET = "target"  # Target platform(s) state required
    SUN = "sun"  # Sun ephemeris required
    MOON = "moon"  # Moon ephemeris required
    TIMELINE = "timeline"  # Timeline epochs required (for time windows)


class EvalMode(Enum):
    """Evaluation mode for access computation.

    Modes control which time points and states are evaluated:
    - CURRENT: Evaluate at the current propagator/timeline index only
    - CUSTOM: Evaluate against externally supplied state arrays (for optimization)
    - BATCH: Evaluate over the full timeline (default)
    """

    CURRENT = "current"  # Current propagator/timeline step only
    CUSTOM = "custom"  # User-supplied states (grid search, optimization)
    BATCH = "batch"  # Full timeline


@dataclass
class CustomStates:
    """Container for custom state arrays in CUSTOM evaluation mode.

    Allows passing pre-computed or candidate states for access evaluation without
    mutating propagator state. Used for trajectory optimization, grid searches, etc.

    All arrays must have compatible time dimensions when provided.
    """

    source_states: Optional[npt.NDArray] = None  # (N, T, 6) or (T, 6)
    target_states: Optional[npt.NDArray] = None  # (K, T, 6) or (T, 6)
    sun_position: Optional[npt.NDArray] = None  # (T, 3)
    moon_position: Optional[npt.NDArray] = None  # (T, 3)
    epochs: Optional[npt.NDArray] = None  # (T,) of Epoch objects

    def validate(self, required_needs: Set[StateNeed]) -> None:
        """Validate that all required states are present and have compatible shapes.

        Args:
            required_needs (Set[StateNeed]): Set of required state types

        Raises:
            ValueError: If required state is missing or shapes are incompatible
        """
        # Check presence
        if StateNeed.SOURCE in required_needs and self.source_states is None:
            raise ValueError("CustomStates: SOURCE required but source_states is None")
        if StateNeed.TARGET in required_needs and self.target_states is None:
            raise ValueError("CustomStates: TARGET required but target_states is None")
        if StateNeed.SUN in required_needs and self.sun_position is None:
            raise ValueError("CustomStates: SUN required but sun_position is None")
        if StateNeed.MOON in required_needs and self.moon_position is None:
            raise ValueError("CustomStates: MOON required but moon_position is None")
        if StateNeed.TIMELINE in required_needs and self.epochs is None:
            raise ValueError("CustomStates: TIMELINE required but epochs is None")

        # Check shape compatibility (all T dimensions must match)
        time_dims = []
        if self.source_states is not None:
            # Extract T from (N, T, 6) or (T, 6)
            t_dim = self.source_states.shape[-2]
            time_dims.append(("source_states", t_dim))
        if self.target_states is not None:
            t_dim = self.target_states.shape[-2]
            time_dims.append(("target_states", t_dim))
        if self.sun_position is not None:
            t_dim = self.sun_position.shape[0]
            time_dims.append(("sun_position", t_dim))
        if self.moon_position is not None:
            t_dim = self.moon_position.shape[0]
            time_dims.append(("moon_position", t_dim))
        if self.epochs is not None:
            t_dim = len(self.epochs)
            time_dims.append(("epochs", t_dim))

        # Verify all T dimensions match
        if time_dims:
            first_t = time_dims[0][1]
            for name, t_dim in time_dims[1:]:
                if t_dim != first_t:
                    raise ValueError(
                        f"CustomStates: Time dimension mismatch. "
                        f"{time_dims[0][0]} has T={first_t}, {name} has T={t_dim}"
                    )


@dataclass
class AccessInputs:
    """Container for all inputs to access constraint evaluation.

    Unified input structure passed to constraint metric/access/eval methods.
    The resolver populates this based on evaluation mode and constraint needs.
    """

    # Required
    mode: EvalMode

    # Optional states (populated based on constraint needs)
    source_positions: Optional[npt.NDArray] = None  # (N, T, 3) in km
    target_positions: Optional[npt.NDArray] = None  # (K, T, 3) in km
    source_states: Optional[npt.NDArray] = None  # (N, T, 6) in km, km/s
    target_states: Optional[npt.NDArray] = None  # (K, T, 6) in km, km/s
    sun_position: Optional[npt.NDArray] = None  # (T, 3) in km
    moon_position: Optional[npt.NDArray] = None  # (T, 3) in km
    epochs: Optional[npt.NDArray] = None  # (T,) of Epoch objects

    # Metadata
    time_count: int = 0  # Number of time points
    source_count: int = 0  # Number of sources
    target_count: int = 0  # Number of targets


class AccessConstraint(ABC):
    """Abstract base class for all access constraints.

    Implements the metric-first contract:
        1. Subclass implements _metric() to compute the physical quantity
        2. Base class provides metric() as a public passthrough
        3. Base class provides access() by thresholding the metric
        4. Base class provides eval() that computes both in one pass

    The bounds (min_value, max_value) define the access threshold:
        access = (min_value <= metric <= max_value)

    For boolean-only constraints (e.g. line-of-sight), the metric is the boolean
    itself and bounds are set to [-inf, inf] making the threshold a no-op.

    Attributes:
        min_value (float): Minimum acceptable metric value. Default -inf (no lower bound).
        max_value (float): Maximum acceptable metric value. Default +inf (no upper bound).
        needs (Set[StateNeed]): Set of required state types for this constraint.
    """

    def __init__(
        self,
        min_value: float = -np.inf,
        max_value: float = np.inf,
        needs: Optional[Set[StateNeed]] = None,
    ):
        """Initialize access constraint with bounds and state requirements.

        Args:
            min_value (float): Minimum acceptable metric value. Defaults to -inf.
            max_value (float): Maximum acceptable metric value. Defaults to +inf.
            needs (Set[StateNeed], optional): Required state types. Defaults to None.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.needs = needs if needs is not None else set()

    @abstractmethod
    def _metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute the raw physical metric for this constraint.

        This is the method subclasses implement. It should return the physical
        quantity (angle, range, etc.) with no thresholding applied.

        Args:
            inputs (AccessInputs): Resolved input states and positions

        Returns:
            npt.NDArray: Raw metric values, typically shape (N, K, T) for pairwise
                constraints or (N, T) / (K, T) for single-platform constraints.
        """
        pass

    def metric(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute and return the raw metric values.

        Public interface to get physical values without thresholding.

        Args:
            inputs (AccessInputs): Resolved input states and positions

        Returns:
            npt.NDArray: Raw metric values from _metric()
        """
        return self._metric(inputs)

    def access(self, inputs: AccessInputs) -> npt.NDArray:
        """Compute access mask by thresholding the metric.

        Applies the min/max bounds to the metric to produce a boolean access mask.

        Args:
            inputs (AccessInputs): Resolved input states and positions

        Returns:
            npt.NDArray: Access mask as float (0.0 or 1.0), same shape as metric.
                1.0 = access granted (metric within bounds)
                0.0 = no access (metric outside bounds)
        """
        metric_values = self._metric(inputs)

        # Apply bounds
        within_bounds = (metric_values >= self.min_value) & (metric_values <= self.max_value)

        return within_bounds.astype(float)

    def eval(self, inputs: AccessInputs) -> Tuple[npt.NDArray, npt.NDArray]:
        """Compute both metric and access in a single pass.

        More efficient than calling metric() and access() separately since the
        metric is computed only once (when access() is not overridden).

        Args:
            inputs (AccessInputs): Resolved input states and positions

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: (metric_values, access_mask)
        """
        metric_values = self._metric(inputs)

        # Check if access() has been overridden by subclass
        # If so, call it; otherwise apply threshold here
        if type(self).access is not AccessConstraint.access:
            # Subclass overrode access() - call it
            access_mask = self.access(inputs)
        else:
            # Use default thresholding
            within_bounds = (metric_values >= self.min_value) & (metric_values <= self.max_value)
            access_mask = within_bounds.astype(float)

        return metric_values, access_mask

    def __repr__(self) -> str:
        """String representation of constraint."""
        bounds_str = f"[{self.min_value}, {self.max_value}]"
        needs_str = ", ".join(n.value for n in self.needs)
        return f"{self.__class__.__name__}(bounds={bounds_str}, needs={{{needs_str}}})"


def resolve_inputs(
    mode: EvalMode,
    constraints: List[AccessConstraint],
    timeline: Optional[Timeline] = None,
    custom_states: Optional[CustomStates] = None,
    sources: Optional[List] = None,
    targets: Optional[List] = None,
    sun_propagator=None,
    moon_propagator=None,
) -> AccessInputs:
    """Resolve all required states for a set of constraints.

    Single resolver that handles all three evaluation modes (CURRENT, CUSTOM, BATCH).
    Fetches only the states required by at least one constraint and validates presence.

    This is the only place that:
    - Fetches sun/moon ephemeris (at most once, only if needed)
    - Validates custom state shape agreement
    - Stacks platform states into (N, T, 6) arrays

    Args:
        mode (EvalMode): Evaluation mode (CURRENT, CUSTOM, or BATCH)
        constraints (List[AccessConstraint]): Constraints to evaluate
        timeline (Timeline, optional): Timeline for BATCH/CURRENT modes
        custom_states (CustomStates, optional): States for CUSTOM mode
        sources (List, optional): Source platforms (Assets)
        targets (List, optional): Target platforms (Assets)
        sun_propagator: Sun propagator for ephemeris
        moon_propagator: Moon propagator for ephemeris

    Returns:
        AccessInputs: Populated input container with all required states

    Raises:
        ValueError: If required inputs are missing or incompatible
    """
    # Compute union of all state needs
    all_needs = set()
    for constraint in constraints:
        all_needs.update(constraint.needs)

    # Validate mode-specific requirements
    if mode == EvalMode.CUSTOM:
        if custom_states is None:
            raise ValueError("CUSTOM mode requires custom_states")
        custom_states.validate(all_needs)

    elif mode in (EvalMode.BATCH, EvalMode.CURRENT):
        if timeline is None:
            raise ValueError(f"{mode.value.upper()} mode requires timeline")

    # Initialize inputs container
    inputs = AccessInputs(mode=mode)

    # Handle CUSTOM mode
    if mode == EvalMode.CUSTOM:
        # Extract dimensions from custom states
        if custom_states.source_states is not None:
            source_states = custom_states.source_states
            # Handle both (N, T, 6) and (T, 6) shapes
            if source_states.ndim == 2:
                # (T, 6) -> (1, T, 6)
                source_states = source_states[np.newaxis, ...]

            N, T, _ = source_states.shape
            inputs.source_states = source_states
            inputs.source_positions = source_states[..., :3]
            inputs.source_count = N
            inputs.time_count = T

        if custom_states.target_states is not None:
            target_states = custom_states.target_states
            # Handle both (K, T, 6) and (T, 6) shapes
            if target_states.ndim == 2:
                # (T, 6) -> (1, T, 6)
                target_states = target_states[np.newaxis, ...]

            K, T, _ = target_states.shape
            inputs.target_states = target_states
            inputs.target_positions = target_states[..., :3]
            inputs.target_count = K

            # Update time_count if not already set
            if inputs.time_count == 0:
                inputs.time_count = T

        # Add sun/moon positions if provided
        if custom_states.sun_position is not None:
            inputs.sun_position = custom_states.sun_position
            if inputs.time_count == 0:
                inputs.time_count = len(custom_states.sun_position)

        if custom_states.moon_position is not None:
            inputs.moon_position = custom_states.moon_position
            if inputs.time_count == 0:
                inputs.time_count = len(custom_states.moon_position)

        # Add epochs if provided
        if custom_states.epochs is not None:
            inputs.epochs = custom_states.epochs
            if inputs.time_count == 0:
                inputs.time_count = len(custom_states.epochs)

        return inputs

    # Handle BATCH and CURRENT modes
    if mode in (EvalMode.BATCH, EvalMode.CURRENT):
        # For CURRENT mode, we need to temporarily switch timeline to STEPPED mode
        # and restore it after extraction
        if mode == EvalMode.CURRENT:
            original_mode = timeline.get_mode()
            if original_mode != TimelineMode.STEPPED:
                timeline.set_mode(TimelineMode.STEPPED)

        # Extract source states if needed
        if StateNeed.SOURCE in all_needs and sources:
            source_states_list = []
            for source in sources:
                state = source.get_state(frame=StateFrame.ECI)
                # Ensure at least 2D
                if state.ndim == 1:
                    state = state[np.newaxis, :]
                source_states_list.append(state)

            # Stack along first axis: (N, T, 6) or (N, 6) for CURRENT
            source_states = (
                np.stack(source_states_list, axis=0)
                if len(source_states_list) > 1
                else source_states_list[0][np.newaxis, ...]
            )
            inputs.source_states = source_states
            inputs.source_positions = source_states[..., :3]
            inputs.source_count = len(sources)

            # Set time_count from source states
            if source_states.ndim == 3:
                inputs.time_count = source_states.shape[1]
            else:
                inputs.time_count = 1

        # Extract target states if needed
        if StateNeed.TARGET in all_needs and targets:
            target_states_list = []
            for target in targets:
                state = target.get_state(frame=StateFrame.ECI)
                # Ensure at least 2D
                if state.ndim == 1:
                    state = state[np.newaxis, :]
                target_states_list.append(state)

            # Stack along first axis: (K, T, 6) or (K, 6) for CURRENT
            target_states = (
                np.stack(target_states_list, axis=0)
                if len(target_states_list) > 1
                else target_states_list[0][np.newaxis, ...]
            )
            inputs.target_states = target_states
            inputs.target_positions = target_states[..., :3]
            inputs.target_count = len(targets)

            # Set time_count if not already set
            if inputs.time_count == 0:
                if target_states.ndim == 3:
                    inputs.time_count = target_states.shape[1]
                else:
                    inputs.time_count = 1

        # Fetch sun position if needed
        if StateNeed.SUN in all_needs and sun_propagator is not None:
            sun_state = sun_propagator.get_state(frame=StateFrame.ECI)
            # Extract position only (first 3 columns)
            if sun_state.ndim == 2:
                inputs.sun_position = sun_state[:, :3]
            else:
                inputs.sun_position = sun_state[:3][np.newaxis, :]

            if inputs.time_count == 0:
                inputs.time_count = len(inputs.sun_position)

        # Fetch moon position if needed
        if StateNeed.MOON in all_needs and moon_propagator is not None:
            moon_state = moon_propagator.get_state(frame=StateFrame.ECI)
            # Extract position only (first 3 columns)
            if moon_state.ndim == 2:
                inputs.moon_position = moon_state[:, :3]
            else:
                inputs.moon_position = moon_state[:3][np.newaxis, :]

            if inputs.time_count == 0:
                inputs.time_count = len(inputs.moon_position)

        # Get timeline epochs if needed
        if StateNeed.TIMELINE in all_needs:
            if mode == EvalMode.BATCH:
                inputs.epochs = np.array(timeline.get_epoch_list())
            else:  # CURRENT mode
                inputs.epochs = np.array([timeline.now()])

            if inputs.time_count == 0:
                inputs.time_count = len(inputs.epochs)

        # Restore original timeline mode if we changed it
        if mode == EvalMode.CURRENT:
            if original_mode != TimelineMode.STEPPED:
                timeline.set_mode(original_mode)

        return inputs

    # Should not reach here
    raise ValueError(f"Unsupported evaluation mode: {mode}")


def evaluate_composite_access(
    sources: List,
    targets: List,
    constraints_per_source: Dict[int, List[AccessConstraint]],
    mode: EvalMode = EvalMode.BATCH,
    timeline: Optional[Timeline] = None,
    custom_states: Optional[CustomStates] = None,
) -> npt.NDArray:
    """Evaluate composite access for multiple sources, targets, and constraints.

    Single-pass batch evaluation engine. No source loop, no target loop - only one
    loop over unique constraint types. Algorithm:
    1. Build union of all constraints across sources and OR their state needs
    2. Resolve states once via resolve_inputs()
    3. Score each unique constraint once over full (N, K, T) grid
    4. Mask each source to only its declared constraints, ANDing into composite

    Args:
        sources (List): Source platforms (Assets)
        targets (List): Target platforms (Assets)
        constraints_per_source (Dict[int, List[AccessConstraint]]):
            Map from source index to list of constraints for that source
        mode (EvalMode): Evaluation mode. Defaults to BATCH.
        timeline (Timeline, optional): Timeline for BATCH/CURRENT modes
        custom_states (CustomStates, optional): States for CUSTOM mode

    Returns:
        npt.NDArray: Composite access mask, shape (N, K, T).
            1.0 = all constraints satisfied, 0.0 = at least one constraint failed
    """
    # Step 1: Build union of all constraints across sources
    all_constraints = []
    for constraints in constraints_per_source.values():
        all_constraints.extend(constraints)

    # If no constraints, return all-access
    if not all_constraints:
        N = len(sources) if sources else 0
        K = len(targets) if targets else 0
        # Determine T from timeline or custom_states
        if mode == EvalMode.CUSTOM and custom_states is not None:
            if custom_states.source_states is not None:
                T = custom_states.source_states.shape[-2]
            elif custom_states.target_states is not None:
                T = custom_states.target_states.shape[-2]
            else:
                T = 1
        elif timeline is not None:
            if mode == EvalMode.CURRENT:
                T = 1
            else:
                T = len(timeline.get_epoch_list())
        else:
            T = 1

        return np.ones((N, K, T))

    # Step 2: Resolve states once via resolve_inputs()
    inputs = resolve_inputs(
        mode=mode,
        constraints=all_constraints,
        timeline=timeline,
        custom_states=custom_states,
        sources=sources,
        targets=targets,
        sun_propagator=None,  # TODO: Add sun/moon propagator support in Phase 6
        moon_propagator=None,
    )

    N = inputs.source_count
    K = inputs.target_count
    T = inputs.time_count

    # Step 3: Score each unique constraint once over full (N, K, T) grid
    # Build map from constraint type to access results
    constraint_results: Dict[int, npt.NDArray] = {}

    for constraint in all_constraints:
        # Use id() to identify unique constraint instances
        constraint_id = id(constraint)

        # Skip if we've already evaluated this constraint instance
        if constraint_id in constraint_results:
            continue

        # Evaluate the constraint
        access_mask = constraint.access(inputs)

        # Store result
        constraint_results[constraint_id] = access_mask

    # Step 4: Mask each source to only its declared constraints, ANDing into composite
    # Initialize composite access mask: start with all-access (1.0)
    composite_access = np.ones((N, K, T))

    # For each source, AND together all its constraints
    for source_idx, constraints in constraints_per_source.items():
        if not constraints:
            continue

        # Start with all-access for this source
        source_composite = np.ones((1, K, T))

        # AND each constraint's result for this source
        for constraint in constraints:
            constraint_id = id(constraint)
            access_mask = constraint_results[constraint_id]

            # Extract this source's slice from the (N, K, T) result
            # Handle case where constraint result might be (K, T) or (1, K, T)
            if access_mask.ndim == 2:
                # (K, T) - broadcast to (1, K, T)
                source_mask = access_mask[np.newaxis, ...]
            elif access_mask.shape[0] == 1:
                # (1, K, T) - already correct
                source_mask = access_mask
            else:
                # (N, K, T) - extract this source's slice
                source_mask = access_mask[source_idx:source_idx+1, :, :]

            # AND into source composite
            source_composite = source_composite * source_mask

        # Place source composite into overall composite
        composite_access[source_idx:source_idx+1, :, :] = source_composite

    return composite_access


def extract_access_windows(
    access_mask: npt.NDArray, timeline: Timeline, source_idx: int = 0, target_idx: int = 0
) -> List[Tuple[Epoch, Epoch]]:
    """Extract continuous access windows from an access mask.

    Identifies contiguous blocks of access (1.0) in the mask and returns them
    as (start_epoch, end_epoch) tuples.

    Args:
        access_mask (npt.NDArray): Access mask, shape (N, K, T) or (T,)
        timeline (Timeline): Timeline providing epoch information
        source_idx (int): Source index to extract. Defaults to 0.
        target_idx (int): Target index to extract. Defaults to 0.

    Returns:
        List[Tuple[Epoch, Epoch]]: List of (start, end) epoch pairs for each window
    """
    # Extract the specific source-target timeline
    if access_mask.ndim == 3:
        mask = access_mask[source_idx, target_idx, :]
    elif access_mask.ndim == 1:
        mask = access_mask
    else:
        raise ValueError(f"Unsupported access_mask shape: {access_mask.shape}")

    # Find transitions: 0->1 (access starts) and 1->0 (access ends)
    mask_bool = mask > 0.5
    transitions = np.diff(mask_bool.astype(int))

    # Indices where access starts (0->1 transition)
    start_indices = np.where(transitions == 1)[0] + 1

    # Indices where access ends (1->0 transition)
    end_indices = np.where(transitions == -1)[0] + 1

    # Handle edge cases: access ongoing at start/end
    if mask_bool[0]:
        start_indices = np.concatenate([[0], start_indices])
    if mask_bool[-1]:
        end_indices = np.concatenate([end_indices, [len(mask)]])

    # Build windows
    epochs = timeline.get_epoch_list()
    windows = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        start_epoch = epochs[start_idx]
        # End epoch is the last epoch with access, or interpolate between last access and first no-access
        end_epoch = epochs[min(end_idx, len(epochs) - 1)]
        windows.append((start_epoch, end_epoch))

    return windows


def extract_access_windows_indices(
    access_mask: npt.NDArray, source_idx: int = 0, target_idx: int = 0
) -> List[Tuple[int, int]]:
    """Extract continuous access windows as timeline index ranges.

    Similar to extract_access_windows but returns integer indices instead of epochs.
    Useful when working directly with array indices.

    Args:
        access_mask (npt.NDArray): Access mask, shape (N, K, T) or (T,)
        source_idx (int): Source index to extract. Defaults to 0.
        target_idx (int): Target index to extract. Defaults to 0.

    Returns:
        List[Tuple[int, int]]: List of (start_idx, end_idx) pairs for each window.
            Indices are inclusive: [start_idx, end_idx].
    """
    # Extract the specific source-target timeline
    if access_mask.ndim == 3:
        mask = access_mask[source_idx, target_idx, :]
    elif access_mask.ndim == 1:
        mask = access_mask
    else:
        raise ValueError(f"Unsupported access_mask shape: {access_mask.shape}")

    # Handle empty mask
    if len(mask) == 0:
        return []

    # Find transitions
    mask_bool = mask > 0.5
    transitions = np.diff(mask_bool.astype(int))

    # Start and end indices
    # transitions == 1: access starts at next index
    # transitions == -1: access ends at current index (before transition)
    start_indices = np.where(transitions == 1)[0] + 1
    end_indices = np.where(transitions == -1)[0]

    # Handle edge cases: access ongoing at start/end
    if mask_bool[0]:
        start_indices = np.concatenate([[0], start_indices])
    if mask_bool[-1]:
        end_indices = np.concatenate([end_indices, [len(mask) - 1]])

    # Build windows
    windows = [(int(start), int(end)) for start, end in zip(start_indices, end_indices)]

    return windows
