# python imports
import numpy as np
from datetime import datetime
from enum import Enum
import hashlib

# COMET imports
from comet.utilities.constants import Constants as c
from comet.time.duration import Duration
from comet.time.epoch import Epoch


class TimelineMode(Enum):
    """Enum that defines the Timeline operation mode.

    BATCH: Timeline exposes full time_deltas array for batch operations.
    STEPPED: Timeline exposes a cursor into time_deltas for stepped iteration.
    """

    BATCH = "batch"
    STEPPED = "stepped"


# Module Methods
def get_time_deltas(start: Epoch, stop: Epoch, step: Duration) -> np.ndarray[float]:
    """Calculates the total time delta durations for a start/stop/step combination.

    Args:
        start (Epoch): Starting Epoch.
        stop (Epoch): Stopping Epoch.
        step (Duration): Step Duration.

    Returns:
        time_deltas (np.ndarray): Array of time deltas in seconds.
    """
    # Calculate total time
    total_seconds = (stop - start).total_seconds()

    # Create time deltas
    dt = step.total_seconds()
    time_deltas = np.arange(0, total_seconds + dt, dt)

    # Check that time deltas don't pass stop time
    if time_deltas[-1] > total_seconds:
        time_deltas = time_deltas[:-1:]

    return time_deltas


def get_relative_time_deltas(start: Epoch, stop: Epoch, step: Duration) -> np.ndarray[float]:
    """Calculates the relative time delta durations for a start/stop/step combination.

    Args:
        start (Epoch): Starting Epoch.
        stop (Epoch): Stopping Epoch.
        step (Duration): Step Duration.

    Returns:
        relative_time_deltas (np.ndarray): Array of relative time deltas in seconds.
    """
    # Calculate relative time deltas
    dt = get_time_deltas(start, stop, step)
    relative_time_deltas = np.append([0], [dt[i] - dt[i - 1] for i in range(1, len(dt))])

    return relative_time_deltas


def get_epoch_list(start: Epoch, stop: Epoch, step: Duration) -> tuple[np.ndarray]:
    """Calculates an array of Epochs for a start/stop/step combination.

    Args:
        start (Epoch): Starting Epoch.
        stop (Epoch): Stopping Epoch.
        step (Duration): Step Duration.

    Returns:
        epochs (np.ndarray): Array of Epochs.
        time_deltas (np.ndarray): Array of time deltas in seconds.
    """
    # Get time deltas in seconds
    time_deltas = get_time_deltas(start, stop, step)
    epochs = np.array([start + Duration(seconds=dt) for dt in time_deltas])

    return epochs, time_deltas


class Timeline:
    """Class that represents a scenario timeline, represented by a start, stop and step.

    Example Constructions:
        * tl = Timeline(start, stop, step)
    """

    def __init__(self, start: Epoch, stop: Epoch, step: Duration):
        """Defines the Timeline based on a start, stop and step.

        Args:
            start (Epoch): Starting Epoch of Timeline.
            stop (Epoch): Stopping Epoch of Timeline.
            step (Duration): Step Duration of Timeline.
        """
        # Error Checking
        if start >= stop:
            raise ValueError("Timeline(): stop Epoch must be before start Epoch")
        if step.total_seconds() <= 0.0:
            raise ValueError("Timeline(): step Duration must be greater than zero")

        # Assign Class Attributes
        self.start = start
        self.stop = stop
        self.step = step

        # Initialize mode to BATCH
        self._mode = TimelineMode.BATCH

        # Initialize stepped-mode cursor state
        self._index_now = 0
        self._now = start
        self._dt_now = 0.0

        # Initialize hash for cache invalidation
        self._hash = None
        self._compute_and_store_hash()

    def update(self, start: Epoch = None, stop: Epoch = None, step: Duration = None):
        """Updates Timeline properties.

        Args:
            start (Epoch, optional): New start Epoch. Defaults to None.
            stop (Epoch, optional): New stop Epoch. Defaults to None.
            step (Duration, optional): New step Duration. Defaults to None.
        """
        # Use existing values if None provided
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        if step is None:
            step = self.step

        # Error Checking
        if start >= stop:
            raise ValueError("Timeline(): stop Epoch must be before start Epoch")
        if step.total_seconds() <= 0.0:
            raise ValueError("Timeline(): step Duration must be greater than zero")

        # Update Timeline Properties
        self.start = start
        self.stop = stop
        self.step = step

        # Clear cached properties to force recalculation
        if hasattr(self, "time_deltas"):
            delattr(self, "time_deltas")
        if hasattr(self, "relative_time_deltas"):
            delattr(self, "relative_time_deltas")
        if hasattr(self, "durations"):
            delattr(self, "durations")
        if hasattr(self, "epochs"):
            delattr(self, "epochs")
        if hasattr(self, "julian_dates"):
            delattr(self, "julian_dates")
        if hasattr(self, "modified_julian_dates"):
            delattr(self, "modified_julian_dates")
        if hasattr(self, "unix"):
            delattr(self, "unix")

        # Reset stepped cursor to start
        self._index_now = 0
        self._now = self.start
        self._dt_now = 0.0

        # Recompute hash to invalidate caches
        self._compute_and_store_hash()

    def get_time_deltas(self) -> np.ndarray[float]:
        """Calculate time deltas for the current Timeline.
        If time_deltas have not been calculated before, it will same internally at self.time_deltas

        Returns:
            time_deltas (np.ndarray): Array of time deltas in seconds.
        """
        if not hasattr(self, "time_deltas"):
            # Calculate time deltas and assign to object
            self.time_deltas = get_time_deltas(self.start, self.stop, self.step)

        return self.time_deltas

    def get_relative_time_deltas(self) -> np.ndarray[float]:
        """Calculate relative time deltas for the current Timeline.
        If time_deltas have not been calculated before, it will save internally at self.relative_time_deltas

        Returns:
            relative_time_deltas (np.ndarray): Array of relative time deltas in seconds.
        """
        if not hasattr(self, "relative_time_deltas"):
            # Calculate time deltas and assign to object
            self.relative_time_deltas = get_relative_time_deltas(self.start, self.stop, self.step)

        return self.relative_time_deltas

    def get_epoch_list(self) -> np.ndarray[Epoch]:
        """Calculates an array of Epochs for the current Timeline.
        If epochs have not been calculated before, it will save internally at self.epoch

        Returns:
            epochs (np.ndarray): Array of Epochs
        """
        if not hasattr(self, "epochs"):
            # Calculate Epoch list and time deltas and assign to object
            self.epochs, self.time_deltas = get_epoch_list(self.start, self.stop, self.step)

        return self.epochs

    def get_duration_list(self) -> np.ndarray[Duration]:
        """Calculates an array of Durations relative to starting Epoch for the current Timeline.
        If durations have not been calculated before, it will save internally at self.durations

        Returns:
            durations (np.ndarray): Array of Durations
        """
        if not hasattr(self, "epochs"):
            # Calculate Epoch list and time deltas and assign to object
            self.epochs, self.time_deltas = get_epoch_list(self.start, self.stop, self.step)
        if not hasattr(self, "durations"):
            # Calculate Duration list and assign to object
            self.durations = np.array([epoch - self.start for epoch in self.epochs])

        return self.durations

    def get_julian_date_list(self) -> np.ndarray[Duration]:
        """Calculates an array of Julian Dates for the current Timeline.
        If Julian Dates have not been calculated before, it will save internally at self.julian_dates

        Returns:
            julian_dates (np.ndarray): Array of Julian Dates
        """
        if not hasattr(self, "time_deltas"):
            # Calculate time deltas and assign to object
            self.time_deltas = get_time_deltas(self.start, self.stop, self.step)
        if not hasattr(self, "julian_dates"):
            # Calculate Julian Dates list and assign to object
            self.julian_dates = np.array(
                [self.start.julian_date() + dt / c.DAY for dt in self.time_deltas]
            )

        return self.julian_dates

    def get_modified_julian_date_list(self) -> np.ndarray[Duration]:
        """Calculates an array of Modified Julian Dates for the current Timeline.
        If Modified Julian Dates have not been calculated before, it will save internally at self.modified_julian_dates

        Returns:
            modified_julian_dates (np.ndarray): Array of Modified Julain Dates
        """
        if not hasattr(self, "time_deltas"):
            # Calculate time deltas and assign to object
            self.time_deltas = get_time_deltas(self.start, self.stop, self.step)
        if not hasattr(self, "modified_julian_dates"):
            # Calculate Modified Julian Dates list and assign to object
            self.modified_julian_dates = np.array(
                [self.start.modified_julian_date() + dt / c.DAY for dt in self.time_deltas]
            )

        return self.modified_julian_dates

    def get_unix_list(self) -> np.ndarray[Duration]:
        """Calculates an array of UNIX Seconds for the current Timeline.
        If UNIX Seconds have not been calculated before, it will save internally at self.unix

        Returns:
            unix (np.ndarray): Array of UNIX Seconds
        """
        if not hasattr(self, "time_deltas"):
            # Calculate time deltas and assign to object
            self.time_deltas = get_time_deltas(self.start, self.stop, self.step)
        if not hasattr(self, "unix"):
            # Calculate UNIX Seconds list and assign to object
            self.unix = np.array([self.start.unix() + dt for dt in self.time_deltas])

        return self.unix

    def set_mode(self, mode: TimelineMode):
        """Set the Timeline operation mode.

        Args:
            mode (TimelineMode): The mode to set (BATCH or STEPPED).
        """
        if not isinstance(mode, TimelineMode):
            raise TypeError("Timeline.set_mode(): mode must be a TimelineMode enum")

        old_mode = self._mode
        self._mode = mode

        # Recompute hash if mode changed to invalidate caches
        if old_mode != mode:
            self._compute_and_store_hash()

    def get_mode(self) -> TimelineMode:
        """Get the current Timeline operation mode.

        Returns:
            mode (TimelineMode): The current mode (BATCH or STEPPED).
        """
        return self._mode

    def advance(self, step_amount: int = 1):
        """Advance the stepped-mode cursor by the specified number of steps.

        Only affects STEPPED mode. Clamps index to valid range [0, n_steps-1].

        Args:
            step_amount (int, optional): Number of steps to advance. Can be negative. Defaults to 1.
        """
        # Get time_deltas to determine valid range
        time_deltas = self.get_time_deltas()
        max_index = len(time_deltas) - 1

        # Update index with clamping
        new_index = self._index_now + step_amount
        self._index_now = max(0, min(new_index, max_index))

        # Update cursor state
        self._dt_now = time_deltas[self._index_now]
        self._now = self.start + Duration(seconds=self._dt_now)

        # Recompute hash to invalidate caches
        self._compute_and_store_hash()

    def reset(self):
        """Reset the stepped-mode to the beginning of the timeline."""
        self._index_now = 0
        self._now = self.start
        self._dt_now = 0.0

        # Recompute hash to invalidate caches
        self._compute_and_store_hash()

    def now_index(self) -> int:
        """Get the current stepped-mode cursor index.

        Returns:
            index (int): The current cursor index into time_deltas.
        """
        return self._index_now

    def now(self) -> Epoch:
        """Get the current stepped-mode cursor epoch.

        Returns:
            epoch (Epoch): The current cursor epoch.
        """
        return self._now

    def now_time_delta(self) -> float:
        """Get the current stepped-mode cursor time delta.

        Returns:
            dt (float): The current cursor time delta in seconds from start.
        """
        return self._dt_now

    def _compute_and_store_hash(self):
        """Compute and store a hash of the timeline state for cache invalidation.

        The hash includes: start, stop, step, mode, and cursor state.
        Assets can use this to detect timeline changes and invalidate caches.
        """
        # Create hash input from timeline state
        hash_input = (
            f"{self.start.julian_date()}"
            f"{self.stop.julian_date()}"
            f"{self.step.total_seconds()}"
            f"{self._mode.value}"
            f"{self._index_now}"
            f"{self._dt_now}"
        )

        # Compute hash
        self._hash = hashlib.md5(hash_input.encode()).hexdigest()

    def get_hash(self) -> str:
        """Get the current timeline hash for cache invalidation.

        Returns:
            hash (str): MD5 hash of the timeline state.
        """
        return self._hash

    def to_dict(self):
        """Method that creates a dictionary of required inputs for Timeline construction.

        Returns:
            constructor (dict): dictionary of required inputs for Timeline construction.
        """
        return {
            "type": "Timeline",
            "start": self.start.to_dict(),
            "stop": self.stop.to_dict(),
            "step": self.step.to_dict(),
            "mode": self._mode.value,
            "index_now": self._index_now,
            "dt_now": self._dt_now,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a Timeline from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            timeline (Timeline): Timeline
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "Timeline":
            raise ValueError("Timeline(): Invalid construction dictionary")

        # Construct Input objects
        start = Epoch.from_dict(dict["start"])
        stop = Epoch.from_dict(dict["stop"])
        step = Duration.from_dict(dict["step"])

        # Create timeline
        timeline = Timeline(start, stop, step)

        # Restore mode and cursor state if present (for backward compatibility)
        if "mode" in dict:
            timeline._mode = TimelineMode(dict["mode"])
        if "index_now" in dict:
            timeline._index_now = dict["index_now"]
        if "dt_now" in dict:
            timeline._dt_now = dict["dt_now"]
            timeline._now = start + Duration(seconds=dict["dt_now"])

        # Recompute hash with restored state
        timeline._compute_and_store_hash()

        return timeline


# Singleton Timeline Definition
TIMELINE = Timeline(
    start=Epoch(2024, 1, 1, 0, 0, 0), stop=Epoch(2024, 1, 2, 0, 0, 0), step=Duration(seconds=10.0)
)
