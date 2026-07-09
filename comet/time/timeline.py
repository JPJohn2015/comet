# python imports
import numpy as np
from datetime import datetime

# COMET imports
from comet.utilities.constants import Constants as c
from comet.time.duration import Duration
from comet.time.epoch import Epoch


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

        # Update only what has been calculated before to prevent calculating unneeded properties
        if not hasattr(self, "durations"):
            self.get_duration_list()
        if not hasattr(self, "epochs"):
            self.get_epoch_list()
        if not hasattr(self, "julian_dates"):
            self.get_julian_date_list()
        if not hasattr(self, "modified_julian_dates"):
            self.get_modified_julian_date_list()
        if not hasattr(self, "unix"):
            self.get_unix_list()
        if not hasattr(self, "time_deltas"):
            self.get_time_deltas()

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

        return Timeline(start, stop, step)


# Singleton Timeline Definition
TIMELINE = Timeline(
    start=Epoch(2024, 1, 1, 0, 0, 0), stop=Epoch(2024, 1, 2, 0, 0, 0), step=Duration(seconds=10.0)
)
