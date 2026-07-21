# python imports
import numpy as np
from enum import Enum
import itertools

# COMET imports
from comet.time import Epoch
from comet.time import Duration
from comet.components import Thruster
from comet.frames import ManeuverFrame


class Maneuver:
    """Base Class that contains Maneuver properties.

    Example Constructions:
        * maneuver = Maneuver(epoch, dv)
        * maneuver = Maneuver(epoch, dv, frame)
    """

    # Maneuver ID Counter
    id_counter = itertools.count()

    def __init__(
        self, epoch: Epoch, dv: np.ndarray, frame: ManeuverFrame | str = ManeuverFrame.ECI
    ):
        """Construct Maneuver.

        Args:
            epoch (Epoch): Burn Epoch.
            dv (np.ndarray): Delta-V vector in km/s.
            frame (ManeuverFrame|str): Frame Delta-V vector is provided. Defaults to ManeuverFrame.ECI.
        """
        # Assign Class attributes
        self.epoch = epoch
        self.dv = np.array(dv)

        # Assign Maneuver ID
        self.id = next(Maneuver.id_counter)

        # Assign ManeuverFrame
        if isinstance(frame, str):
            frame = ManeuverFrame(frame.upper())
        self.frame = frame

    def copy(self):
        """Returns a copy of the Maneuver.

        Returns:
            maneuver (Maneuver): Copy of Maneuver.
        """
        return Maneuver(self.epoch, self.dv, self.frame)

    def get_dv(self):
        """Returns the Delta-V vector in km/s.

        Returns:
            dv (np.ndarray): Delta-V vector in km/s
        """
        return self.dv

    def get_frame(self):
        """Returns the ManeuverFrame the Delta-V vector is in.

        Returns:
            frame (ManeuverFrame): ManeuverFrame
        """
        return self.frame

    def get_magnitude(self):
        """Returns the magnitude of the Delta-V vector in km/s.

        Returns:
            dv_mag (float): Magnitude of the Delta-V vector in km/s.
        """
        return np.linalg.norm(self.dv)

    def get_epoch(self):
        """Returns the Maneuver Epoch.

        Returns:
            epoch (Epoch): Maneuver Epoch.
        """
        return self.epoch

    def get_id(self):
        """Returns the Maneuver ID.

        Returns:
            id (int): Maneuver ID.
        """
        return self.id

    def update(
        self,
        epoch: Epoch = None,
        dv: np.ndarray = None,
        frame: ManeuverFrame | str = None,
        id: int = None,
    ):
        """Updates any Maneuver properties.

        Args:
            epoch (Epoch, optional): Burn Epoch. Defaults to None.
            dv (np.ndarray, optional): Delta-V vector in km/s. Defaults to None.
            frame (ManeuverFrame|str, optional): Frame Delta-V vector is provided. Defaults to None.
            id (int, optional): Maneuver ID. Defaults to None.
        """
        # Update any specified properties
        if epoch is not None:
            self.epoch = epoch
        if dv is not None:
            self.dv = np.array(dv)
        if frame is not None:
            if isinstance(frame, str):
                frame = ManeuverFrame(frame.upper())
            self.frame = frame
        if id is not None:
            self.id = id

    def duration_to_maneuver(self, epoch):
        """Returns the Duration between the specified Epoch and the Maneuver Epoch.

        Args:
            epoch (Epoch): Epoch to compare.

        Returns:
            duration (Duration): Duration between specified Epoch and Maneuver Epoch.
        """
        return epoch - self.epoch

    def to_dict(self):
        """Method that creates a dictionary of required inputs for Maneuver construction.

        Returns:
            constructor (dict): dictionary of required inputs for Maneuver construction.
        """
        return {
            "type": "Maneuver",
            "epoch": self.epoch.to_dict(),
            "dv": self.dv.tolist(),
            "frame": self.frame.value,
            "id": self.id,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a Maneuver from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            maneuver (Maneuver): Maneuver
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "Maneuver":
            raise ValueError("Maneuver(): Invalid construction dictionary")

        # Construct Inputs
        frame = ManeuverFrame(dict["frame"])
        epoch = Epoch.from_dict(dict["epoch"])
        maneuver = Maneuver(epoch, dict["dv"], frame)
        maneuver.id = dict["id"]

        return maneuver


class ImpulsiveManeuver(Maneuver):
    """Class that contains ImpulsiveManeuver properties.

    Example Constructions:
        * impulse = ImpulsiveManeuver(epoch, dv)
        * impulse = ImpulsiveManeuver(epoch, dv, frame)
    """

    def to_dict(self):
        """Method that creates a dictionary of required inputs for ImpulsiveManeuver construction.

        Returns:
            constructor (dict): dictionary of required inputs for ImpulsiveManeuver construction.
        """
        return {
            "type": "ImpulsiveManeuver",
            "epoch": self.epoch.to_dict(),
            "dv": self.dv.tolist(),
            "frame": self.frame.value,
            "id": self.id,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a ImpulsiveManeuver from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            maneuver (ImpulsiveManeuver): ImpulsiveManeuver
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "ImpulsiveManeuver":
            raise ValueError("ImpulsiveManeuver(): Invalid construction dictionary")

        # Construct Inputs
        frame = ManeuverFrame(dict["frame"])
        epoch = Epoch.from_dict(dict["epoch"])
        maneuver = ImpulsiveManeuver(epoch, dict["dv"], frame)
        maneuver.id = dict["id"]

        return maneuver

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, ImpulsiveManeuver):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare ImpulsiveManeuvers
        return self.__dict__ == other.__dict__

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, ImpulsiveManeuver):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare ImpulsiveManeuvers
        return self.__dict__ != other.__dict__

    def __str__(self):
        """String Representation of ImpulsiveManeuver Class"""
        return f"Maneuver {self.id:03d}: {self.get_magnitude():05.3f} km/s ({self.frame.value}) at {self.epoch}"

    def __repr__(self):
        """Class Representation of ImpulsiveManeuver Class"""
        return f"ImpulsiveManeuver({self.epoch}, {self.dv}, {self.frame.value}, {self.id})"


class FiniteManeuver(Maneuver):
    """Class that contains FiniteManeuver properties.

    Example Constructions:
        * finite = FiniteManeuver(epoch, dv)
        * finite = FiniteManeuver(epoch, dv, frame)
        * finite = FiniteManeuver(epoch, dv, frame, thruster)
    """

    def __init__(
        self,
        epoch: Epoch,
        dv: np.ndarray,
        frame: ManeuverFrame | str = ManeuverFrame.ECI,
        thruster: Thruster = Thruster(),
    ):
        """Construct FiniteManeuver. If no Thruster Component is provided, the default Thruster will be used:
            * 10 N, 200 Isp Thruster -> Thruster(thrust = 10.0, isp = 200.0)

        Args:
            epoch (Epoch): Burn Epoch.
            dv (np.ndarray): Delta-V vector in km/s.
            frame (ManeuverFrame|str): Frame Delta-V vector is provided. Defaults to ManeuverFrame.ECI.
            thruster (Thruster): Thruster Component performing the Maneuver.
        """
        # Initialize Parent Class
        super().__init__(epoch, np.array(dv), frame)

        # Assign Thruster Component
        self.thruster = thruster

    def burn_duration(self, mass: float | int):
        """Estimates the Burn Duration based on the Delta-V.

        Args:
            mass (float|int): Total system mass in kg.

        Returns:
            dt (float): Burn Duration in sec.
        """
        dt, _ = self.thruster.burn_given_dv(mass, self.dv)

        return dt

    def burn_start(self, mass: float | int):
        """Calculates the Burn Starting Epoch based on the Delta-V.

        Args:
            mass (float|int): Total system mass in kg.

        Returns:
            start (Epoch): Burn Starting Epoch.
        """
        dt, _ = self.thruster.burn_given_dv(mass, self.dv)

        return self.epoch - Duration(seconds=dt)

    def burn_stop(self, mass: float | int):
        """Calculates the Burn Ending Epoch based on the Delta-V.

        Args:
            mass (float|int): Total system mass in kg.

        Returns:
            stop (Epoch): Burn Ending Epoch.
        """
        dt, _ = self.thruster.burn_given_dv(mass, self.dv)

        return self.epoch + Duration(seconds=dt)

    def burn_start_stop(self, mass: float | int):
        """Calculates the Burn Starting and Ending Epoch based on the Delta-V.

        Args:
            mass (float|int): Total system mass in kg.

        Returns:
            start (Epoch): Burn Ending Epoch.
            stop (Epoch): Burn Ending Epoch.
        """
        dt, _ = self.thruster.burn_given_dv(mass, self.dv)

        return self.epoch - Duration(seconds=dt), self.epoch + Duration(seconds=dt)

    def mass_expended(self, mass: float | int):
        """Estimates the Mass Expended based on the Delta-V.

        Args:
            mass (float|int): Total system mass in kg.

        Returns:
            dm (float): Propellant mass expended in kg. Value is negative to show mass is being lost.
        """
        _, dm = self.thruster.burn_given_dv(mass, self.dv)

        return dm

    def to_dict(self):
        """Method that creates a dictionary of required inputs for FiniteeManeuver construction.

        Returns:
            constructor (dict): dictionary of required inputs for FiniteManeuver construction.
        """
        return {
            "type": "FiniteManeuver",
            "epoch": self.epoch.to_dict(),
            "dv": self.dv.tolist(),
            "frame": self.frame.value,
            "id": self.id,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a FiniteManeuver from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            maneuver (FiniteManeuver): FiniteManeuver
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "FiniteManeuver":
            raise ValueError("FiniteManeuver(): Invalid construction dictionary")

        # Construct Inputs
        frame = ManeuverFrame(dict["frame"])
        epoch = Epoch.from_dict(dict["epoch"])
        maneuver = FiniteManeuver(epoch, dict["dv"], frame)
        maneuver.id = dict["id"]

        return maneuver

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, FiniteManeuver):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare FiniteManeuvers
        return self.__dict__ == other.__dict__

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, FiniteManeuver):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare FiniteManeuvers
        return self.__dict__ != other.__dict__

    def __str__(self):
        """String Representation of FiniteManeuver Class"""
        return f"Maneuver {self.id:03d}: {self.get_magnitude():05.3f} km/s ({self.frame.value}) at {self.epoch}"

    def __repr__(self):
        """Class Representation of FiniteManeuver Class"""
        return f"FiniteManeuver({self.epoch}, {self.dv}, {self.frame.value}, {self.id})"
