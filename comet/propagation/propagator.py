# python imports
from enum import Enum
import numpy as np
from scipy import integrate
from sgp4.api import Satrec
import math

# COMET imports
from comet.celestial.moon import Moon
from comet.utilities.constants import Constants as c
from comet.celestial.celestial_fidelity import CelestialFidelity
from comet.celestial.sun import Sun
from comet.state.lla import LLA
from comet.state.state import State
from comet.state.elements import Elements
from comet.state.tle import TLE
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.propagation.dynamics import two_body, full_perturbations
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.time.timeline import TIMELINE, get_relative_time_deltas
from comet.maneuver.maneuver import FiniteManeuver, Maneuver, ImpulsiveManeuver


class Integrator(Enum):
    """Enum for which SciPy integrator to use.

    Integrator options are: DOPRI5, DOP853, LSODA, VODE
    """

    # Status Enums
    DOPRI5 = "dopri5"
    DOP853 = "dop853"
    LSODA = "lsoda"
    VODE = "vode"


class Propagator:
    """Parent Class that Propagates States.

    Example Constructions:
        * propagator = Propagator(epoch, state)
        * propagator = Propagator(epoch, state, force_model, sat_properties)
        * propagator = Propagator(epoch, state, integrator, abs_tol, rel_tol, min_step, max_step)
    """

    def __init__(
        self,
        epoch: Epoch,
        state: State | Elements,
        force_model: ForceModel = ForceModel(),
        sat_properties: SatelliteProperties = SatelliteProperties(),
        integrator: Integrator = Integrator.DOPRI5,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
        min_step: float = 10.0,
        max_step: float = 60.0,
    ):

        # Assign Class Attributes
        self.epoch = epoch
        self.state = state
        self.force_model = force_model
        self.sat_properties = sat_properties

        # Assign Integrator Attributes
        self.integrator = integrator
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.min_step = min_step
        self.max_step = max_step

    def copy(self):
        """Returns a copy of the Propagator.

        Returns:
            Propagator: Copy of the Propagator.
        """
        return Propagator(
            self.epoch,
            self.state,
            self.force_model,
            self.sat_properties,
            self.integrator,
            self.abs_tol,
            self.rel_tol,
            self.min_step,
            self.max_step,
        )

    def get_state(self) -> State:
        """Returns the current State.

        Returns:
            state (State): Current State.
        """
        if isinstance(self.state, State):
            return self.state
        else:
            return self.state.to_state()

    def get_elements(self) -> Elements:
        """Returns the current Elements.

        Returns:
            elements (Elements): Current Elements.
        """
        if isinstance(self.state, Elements):
            return self.state
        else:
            return self.state.to_elements()

    def get_epoch(self) -> Epoch:
        """Returns the current Epoch.

        Returns:
            epoch (Epoch): Current Epoch.
        """
        return self.epoch

    def get_force_model(self) -> ForceModel:
        """Returns the current ForceModel for the Propagator.

        Returns:
            fm (ForceModel): Current ForceModel.
        """
        return self.force_model

    def get_satellite_properties(self) -> SatelliteProperties:
        """Returns the current SatelliteProperties for the Propagator.

        Returns:
            sp (SatelliteProperties): Current SatelliteProperties.
        """
        return self.sat_properties

    def project_to_epoch(self, epoch: Epoch, step: Duration = None) -> np.ndarray:
        """Returns the state projected to the specified Epoch. If a step is provided, the output is
        an array of states.

        NOTE: This method DOES NOT update the state internally for future propagation calls.

        Args:
            epoch (Epoch): Propagation Epoch.
            step (Duration, optional): Step Size for state array.

        Returns:
            state (np.ndarray): State array.
        """
        if step is None:
            # Calculate total propagation time and project to Epoch
            total_seconds = (epoch - self.epoch).total_seconds()
            return self._integrate([total_seconds])
        else:
            # Calculate total propagation time and project to Epoch
            time_deltas = get_relative_time_deltas(self.epoch, epoch, step)
            return self._integrate(time_deltas)

    def propagate_to_epoch(self, epoch: Epoch, step: Duration = None) -> np.ndarray:
        """Returns the state propagated to the specified Epoch. If a step is provided, the output is
        an array of states.

        NOTE: This method updates the state internally for future propagation calls.

        Args:
            epoch (Epoch): Propagation Epoch.
            step (Duration, optional): Step Size for state array.

        Returns:
            state (np.ndarray): State array.
        """
        if step is None:
            # Calculate total propagation time and project to Epoch
            total_seconds = (epoch - self.epoch).total_seconds()
            try:
                self.state = State(self._integrate([total_seconds]))
            except:
                self.state = self._integrate([total_seconds])
            self.epoch = epoch
            return self.state._raw
        else:
            # Calculate total propagation time and project to Epoch
            time_deltas = get_relative_time_deltas(self.epoch, epoch, step)
            state_array = self._integrate(time_deltas)
            try:
                self.state = State(state_array[-1, :])
            except:
                self.state = state_array[-1, :]
            self.epoch = epoch
            return state_array

    def project_to_duration(self, duration: Duration, step: Duration = None) -> np.ndarray:
        """Returns the state projected for a specific Duration. If a step is provided, the output is
        an array of states.

        NOTE: This method DOES NOT update the state internally for future propagation calls.

        Args:
            duration (Duration): Propagation Duration.
            step (Duration, optional): Step Size for state array.

        Returns:
            state (np.ndarray): State array.
        """
        if step is None:
            # Calculate total propagation time and project to Duration
            total_seconds = duration.total_seconds()
            return self._integrate([total_seconds])
        else:
            # Calculate total propagation time and project for Duration
            time_deltas = get_relative_time_deltas(self.epoch, self.epoch + duration, step)
            return self._integrate(time_deltas)

    def propagate_to_duration(self, duration: Duration, step: Duration = None) -> np.ndarray:
        """Returns the state projected for a specific Duration. If a step is provided, the output is
        an array of states.

        NOTE: This method updates the state internally for future propagation calls.

        Args:
            duration (Duration): Propagation Duration.
            step (Duration, optional): Step Size for state array.

        Returns:
            state (np.ndarray): State array.
        """
        if step is None:
            # Calculate total propagation time and project to Duration
            total_seconds = duration.total_seconds()
            try:
                self.state = State(self._integrate([total_seconds]))
            except:
                self.state = self._integrate([total_seconds])
            self.epoch = self.epoch + duration
            return self.state._raw
        else:
            # Calculate total propagation time and project for Duration
            time_deltas = get_relative_time_deltas(self.epoch, self.epoch + duration, step)
            state_array = self._integrate(time_deltas)
            try:
                self.state = State(state_array[-1, :])
            except:
                self.state = state_array[-1, :]
            self.epoch = self.epoch + duration
            return state_array

    def _integrate(self, relative_time_deltas: np.ndarray):
        """Core Method that handles Propagator Integration of Equations of Motion.

        Args:
            relative_time_deltas (np.ndarray): Array of Relative Time Deltas.

        Returns:
            state (np.ndarray): Nx6 State Array.
        """
        # Parse Initial state and duration
        if isinstance(self.state, State):
            state_raw = self.state._raw
        elif isinstance(self.state, Elements):
            state_raw = self.state.to_state()._raw
        else:
            raise ValueError("Propagator(): Invalid State Representation for state")

        # Define the Dynamics based on the ForceModel
        if self.force_model.is_two_body():
            prop = integrate.ode(two_body)
        else:
            prop = integrate.ode(full_perturbations)
            prop.set_f_params(self.epoch, self.force_model, self.sat_properties)

        # Build SciPy ODE integrator
        prop.set_integrator(
            self.integrator.value,
            atol=self.abs_tol,
            rtol=self.rel_tol,
            first_step=self.min_step,
            max_step=self.max_step,
        )
        prop.set_initial_value(state_raw)

        # Integrate State
        if len(relative_time_deltas) == 1:
            # No Step provided, only provide the final value of Propagation
            sign = np.sign(relative_time_deltas[0])
            if sign == 1.0:
                # Forward propagation
                while prop.successful() and (prop.t < relative_time_deltas[0]):
                    # Step forward
                    prop.integrate(prop.t + self.min_step)
            else:
                # Backward propagation
                while prop.successful() and (prop.t > relative_time_deltas[0]):
                    # Step forward
                    prop.integrate(prop.t - self.min_step)

            return prop.y
        else:
            # Step Provided, calculate and record all Propagation
            n = 0
            state = np.zeros((len(relative_time_deltas), 6))
            while prop.successful() and (n < len(relative_time_deltas)):
                # Step forward
                prop.integrate(prop.t + relative_time_deltas[n])

                # Append Arrays
                state[n, :] = prop.y
                n += 1

            return state

    def to_dict(self):
        """Method that creates a dictionary of required inputs for Propagator construction.

        Returns:
            constructor (dict): dictionary of required inputs for Propagator construction.
        """
        return {
            "type": "Propagator",
            "epoch": self.epoch.to_dict(),
            "state": self.state.to_dict(),
            "force_model": self.force_model.to_dict(),
            "satellite_properties": self.sat_properties.to_dict(),
            "integrator": self.integrator.value,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a Propagator from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            prop (Propagator): Propagator
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "Propagator":
            raise ValueError("Propagator(): Invalid construction dictionary")

        # Construct Inputs
        epoch = Epoch.from_dict(dict["epoch"])
        if dict["state"]["type"] == "State":
            state = State.from_dict(dict["state"])
        elif dict["state"]["type"] == "Elements":
            state = Elements.from_dict(dict["state"])
        else:
            raise ValueError("Propagator(): Invalid construction dictionary")
        force_model = ForceModel.from_dict(dict["force_model"])
        sat_properties = SatelliteProperties.from_dict(dict["sat_properties"])
        integrator = Integrator(dict["integrator"])

        return Propagator(
            epoch,
            state,
            force_model,
            sat_properties,
            integrator,
            dict["abs_tol"],
            dict["rel_tol"],
            dict["min_step"],
            dict["max_step"],
        )

    def __str__(self):
        """String Representation of Propagator Class"""
        return f"{self.epoch} {self.state}"

    def __repr__(self):
        """Class Representation of Propagator Class"""
        return f"Propagator({self.epoch}, {self.state})"


class SpacePropagator(Propagator):
    """Class that Propagates Satellite States.

    Example Constructions:
        * propagator = SpacePropagator(epoch, state)
        * propagator = SpacePropagator(epoch, state, force_model, sat_properties)
        * propagator = SpacePropagator(epoch, state, integrator, abs_tol, rel_tol, min_step, max_step)
    """

    def __init__(
        self,
        epoch: Epoch,
        state: State | Elements,
        force_model: ForceModel = ForceModel(),
        sat_properties: SatelliteProperties = SatelliteProperties(),
        integrator: Integrator = Integrator.DOPRI5,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
        min_step: float = 10.0,
        max_step: float = 60.0,
    ):

        # Initialize Parent Class
        super().__init__(
            epoch,
            state,
            force_model,
            sat_properties,
            integrator,
            abs_tol,
            rel_tol,
            min_step,
            max_step,
        )

    def add_maneuvers(self, maneuvers: Maneuver | list | np.ndarray):
        """Adds Maneuvers to the Propagator.

        Args:
            maneuvers (Maneuver|list|np.ndarray): Maneuver or Array of Maneuvers.
        """
        # Add Maneuvers to Propagator
        if isinstance(maneuvers, Maneuver):
            maneuvers = [maneuvers]
        self.maneuvers += list(maneuvers)

        # Sort Maneuvers by Epoch
        maneuver_jd = [maneuver.epoch.julian_date() for maneuver in self.maneuvers]
        _, self.maneuvers = zip(*sorted(zip(maneuver_jd, self.maneuvers)))

    def remove_maneuvers(self, maneuver_id: int | list[int]):
        """Removes Maneuvers from the Propagator.

        Args:
            maneuver_id (int|list[int]): Maneuver ID or list of Maneuver IDs.
        """
        # Remove Maneuvers to Propagator
        if isinstance(maneuver_id, int):
            maneuver_id = [maneuver_id]
        for maneuver in self.maneuvers:
            if maneuver.id in maneuver_id:
                self.maneuvers.remove(maneuver)

    def clear_maneuvers(self):
        """Clears all Maneuvers from the Propagator."""
        # Clear all Maneuvers
        self.maneuvers = []

    def _integrate(self, relative_time_deltas: np.ndarray):
        """Core Method that handles Propagator Integration of Equations of Motion.

        Args:
            relative_time_deltas (np.ndarray): Array of Relative Time Deltas.

        Returns:
            state (np.ndarray): Nx6 State Array.
        """

        # Get Maneuver Relative Time Deltas that will occur during propagation
        maneuvers = []
        for maneuver in self.maneuvers:
            # Check if maneuver is inside Propagation Bounds
            after_start = maneuver.epoch >= self.epoch
            before_end = maneuver.epoch <= self.epoch + Duration(
                seconds=np.sum(relative_time_deltas)
            )
            if after_start and before_end and type(maneuver) == ImpulsiveManeuver:
                maneuvers.append(maneuver)

        # No maneuvers in propagation window, integrate as normal
        if not maneuvers:
            state = self.__section_integrate(self.state, self.epoch, relative_time_deltas)

            return state

        # Create Section bounds if maneuvers do not align perfectly with starting and ending Epochs
        maneuver_time_deltas = [
            (maneuver.epoch - self.epoch).total_seconds() for maneuver in maneuvers
        ]
        time_deltas = np.cumsum(relative_time_deltas)
        if maneuver_time_deltas[0] != 0.0:
            maneuver_time_deltas = [0.0, *maneuver_time_deltas]
        if maneuver_time_deltas[-1] != np.sum(relative_time_deltas):
            maneuver_time_deltas = [*maneuver_time_deltas, np.sum(relative_time_deltas)]

        # Create Section Time Deltas
        section_time_deltas = []
        section_abs_time_deltas = []
        for i in range(0, len(maneuver_time_deltas) - 1):
            # Find time deltas inside the Maneuver Windows
            in_section = np.logical_and(
                time_deltas >= maneuver_time_deltas[i], time_deltas <= maneuver_time_deltas[i + 1]
            )
            section = relative_time_deltas[in_section]
            abs_section = time_deltas[in_section]

            # Modify time deltas in case Maneuvers do not line up on nice timesteps
            if abs_section[-1] != maneuver_time_deltas[i + 1]:
                section = np.append(section, maneuver_time_deltas[i + 1] - abs_section[-1])
                abs_section = np.append(abs_section, maneuver_time_deltas[i + 1])
            if abs_section[0] != maneuver_time_deltas[i]:
                section = np.append(abs_section[0] - maneuver_time_deltas[i], section)
                abs_section = np.append(maneuver_time_deltas[i], abs_section)
            section_time_deltas.append(section)
            section_abs_time_deltas.append(abs_section)

        # Initialize State and Epoch, checking for a Maneuver at start of propagation
        if maneuvers[0].epoch == self.epoch:
            # First Maneuver is at beginning of propagation
            maneuver_at_start = True
            if isinstance(self.state, State):
                current_state = self.state + np.append([0, 0, 0], maneuvers[0].dv)
            else:
                current_state = self.state.to_state() + np.append([0, 0, 0], maneuvers[0].dv)
        else:
            # No Maneuver at beginning
            maneuver_at_start = False
            current_state = self.state
        current_epoch = self.epoch
        state = np.expand_dims(np.array(self.state._raw), axis=0)

        # Loop through propagation sections
        for i in range(0, len(section_time_deltas)):
            # Integrate Section
            section_state = self._section_integrate(
                current_state, current_epoch, section_time_deltas[i]
            )

            # Determine if an intermediate step was added and adjust staet
            extra_last_step = section_abs_time_deltas[i][-1] not in time_deltas

            # Update Current State and Epoch with Maneuvers until the last section
            if i != len(section_time_deltas) - 1:
                if maneuver_at_start:
                    current_state = State(section_state[-1, :]) + np.append(
                        [0, 0, 0], maneuvers[i + 1].dv
                    )
                else:
                    current_state = State(section_state[-1, :]) + np.append(
                        [0, 0, 0], maneuvers[i].dv
                    )
                if extra_last_step:
                    current_epoch = current_epoch + Duration(seconds=np.sum(section_time_deltas[i]))
                else:
                    current_epoch = current_epoch + Duration(
                        seconds=np.sum(section_time_deltas[i][:-1])
                    )

            # Update Section lengths such that state appending matches relative_time_delta length
            section_state = section_state[1:, :]
            if extra_last_step:
                section_state = section_state[:-1, :]

            # Append State
            state = np.append(state, section_state, axis=0)

        # Check that State dimesions match time deltas
        if len(state) != len(relative_time_deltas):
            string = f"SpacePropagator(): State ({len(state)}) does not match time_deltas ({len(relative_time_deltas)})"
            raise Exception(string)

        return state

    def _section_integrate(self, initial_state, initial_epoch, relative_time_deltas: np.ndarray):
        """Core Method that handles Propagator Integration of Equations of Motion between
        impulsive maneuvers.

        Args:
            initial_state (State|Elements): Initial State.
            initial_epoch (Epoch): Initial Epoch.
            relative_time_deltas (np.ndarray): Array of Relative Time Deltas.

        Returns:
            state (np.ndarray): Nx6 State Array.
        """
        # Parse Initial state and duration
        if isinstance(initial_state, State):
            state_raw = initial_state._raw
        elif isinstance(initial_state, Elements):
            state_raw = initial_state.to_state()._raw
        else:
            raise ValueError("SpacePropagator(): Invalid State Representation for state")

        # Define the Dynamics based on the ForceModel
        if self.force_model.is_two_body():
            prop = integrate.ode(two_body)
        else:
            prop = integrate.ode(full_perturbations)
            prop.set_f_params(initial_epoch, self.force_model, self.sat_properties)

        # Build SciPy ODE integrator
        prop.set_integrator(
            self.integrator.value,
            atol=self.abs_tol,
            rtol=self.rel_tol,
            first_step=self.min_step,
            max_step=self.max_step,
        )
        prop.set_initial_value(state_raw)

        # Integrate State
        if len(relative_time_deltas) == 1:
            # No Step provided, only provide the final value of Propagation
            sign = np.sign(relative_time_deltas[0])
            if sign == 1.0:
                # Forward propagation
                while prop.successful() and (prop.t < relative_time_deltas[0]):
                    # Step forward
                    prop.integrate(prop.t + self.min_step)
            else:
                # Backward propagation
                while prop.successful() and (prop.t > relative_time_deltas[0]):
                    # Step forward
                    prop.integrate(prop.t - self.min_step)

            return prop.y
        else:
            # Step Provided, calculate and record all Propagation
            n = 0
            state = np.zeros((len(relative_time_deltas), 6))
            while prop.successful() and (n < len(relative_time_deltas)):
                # Step forward
                prop.integrate(prop.t + relative_time_deltas[n])

                # Append Arrays
                state[n, :] = prop.y
                n += 1

            return state

    def to_dict(self):
        """Method that creates a dictionary of required inputs for SpacePropagator construction.

        Returns:
            constructor (dict): dictionary of required inputs for SpacePropagator construction.
        """
        if self.maneuvers:
            maneuvers = [man.to_dict() for man in self.maneuvers]
        else:
            maneuvers = []

        return {
            "type": "SpacePropagator",
            "epoch": self.epoch.to_dict(),
            "state": self.state.to_dict(),
            "force_model": self.force_model.to_dict(),
            "satellite_properties": self.sat_properties.to_dict(),
            "integrator": self.integrator.value,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "maneuvers": maneuvers,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a SpacePropagator from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            prop (SpacePropagator): SpacePropagator
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "SpacePropagator":
            raise ValueError("SpacePropagator(): Invalid construction dictionary")

        # Construct Inputs
        epoch = Epoch.from_dict(dict["epoch"])
        if dict["state"]["type"] == "State":
            state = State.from_dict(dict["state"])
        elif dict["state"]["type"] == "Elements":
            state = Elements.from_dict(dict["state"])
        else:
            raise ValueError("SpacePropagator(): Invalid construction dictionary")
        force_model = ForceModel.from_dict(dict["force_model"])
        sat_properties = SatelliteProperties.from_dict(dict["sat_properties"])
        integrator = Integrator(dict["integrator"])

        # Build SpacePropagator
        prop = SpacePropagator(
            epoch,
            state,
            force_model,
            sat_properties,
            integrator,
            dict["abs_tol"],
            dict["rel_tol"],
            dict["min_step"],
            dict["max_step"],
        )

        # Build Maneuvers and load into Propagator
        built_maneuvers = []
        for maneuver in dict["maneuvers"]:
            if maneuver["type"] == "Maneuver":
                built_maneuvers.append(Maneuver.from_dict(maneuver))
            elif maneuver["type"] == "ImpulsiveManeuver":
                built_maneuvers.append(ImpulsiveManeuver.from_dict(maneuver))
            elif maneuver["type"] == "FiniteManeuver":
                built_maneuvers.append(FiniteManeuver.from_dict(maneuver))
            else:
                ValueError("SpacePropagator(): Invalid construction dictionary")
        prop.add_maneuvers(built_maneuvers)

        return prop


class GroundPropagator(Propagator):
    """Class that Propagates Ground States.

    Example Constructions:
        * propagator = GroundPropagator(epoch, lla)
        * propagator = GroundPropagator(epoch, lla, integrator, abs_tol, rel_tol, min_step, max_step)
    """

    def __init__(
        self,
        epoch: Epoch,
        state: LLA,
        integrator: Integrator = Integrator.DOPRI5,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
        min_step: float = 10.0,
        max_step: float = 60.0,
    ):

        # Get Ground state
        ground_state = State(state.eci_position(epoch))

        # Initialize Parent Class
        super().__init__(
            epoch,
            ground_state,
            ForceModel(),
            SatelliteProperties(0, 0, 0, 0, 0),
            integrator,
            abs_tol,
            rel_tol,
            min_step,
            max_step,
        )
        self.lla = state

    def get_state(self) -> State:
        """Returns the current LLA Coordinates as an ECI State.

        Returns:
            lla (LLA): Current ECI State of LLA Cordinates.
        """
        return self.state

    def get_elements(self) -> Elements:
        """Returns the current Elements.

        Returns:
            elements (Elements): Current Elements.
        """
        raise NotImplementedError(
            "GroundPropagator(): LLA Coordinates cannot be converted to Elements"
        )

    def get_lla(self) -> LLA:
        """Returns the current LLA Coordinates.

        Returns:
            lla (LLA): Current LLA Cordinates.
        """
        return self.lla

    def get_force_model(self) -> ForceModel:
        """Returns the current ForceModel for the Propagator.

        Returns:
            fm (ForceModel): Current ForceModel.
        """
        return NotImplementedError("GroundPropagator(): LLA Coordinates do not have a ForceModel")

    def get_satellite_properties(self) -> SatelliteProperties:
        """Returns the current SatelliteProperties for the Propagator.

        Returns:
            sp (SatelliteProperties): Current SatelliteProperties.
        """
        return NotImplementedError(
            "GroundPropagator(): LLA Coordinates do not have SatelliteProperties"
        )

    def _integrate(self, relative_time_deltas: np.ndarray):
        """Core Method that handles GroundPropagator Integration of ECEF Coordinates.

        Args:
            relative_time_deltas (np.ndarray): Array of Relative Time Deltas.

        Returns:
            state (np.ndarray): Nx6 State Array.
        """
        # Build list of epochs to calculate ECI position of the LLA coordinates
        epoch_list = [self.epoch + Duration(seconds=dt) for dt in relative_time_deltas]
        state = self.lla.eci_position(epoch_list)

        return state

    def to_dict(self):
        """Method that creates a dictionary of required inputs for GroundPropagator construction.

        Returns:
            constructor (dict): dictionary of required inputs for GroundPropagator construction.
        """
        return {
            "type": "GroundPropagator",
            "epoch": self.epoch.to_dict(),
            "state": self.state.to_dict(),
            "integrator": self.integrator.value,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a GroundPropagator from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            prop (GroundPropagator): GroundPropagator
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "GroundPropagator":
            raise ValueError("GroundPropagator(): Invalid construction dictionary")

        # Construct Inputs
        epoch = Epoch.from_dict(dict["epoch"])
        state = TLE.from_dict(dict["state"])
        integrator = Integrator(dict["integrator"])

        return GroundPropagator(
            epoch,
            state,
            integrator,
            dict["abs_tol"],
            dict["rel_tol"],
            dict["min_step"],
            dict["max_step"],
        )


class SunPropagator(Propagator):
    """Class that Propagates Sun States.

    Example Constructions:
        * propagator = SunPropagator(epoch)
    """

    def __init__(self, epoch: Epoch, fidelity: CelestialFidelity = CelestialFidelity.LoFi):

        self._sun = Sun()
        sun_state = State(self._sun.get_position(epoch), [0, 0, 0])

        # Initialize Parent Class
        super().__init__(
            epoch,
            sun_state,
            ForceModel(),
            SatelliteProperties(0, 0, 0, 0, 0),
            fidelity,
            1e-6,
            1e-6,
            10.0,
            60.0,
        )

    def get_state(self) -> np.ndarray:
        """Returns the current Sun State.

        NOTE: Velocity term is meaningless.

        Returns:
            sun (Sun): Current Sun State.
        """
        return self.state

    def get_elements(self) -> Elements:
        """Returns the current Elements.

        Returns:
            elements (Elements): Current Elements.
        """
        raise NotImplementedError(
            "SunPropagator(): Sun Coordinates cannot be converted to Elements"
        )

    def get_force_model(self) -> ForceModel:
        """Returns the current ForceModel for the Propagator.

        Returns:
            fm (ForceModel): Current ForceModel.
        """
        return NotImplementedError("SunPropagator(): Sun Coordinates does not have a ForceModel")

    def get_satellite_properties(self) -> SatelliteProperties:
        """Returns the current SatelliteProperties for the Propagator.

        Returns:
            sp (SatelliteProperties): Current SatelliteProperties.
        """
        return NotImplementedError(
            "SunPropagator(): Sun Coordinates does not have SatelliteProperties"
        )

    def _integrate(self, relative_time_deltas: np.ndarray):
        """Core Method that handles SunPropagator Integration of ECI Coordinates.

        Args:
            relative_time_deltas (np.ndarray): Array of Relative Time Deltas.

        Returns:
            state (np.ndarray): Nx6 State Array.
        """
        # Build list of epochs to calculate ECI position of the Sun
        jd_list = np.array([self.epoch.julian_date() + dt / c.DAY for dt in relative_time_deltas])
        state = self.state._sun_position_low_fidelity(self.state, jd_list)
        return state

    def to_dict(self):
        """Method that creates a dictionary of required inputs for SunPropagator construction.

        Returns:
            constructor (dict): dictionary of required inputs for SunPropagator construction.
        """
        return {
            "type": "SunPropagator",
            "epoch": self.epoch.to_dict(),
            "fidelity": self.integrator.value,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a SunPropagator from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            prop (SunPropagator): SunPropagator
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "SunPropagator":
            raise ValueError("SunPropagator(): Invalid construction dictionary")

        # Construct Inputs
        epoch = Epoch.from_dict(dict["epoch"])
        fidelity = CelestialFidelity(dict["fidelity"])

        return SunPropagator(epoch, fidelity)


class MoonPropagator(Propagator):
    """Class that Propagates Moon States.

    Example Constructions:
        * propagator = MoonPropagator(epoch)
    """

    def __init__(self, epoch: Epoch, fidelity: CelestialFidelity = CelestialFidelity.LoFi):

        self._moon = Moon()
        moon_state = State(self._moon.get_position(epoch), [0, 0, 0])

        # Initialize Parent Class
        super().__init__(
            epoch,
            moon_state,
            ForceModel(),
            SatelliteProperties(0, 0, 0, 0, 0),
            fidelity,
            1e-6,
            1e-6,
            10.0,
            60.0,
        )

    def get_state(self) -> np.ndarray:
        """Returns the current Moon State.

        NOTE: Velocity term is meaningless.

        Returns:
            moon (Moon): Current Moon State.
        """
        return self.state

    def get_elements(self) -> Elements:
        """Returns the current Elements.

        Returns:
            elements (Elements): Current Elements.
        """
        raise NotImplementedError(
            "MoonPropagator(): Moon Coordinates cannot be converted to Elements"
        )

    def get_force_model(self) -> ForceModel:
        """Returns the current ForceModel for the Propagator.

        Returns:
            fm (ForceModel): Current ForceModel.
        """
        return NotImplementedError("MoonPropagator(): Moon Coordinates does not have a ForceModel")

    def get_satellite_properties(self) -> SatelliteProperties:
        """Returns the current SatelliteProperties for the Propagator.

        Returns:
            sp (SatelliteProperties): Current SatelliteProperties.
        """
        return NotImplementedError(
            "MoonPropagator(): Moon Coordinates does not have SatelliteProperties"
        )

    def _integrate(self, relative_time_deltas: np.ndarray):
        """Core Method that handles MoonPropagator Integration of ECI Coordinates.

        Args:
            relative_time_deltas (np.ndarray): Array of Relative Time Deltas.

        Returns:
            state (np.ndarray): Nx6 State Array.
        """
        # Build list of epochs to calculate ECI position of the Sun
        jd_list = np.array([self.epoch.julian_date() + dt / c.DAY for dt in relative_time_deltas])
        state = self.state._moon_position_low_fidelity(self.state, jd_list)
        return state

    def to_dict(self):
        """Method that creates a dictionary of required inputs for MoonPropagator construction.

        Returns:
            constructor (dict): dictionary of required inputs for MoonPropagator construction.
        """
        return {
            "type": "MoonPropagator",
            "epoch": self.epoch.to_dict(),
            "fidelity": self.integrator.value,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a MoonPropagator from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            prop (MoonPropagator): MoonPropagator
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "MoonPropagator":
            raise ValueError("MoonPropagator(): Invalid construction dictionary")

        # Construct Inputs
        epoch = Epoch.from_dict(dict["epoch"])
        fidelity = CelestialFidelity(dict["fidelity"])

        return MoonPropagator(epoch, fidelity)


class TLEPropagator(Propagator):
    """Class that Propagates TLEs using SGP4 propagation model.

    Example Constructions:
        * propagator = TLEPropagator(tle)
        * propagator = TLEPropagator(tle, integrator, abs_tol, rel_tol, min_step, max_step)
    """

    def __init__(
        self,
        tle: TLE,
        force_model: ForceModel = ForceModel(),
        sat_properties: SatelliteProperties = SatelliteProperties(),
        integrator: Integrator = Integrator.DOPRI5,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
        min_step: float = 10.0,
        max_step: float = 60.0,
    ):

        # Initialize Parent Class
        self._tle = tle
        super().__init__(
            self._tle.epoch(),
            self._tle.to_state(),
            force_model,
            sat_properties,
            integrator,
            abs_tol,
            rel_tol,
            min_step,
            max_step,
        )

    def get_state(self) -> LLA:
        """Returns the current TLE State.

        Returns:
            state (State): Current TLE State.
        """
        return self.state

    def get_elements(self) -> Elements:
        """Returns the current Elements.

        Returns:
            elements (Elements): Current Elements.
        """
        raise self.state.to_elements()

    def get_tle(self) -> TLE:
        """Returns the TLE.

        Returns:
            tle (TLE): TLE.
        """
        return self._tle

    def get_force_model(self) -> ForceModel:
        """Returns the current ForceModel for the Propagator.

        Returns:
            fm (ForceModel): Current ForceModel.
        """
        return self.force_model

    def get_satellite_properties(self) -> SatelliteProperties:
        """Returns the current SatelliteProperties for the Propagator.

        Returns:
            sp (SatelliteProperties): Current SatelliteProperties.
        """
        return self.sat_properties

    def _integrate(self, relative_time_deltas: np.ndarray):
        """Core Method that handles GroundPropagator Integration of ECEF Coordinates.

        Args:
            relative_time_deltas (np.ndarray): Array of Relative Time Deltas.

        Returns:
            state (np.ndarray): Nx6 State Array.
        """
        # Build list of julian dates
        epoch_list = [self.epoch + Duration(seconds=dt) for dt in np.cumsum(relative_time_deltas)]
        jd, jd_frac = math.modf([np.floor(epoch.julian_date()) for epoch in epoch_list])

        # Create Satrec object and propagate using SGP4
        satellite = Satrec.twoline2rv(self.state._line1, self.state._line2)
        state = np.zeros((len(jd), 6))
        for i in range(0, len(jd)):
            _, r, v = satellite.sgp4(jd, jd_frac)
            state[i, :] = [*r, *v]

        return state

    def to_dict(self):
        """Method that creates a dictionary of required inputs for TLEPropagator construction.

        Returns:
            constructor (dict): dictionary of required inputs for TLEPropagator construction.
        """
        return {
            "type": "TLEPropagator",
            "tle": self._tle.to_dict(),
            "force_model": self.force_model.to_dict(),
            "satellite_properties": self.sat_properties.to_dict(),
            "integrator": self.integrator.value,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a TLEPropagator from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            prop (TLEPropagator): TLEPropagator
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "TLEPropagator":
            raise ValueError("TLEPropagator(): Invalid construction dictionary")

        # Construct Inputs
        tle = TLE.from_dict(dict["tle"])
        force_model = ForceModel.from_dict(dict["force_model"])
        sat_properties = SatelliteProperties.from_dict(dict["sat_properties"])
        integrator = Integrator(dict["integrator"])

        return TLEPropagator(
            tle,
            force_model,
            sat_properties,
            integrator,
            dict["abs_tol"],
            dict["rel_tol"],
            dict["min_step"],
            dict["max_step"],
        )


# Testing
if __name__ == "__main__":
    # Parse Timeline
    start = TIMELINE.start
    stop = TIMELINE.stop
    step = TIMELINE.step

    # Create Satellite Properties for Propagator
    start_elements = Elements(6378 + 780, 0, np.pi / 4, 0, 0, 0)
    epoch = Epoch(2024, 1, 1, 0, 0, 0)
    fm = ForceModel()
    sp = SatelliteProperties()

    impulse1 = ImpulsiveManeuver(Epoch(2024, 1, 1, 18, 0, 12), [1, -1, 0])
    impulse2 = ImpulsiveManeuver(Epoch(2024, 1, 1, 0, 0, 0), [1, -1, 0])
    impulse3 = ImpulsiveManeuver(Epoch(2024, 1, 2, 0, 0, 0), [1, -1, 0])
    maneuvers = [impulse1, impulse2, impulse3]

    space = SpacePropagator(epoch, start_elements)
    ground = GroundPropagator(epoch, LLA(0, 0, 0))
    sun = SunPropagator(epoch)
    moon = MoonPropagator(epoch)

    space.add_maneuvers(maneuvers)

    space_state = space.propagate_to_duration(Duration(days=2), step=Duration(seconds=60))
    ground_state = ground.propagate_to_duration(Duration(days=2), step=Duration(seconds=60))
    sun_state = sun.propagate_to_duration(Duration(days=2), step=Duration(seconds=60))
    moon_state = moon.propagate_to_duration(Duration(days=2), step=Duration(seconds=60))

    print(space_state.shape)
    print(ground_state.shape)
    print(sun_state.shape)
    print(moon_state.shape)
