# python imports
import numpy as np

# COMET imports
from comet.propagation.perturbation_model import (
    AtmosphereModel,
    SRPModel,
    NBodyModel,
    GravityPotentialModel,
)


class ForceModel:
    """Class that contains which perturbations and their respective models to be included.

    Example Constructions:
        * fm = ForceModel()
        * fm = ForceModel(drag, srp, nbody, gravity)
        * fm = ForceModel(drag, srp, nbody, gravity, atmosphere_model, srp_model, nbody_model, gravity_model)
    """

    def __init__(
        self,
        drag: bool = False,
        srp: bool = False,
        nbody: bool = False,
        gravity: bool = False,
        atmosphere_model: AtmosphereModel = AtmosphereModel.EXPONENTIAL,
        srp_model: SRPModel = SRPModel.SUN,
        nbody_model: NBodyModel = NBodyModel.SUNandMOON,
        gravity_model: GravityPotentialModel = GravityPotentialModel.J2andJ3,
    ):
        """Defines various ForceModel Properties.

        Args:
            drag (bool, optional): Include Drag Perturbations. Defaults to False.
            srp (bool, optional): Include SRP Perturbations. Defaults to False.
            nbody (bool, optional): Include N-Body Perturbations. Defaults to False.
            gravity (bool, optional): Include Gravitational Potential Perturbations. Defaults to False.
            atmosphere_model (AtmosphereModel, optional): Atmospheric Model. Defaults to AtmosphereModel.EXPONENTIAL.
            srp_model (SRPModel, optional): SRP Model. Defaults to SRPModel.SUN.
            nbody_model (NBodyModel, optional): N-Body Celestial Bodies to include. Defaults to NBodyModel.SUNandMOON.
            gravity_model (GravityPotentialModel, optional): Gravitational Potential Model.
                Defaults to GravityPotentialModel.J2andJ3.
        """

        # Assign which Perturbations are being included
        self.drag = drag
        self.srp = srp
        self.nbody = nbody
        self.gravity = gravity

        # Assign which Models are being used
        self.atmosphere_model = atmosphere_model
        self.srp_model = srp_model
        self.nbody_model = nbody_model
        self.gravity_model = gravity_model

    def copy(self):
        """Returns a copy of the ForceModel.

        Returns:
            force_model: Copy of the ForceModel.
        """
        return ForceModel(
            drag=self.drag,
            srp=self.srp,
            nbody=self.nbody,
            gravity=self.gravity,
            atmosphere_model=self.atmosphere_model,
            srp_model=self.srp_model,
            nbody_model=self.nbody_model,
            gravity_model=self.gravity_model,
        )

    def is_two_body(self):
        """Determines if any Perturbations are active.

        Returns:
            two_body (bool): Is the ForceModel Two Body.
        """
        if np.any([self.drag, self.srp, self.nbody, self.gravity]):
            return False
        else:
            return True

    def is_sun_needed(self):
        """Determines if Sun position calculations are needed given the current ForceModel configuration.

        Returns:
            sun_needed (bool): Are Sun calculations needed.
        """
        # Check relevant Models for Sun inclusion
        if (self.srp_model != SRPModel.ALBEDO) or (self.nbody_model != NBodyModel.MOON):
            return True
        else:
            return False

    def is_moon_needed(self):
        """Determines if Moon position calculations are needed given the current ForceModel configuration.

        Returns:
            moon_needed (bool): Are Moon calculations needed.
        """
        # Check relevant Models for Moon inclusion
        if self.nbody_model != NBodyModel.SUN:
            return True
        else:
            return False

    def update_atmosphere_model(self, atmosphere_model: AtmosphereModel):
        """Updates the AtmosphereModel for the ForceModel.

        Args:
            atmosphere_model (AtmosphereModel): New AtmosphereModel
        """
        # Update Atmosphere Model
        self.atmosphere_model = atmosphere_model

    def update_srp_model(self, srp_model: SRPModel):
        """Updates the SRPModel for the ForceModel.

        Args:
            srp_model (SRPModel): New SRPModel
        """
        # Update SRP Model
        self.srp_model = srp_model

    def update_nbody_model(self, nbody_model: NBodyModel):
        """Updates the NBodyModel for the ForceModel.

        Args:
            nbody_model (NBodyModel): New NBodyModel
        """
        # Update N-Body Model
        self.nbody_model = nbody_model

    def update_gravity_model(self, gravity_model: GravityPotentialModel):
        """Updates the GravityPotentialModel for the ForceModel.

        Args:
            gravity_model (GravityPotentialModel): New GravityPotentialModel
        """
        # Update Gravity Model
        self.gravity_model = gravity_model

    def to_dict(self):
        """Method that creates a dictionary of required inputs for ForceModel construction.

        Returns:
            constructor (dict): dictionary of required inputs for ForceModel construction.
        """
        return {
            "type": "ForceModel",
            "drag": self.drag,
            "srp": self.srp,
            "nbody": self.nbody,
            "gravity": self.gravity,
            "atmosphere_model": self.atmosphere_model.value,
            "srp_model": self.srp_model.value,
            "nbody_model": self.nbody_model.value,
            "gravity_model": self.gravity_model.value,
        }

    @staticmethod
    def from_dict(dict):
        """Method that creates a ForceModel from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            fm (ForceModel): ForceModel
        """
        # Check that dictionary of construction is of the correct type
        if dict["type"] != "ForceModel":
            raise ValueError("ForceModel(): Invalid construction dictionary")

        # Construct Input objects
        am = AtmosphereModel(dict["atmosphere_model"])
        srpm = SRPModel(dict["srp_model"])
        nbm = NBodyModel(dict["nbody_model"])
        gm = GravityPotentialModel(dict["gravity_model"])

        return ForceModel(
            dict["drag"], dict["srp"], dict["nbody"], dict["gravity"], am, srpm, nbm, gm
        )

    def __eq__(self, other) -> bool:
        """Override Equality operator."""
        # Error checking
        if not isinstance(other, ForceModel):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Force Models
        if isinstance(other, ForceModel):
            return np.all(self._dict__ == other.__dict__)

    def __ne__(self, other) -> bool:
        """Override Non-Equality operator."""
        # Error checking
        if not isinstance(other, ForceModel):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")

        # Compare Force Models
        if isinstance(other, ForceModel):
            return np.all(self._dict__ != other.__dict__)

    def __str__(self):
        """String Representation of ForceModel Class"""
        if self.is_two_body():
            return f"| TWO-BODY |"
        else:
            class_string = "| "
            if self.drag:
                class_string += f"Drag={self.atmosphere_model.name} | "
            if self.srp:
                class_string += f"SRP={self.srp_model.name} | "
            if self.nbody:
                class_string += f"NBody={self.nbody_model.name} | "
            if self.gravity:
                class_string += f"Gravity={self.gravity_model.name} | "

            return class_string

    def __repr__(self):
        """Class Representation of ForceModel Class"""
        return f"ForceModel(drag={self.drag}, srp={self.srp}, nbody={self.nbody}, gravity={self.gravity})"
