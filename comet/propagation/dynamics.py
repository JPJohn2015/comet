# python imports
import numpy as np
from poliastro.core.perturbations import (
    atmospheric_drag_exponential,
    J2_perturbation,
    J3_perturbation,
)

# COMET imports
from comet.utilities.constants import Constants as c
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.celestial.sun import Sun
from comet.celestial.moon import Moon
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.propagation.perturbation_model import AtmosphereModel, SRPModel, NBodyModel, GravityPotentialModel

# ---------------------------------------------------------------------------------------------------------------------------
def two_body(t, y):
    """Dynamics for Two-Body Gravitation.

    Args:
        t (float): Time in s.
        y (np.ndarray): Orbital State in km and km/s.

    Returns:
        dydt (np.ndarray): First Derivative of Orbital State in km/s and km/s^2
    """
    # Two Body propagation
    velocity = y[3:]
    acceleration = -c.MU_EARTH*y[:3]/(np.linalg.norm(y[:3])**3)

    return np.append(velocity, acceleration)

# ---------------------------------------------------------------------------------------------------------------------------
def full_perturbations(t, y, start_epoch: Epoch, force_model: ForceModel, sat_properties: SatelliteProperties):
    # Calculate current time 
    current_epoch = start_epoch + Duration(seconds=t)
    
    # Calculate Two Body propagation
    velocity = np.array(y[3:])
    acceleration = np.array(-c.MU_EARTH*y[:3]/(np.linalg.norm(y[:3])**3))

    # Calculate Atmospheric Drag
    if force_model.drag:
        # Calculate Atmospheric Drag using an Exponential Model
        #TODO: Implement Drag
        pass

    # Calculate Earth Oblateness
    if force_model.gravity:
        if force_model.gravity_model == GravityPotentialModel.EGM08:
            #TODO: Implement EGM08 Gravity Model
            pass
        else:
            if force_model.gravity_model in [GravityPotentialModel.J2, GravityPotentialModel.J2andJ3]:
                # Calculate J2 Gravitational Potential
                ax, ay, az = J2_perturbation(t, y, c.MU_EARTH, c.J2_EARTH, c.RADIUS_EARTH)

                # Add to Acceleration Vector
                acceleration += np.array([ax, ay, az])

            if force_model.gravity_model == GravityPotentialModel.J2andJ3:
                # Calculate J3 Gravitational Potential
                ax, ay, az = J3_perturbation(t, y, c.MU_EARTH, c.J3_EARTH, c.RADIUS_EARTH)

                # Add to Acceleration Vector
                acceleration += np.array([ax, ay, az])

    # Calculate N-Body
    if force_model.nbody:
        if force_model.nbody_model in [NBodyModel.SUN , NBodyModel.SUNandMOON]:
            # Calculate Third Body Acceleration due to Sun
            ax, ay, az = third_body_acceleration(t, y, c.MU_SUN, Sun().get_position(current_epoch))
            
            # Add to Acceleration Vector
            acceleration += np.array([ax, ay, az])

        if force_model.nbody_model in [NBodyModel.MOON , NBodyModel.SUNandMOON]:
            # Calculate Third Body Acceleration due to Moon
            ax, ay, az = third_body_acceleration(t, y, c.MU_MOON, Moon().get_position(current_epoch))

            # Add to Acceleration Vector
            acceleration += np.array([ax, ay, az])

    # Calculate Solar Radiation Pressure
    if force_model.srp:
        if force_model.srp_model in [SRPModel.SUN, SRPModel.SUNandALBEDO]:
            # Calculate Solar Radiation Pressure Acceleration due to Sun
            ax, ay, az = solar_pressure_acceleration(t, y, sat_properties.Cr, sat_properties.area_to_mass(), 
                                                     Sun().get_position(current_epoch))

            # Add to Acceleration Vector
            acceleration += np.array([ax, ay, az])

    # Calculate Thrust Acceleration
    #TODO: Implement finite burns

    return np.append(velocity, acceleration)

# ---------------------------------------------------------------------------------------------------------------------------
def third_body_acceleration(t, y, mu_3rd, r_third):
    """Calculates the N-Body Acceleration due to another celestial object.

    Args:
        t (float): Current Time in s.
        y (np.ndarray): Current Orbital State in km and km/s.
        mu_3rd (float): Gravitational Parameter of 3rd Body in km^3/s^2.
        r_third (np.ndarray): Position of 3rd Body in km.

    Returns:
        accel (float): ax, ay, az components of perturbing acceleration.
    """
    # Calculate the Position Vectors relative to 3rd Body
    sat_to_3rd = r_third - y[:3]

    # Calculate Accleration
    accel = mu_3rd*((sat_to_3rd/(np.linalg.norm(sat_to_3rd)**3)) - (r_third/(np.linalg.norm(r_third)**3)))

    return accel[0], accel[1], accel[2]

# ---------------------------------------------------------------------------------------------------------------------------
def solar_pressure_acceleration(t, y, Cr, a_to_m, r_sun):
    """Calculates the N-Body Acceleration due to another celestial object.

    Args:
        t (float): Current Time in s.
        y (np.ndarray): Current Orbital State in km and km/s.
        Cr (float): Coefficient of Reflectivity of the satellite.
        a_to_m (float): Area-to-Mass Ratio of the satellite in km^2/kg.
        r_sun (np.ndarray): Position of the Sun in km.

    Returns:
        accel (float): ax, ay, az components of perturbing acceleration.
    """
    # Calculate the Position Vectors relative to 3rd Body
    sat_to_sun = r_sun - y[:3]

    # Calculate Acceleration
    accel = (-c.SOLAR_PRESSURE*Cr*a_to_m)*(sat_to_sun/np.linalg.norm(sat_to_sun))

    return accel[0], accel[1], accel[2]