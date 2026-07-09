# python imports
import numpy as np

# COMET imports
from comet.utilities.constants import Constants as c
from comet.state.state_conversions import cartesian_to_elements, elements_to_cartesian
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.celestial.sun import Sun
from comet.celestial.moon import Moon
from comet.propagation.force_model import ForceModel
from comet.propagation.satellite_properties import SatelliteProperties
from comet.propagation.perturbation_model import (
    AtmosphereModel,
    SRPModel,
    NBodyModel,
    GravityPotentialModel,
)
from comet.propagation.atmosphere_data import get_density_coesa76


def keplerian_propagate(initial_state: np.ndarray, time_deltas: np.ndarray) -> np.ndarray:
    """Propagates a state using analytical Keplerian (two-body) motion.

    This is much faster than numerical integration for unperturbed orbits,
    as it uses the analytical solution to Kepler's equation.

    For nearly circular orbits (e < 1e-8), uses a direct rotation method to
    avoid round-trip conversion issues with orbital elements.

    Reference: Vallado, "Fundamentals of Astrodynamics and Applications",
    4th edition, Algorithm 8 (Mean-to-True Anomaly), pg. 65-66.

    Args:
        initial_state (np.ndarray): Initial Cartesian state [x,y,z,vx,vy,vz] in km and km/s.
        time_deltas (np.ndarray): Array of time deltas from initial state in seconds.

    Returns:
        states (np.ndarray): Array of propagated states (N x 6) in km and km/s.
    """
    # Ensure time_deltas is array
    time_deltas = np.atleast_1d(time_deltas)

    # Extract position and velocity
    r0 = initial_state[:3]
    v0 = initial_state[3:]

    # Calculate orbital angular momentum vector
    h = np.cross(r0, v0)
    h_mag = np.linalg.norm(h)

    # Calculate semi-major axis and eccentricity
    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    energy = (v_mag**2) / 2.0 - c.MU_EARTH / r_mag
    a = -c.MU_EARTH / (2.0 * energy)

    # Eccentricity vector
    e_vec = ((v_mag**2 - c.MU_EARTH / r_mag) * r0 - np.dot(r0, v0) * v0) / c.MU_EARTH
    e = np.linalg.norm(e_vec)

    # For nearly circular orbits, use direct rotation method
    if e < 1e-8 and abs(a - r_mag) < 1.0:  # Circular orbit
        # Mean motion
        n = np.sqrt(c.MU_EARTH / (a**3))

        # Unit vector normal to orbit plane
        h_unit = h / h_mag

        # Propagate by rotating state vector
        states = np.zeros((len(time_deltas), 6))
        for idx, dt in enumerate(time_deltas):
            # Rotation angle
            theta = n * dt

            # Rotation matrix using Rodrigues' formula
            # R = I + sin(θ)*K + (1-cos(θ))*K²
            # where K is the skew-symmetric matrix of h_unit
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Rotate position
            r_new = (
                cos_theta * r0
                + sin_theta * np.cross(h_unit, r0)
                + (1 - cos_theta) * np.dot(h_unit, r0) * h_unit
            )

            # Rotate velocity
            v_new = (
                cos_theta * v0
                + sin_theta * np.cross(h_unit, v0)
                + (1 - cos_theta) * np.dot(h_unit, v0) * h_unit
            )

            states[idx, :3] = r_new
            states[idx, 3:] = v_new

        # Return single state if input was scalar, else return array
        if len(time_deltas) == 1:
            return states[0, :]
        else:
            return states

    # For eccentric orbits, use element conversion method
    # Convert initial state to orbital elements
    elements = cartesian_to_elements(initial_state)
    a, e, i, raan, aop, ta = elements

    # Calculate mean motion n = sqrt(mu/a^3)
    n = np.sqrt(c.MU_EARTH / (a**3))

    # Calculate initial mean anomaly from true anomaly
    # Eccentric anomaly from true anomaly
    E0 = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(ta / 2.0))
    # Mean anomaly from eccentric anomaly
    M0 = E0 - e * np.sin(E0)

    # Propagate mean anomaly: M = M0 + n*dt
    M_array = M0 + n * time_deltas

    # Wrap mean anomaly to [0, 2π]
    M_array = M_array % (2.0 * np.pi)

    # Convert back to true anomaly for each time
    ta_array = np.zeros_like(M_array)

    for idx, M in enumerate(M_array):
        # Solve Kepler's equation iteratively: E - e*sin(E) = M
        # Newton-Raphson iteration
        E = M  # Initial guess
        for _ in range(50):  # Max 50 iterations
            f = E - e * np.sin(E) - M
            fp = 1.0 - e * np.cos(E)
            E_new = E - f / fp
            if abs(E_new - E) < 1e-12:
                E = E_new
                break
            E = E_new

        # True anomaly from eccentric anomaly
        ta_array[idx] = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0))

    # Convert elements back to Cartesian states
    states = np.zeros((len(time_deltas), 6))
    for idx, ta_new in enumerate(ta_array):
        elements_new = np.array([a, e, i, raan, aop, ta_new])
        states[idx, :] = elements_to_cartesian(elements_new)

    # Return single state if input was scalar, else return array
    if len(time_deltas) == 1:
        return states[0, :]
    else:
        return states


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
    acceleration = -c.MU_EARTH * y[:3] / (np.linalg.norm(y[:3]) ** 3)

    return np.append(velocity, acceleration)


def full_perturbations(
    t, y, start_epoch: Epoch, force_model: ForceModel, sat_properties: SatelliteProperties
):
    # Calculate current time
    current_epoch = start_epoch + Duration(seconds=t)

    # Calculate Two Body propagation
    velocity = np.array(y[3:])
    acceleration = np.array(-c.MU_EARTH * y[:3] / (np.linalg.norm(y[:3]) ** 3))

    # Calculate Atmospheric Drag
    if force_model.drag:
        if force_model.atmosphere_model == AtmosphereModel.EXPONENTIAL:
            # Calculate Atmospheric Drag using Exponential Model
            ax, ay, az = atmospheric_drag_exponential(
                t, y, sat_properties.Cd, sat_properties.area_to_mass()
            )
            # Add to Acceleration Vector
            acceleration += np.array([ax, ay, az])
        elif force_model.atmosphere_model == AtmosphereModel.COESA76:
            # Calculate Atmospheric Drag using COESA76 Tables
            ax, ay, az = atmospheric_drag_coesa76(
                t, y, sat_properties.Cd, sat_properties.area_to_mass()
            )
            # Add to Acceleration Vector
            acceleration += np.array([ax, ay, az])

    # Calculate Earth Oblateness
    if force_model.gravity:
        if force_model.gravity_model in [
            GravityPotentialModel.J2,
            GravityPotentialModel.J2andJ3,
        ]:
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
        if force_model.nbody_model in [NBodyModel.SUN, NBodyModel.SUNandMOON]:
            # Calculate Third Body Acceleration due to Sun
            ax, ay, az = third_body_acceleration(t, y, c.MU_SUN, Sun().get_position(current_epoch))

            # Add to Acceleration Vector
            acceleration += np.array([ax, ay, az])

        if force_model.nbody_model in [NBodyModel.MOON, NBodyModel.SUNandMOON]:
            # Calculate Third Body Acceleration due to Moon
            ax, ay, az = third_body_acceleration(
                t, y, c.MU_MOON, Moon().get_position(current_epoch)
            )

            # Add to Acceleration Vector
            acceleration += np.array([ax, ay, az])

    # Calculate Solar Radiation Pressure
    if force_model.srp:
        r_sun = Sun().get_position(current_epoch)

        # Direct solar pressure
        if force_model.srp_model in [SRPModel.SUN, SRPModel.SUNandALBEDO]:
            ax, ay, az = solar_pressure_acceleration(
                t, y, sat_properties.Cr, sat_properties.area_to_mass(), r_sun
            )
            acceleration += np.array([ax, ay, az])

        # Earth albedo pressure
        if force_model.srp_model in [SRPModel.ALBEDO, SRPModel.SUNandALBEDO]:
            ax, ay, az = albedo_pressure_acceleration(
                t, y, sat_properties.Cr, sat_properties.area_to_mass(), r_sun
            )
            acceleration += np.array([ax, ay, az])

    # Calculate Thrust Acceleration
    # TODO: Implement finite burns

    return np.append(velocity, acceleration)


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
    accel = mu_3rd * (
        (sat_to_3rd / (np.linalg.norm(sat_to_3rd) ** 3))
        - (r_third / (np.linalg.norm(r_third) ** 3))
    )

    return accel[0], accel[1], accel[2]


def solar_pressure_acceleration(t, y, Cr, a_to_m, r_sun):
    """Calculates direct solar radiation pressure acceleration.

    Args:
        t (float): Current Time in s.
        y (np.ndarray): Current Orbital State in km and km/s.
        Cr (float): Coefficient of Reflectivity of the satellite.
        a_to_m (float): Area-to-Mass Ratio of the satellite in km^2/kg.
        r_sun (np.ndarray): Position of the Sun in km.

    Returns:
        tuple: (ax, ay, az) components of SRP acceleration in km/s^2.
    """
    # Calculate the Position Vectors relative to Sun
    sat_to_sun = r_sun - y[:3]

    # Calculate Acceleration
    # Note: Negative sign because pressure pushes away from Sun
    accel = (-c.SOLAR_PRESSURE * Cr * a_to_m) * (sat_to_sun / np.linalg.norm(sat_to_sun))

    return accel[0], accel[1], accel[2]


def albedo_pressure_acceleration(t, y, Cr, a_to_m, r_sun):
    """Calculates Earth albedo radiation pressure acceleration.

    Earth reflects sunlight (albedo effect), which exerts additional pressure
    on satellites. This is most significant for large area-to-mass ratio satellites
    and those in low Earth orbit where Earth subtends a large solid angle.

    The model assumes:
    - Average Earth albedo of ~0.3 (30% of sunlight is reflected)
    - Lambertian reflection (diffuse, not specular)
    - Only the illuminated and visible portion of Earth contributes

    Reference: Montenbruck & Gill, "Satellite Orbits", pg. 77-80
              Vallado, "Fundamentals of Astrodynamics and Applications", pg. 551

    Args:
        t (float): Current Time in s.
        y (np.ndarray): Current Orbital State in km and km/s.
        Cr (float): Coefficient of Reflectivity of the satellite.
        a_to_m (float): Area-to-Mass Ratio of the satellite in km^2/kg.
        r_sun (np.ndarray): Position of the Sun in km (ECI frame).

    Returns:
        tuple: (ax, ay, az) components of albedo pressure acceleration in km/s^2.
    """
    # Extract satellite position
    r_sat = y[:3]  # km
    r_sat_mag = np.linalg.norm(r_sat)

    # Check if satellite is above Earth's surface
    if r_sat_mag <= c.RADIUS_EARTH:
        return 0.0, 0.0, 0.0

    # Calculate Sun-Earth vector
    r_sun_mag = np.linalg.norm(r_sun)

    # Unit vectors
    e_sat = r_sat / r_sat_mag  # Unit vector from Earth to satellite
    e_sun = r_sun / r_sun_mag  # Unit vector from Earth to Sun

    # Calculate angle between satellite and Sun as seen from Earth
    cos_psi = np.dot(e_sat, e_sun)

    # If satellite is on night side relative to Sun, no albedo effect
    if cos_psi < 0:
        return 0.0, 0.0, 0.0

    # Calculate Earth's angular radius as seen from satellite
    sin_rho = c.RADIUS_EARTH / r_sat_mag

    # If Earth doesn't subtend much angle (high altitude), effect is negligible
    if sin_rho < 0.01:  # Less than ~0.6 degrees
        return 0.0, 0.0, 0.0

    # Calculate reflected solar flux at satellite position
    # The reflected intensity follows a Lambertian distribution
    # F_albedo = (albedo * F_solar * R_E^2 * cos(psi)) / r_sat^2
    # where cos(psi) is the angle factor for illumination

    # Geometric factor for visible illuminated Earth disk
    # For simplicity, use approximate model that accounts for:
    # 1. Earth's angular size from satellite
    # 2. Illumination geometry
    geometric_factor = c.EARTH_ALBEDO * (c.RADIUS_EARTH / r_sat_mag)**2 * cos_psi

    # Calculate albedo pressure acceleration
    # Direction is from Earth to satellite (reflected light pushes away from Earth)
    # Magnitude scales with solar pressure and geometric factors
    accel_mag = c.SOLAR_PRESSURE * Cr * a_to_m * geometric_factor
    accel = accel_mag * e_sat

    return accel[0], accel[1], accel[2]


def J2_perturbation(t, y, mu, J2, R):
    """Calculates the J2 perturbation acceleration due to Earth oblateness.

    Implementation based on standard J2 perturbation formulation.
    Reference: Vallado, "Fundamentals of Astrodynamics and Applications",
    4th edition, Algorithm 64, pg. 639-640.

    Args:
        t (float): Current Time in s (unused, included for consistency).
        y (np.ndarray): Current Orbital State in km and km/s.
        mu (float): Gravitational parameter in km^3/s^2.
        J2 (float): J2 zonal harmonic coefficient (dimensionless).
        R (float): Reference radius (Earth equatorial radius) in km.

    Returns:
        tuple: (ax, ay, az) components of J2 perturbing acceleration in km/s^2.
    """
    # Extract position
    x, y_pos, z = y[0], y[1], y[2]
    r = np.sqrt(x**2 + y_pos**2 + z**2)

    # Common factor
    factor = (3.0 / 2.0) * J2 * mu * (R**2) / (r**5)

    # Acceleration components
    ax = factor * x * (5.0 * (z**2) / (r**2) - 1.0)
    ay = factor * y_pos * (5.0 * (z**2) / (r**2) - 1.0)
    az = factor * z * (5.0 * (z**2) / (r**2) - 3.0)

    return ax, ay, az


def J3_perturbation(t, y, mu, J3, R):
    """Calculates the J3 perturbation acceleration due to Earth oblateness.

    Implementation based on standard J3 perturbation formulation.
    Reference: Vallado, "Fundamentals of Astrodynamics and Applications",
    4th edition, Algorithm 64, pg. 639-640.

    Args:
        t (float): Current Time in s (unused, included for consistency).
        y (np.ndarray): Current Orbital State in km and km/s.
        mu (float): Gravitational parameter in km^3/s^2.
        J3 (float): J3 zonal harmonic coefficient (dimensionless).
        R (float): Reference radius (Earth equatorial radius) in km.

    Returns:
        tuple: (ax, ay, az) components of J3 perturbing acceleration in km/s^2.
    """
    # Extract position
    x, y_pos, z = y[0], y[1], y[2]
    r = np.sqrt(x**2 + y_pos**2 + z**2)

    # Common factor
    factor = (1.0 / 2.0) * J3 * mu * (R**3) / (r**7)

    # Acceleration components
    ax = factor * 5.0 * x * z * (7.0 * (z**2) / (r**2) - 3.0)
    ay = factor * 5.0 * y_pos * z * (7.0 * (z**2) / (r**2) - 3.0)
    az = factor * (3.0 * (5.0 * (z**2) - (r**2)) * (7.0 * (z**2) / (r**2) - 1.0))

    return ax, ay, az


def atmospheric_drag_exponential(t, y, Cd, a_to_m):
    """Calculates atmospheric drag acceleration using exponential atmosphere model.

    Uses a simple exponential density model: ρ = ρ₀ * exp(-(h - h₀)/H)
    where h is altitude, ρ₀ is reference density, and H is scale height.

    Accounts for Earth's rotation by computing velocity relative to the
    rotating atmosphere.

    Reference: Vallado, "Fundamentals of Astrodynamics and Applications",
    4th edition, pg. 551-556.

    Args:
        t (float): Current Time in s (unused, included for consistency).
        y (np.ndarray): Current Orbital State in km and km/s.
        Cd (float): Coefficient of drag (dimensionless).
        a_to_m (float): Area-to-Mass Ratio of the satellite in km^2/kg.

    Returns:
        tuple: (ax, ay, az) components of drag acceleration in km/s^2.
    """
    # Extract position and velocity
    r_vec = y[:3]  # km
    v_vec = y[3:]  # km/s

    # Calculate altitude above Earth's surface
    r = np.linalg.norm(r_vec)
    altitude = r - c.RADIUS_EARTH  # km

    # Calculate atmospheric density using exponential model
    # ρ = ρ₀ * exp(-(h - h₀)/H)
    rho = c.ATM_RHO_0 * np.exp(-(altitude - c.ATM_H_0) / c.ATM_SCALE_HEIGHT)  # kg/m³

    # Calculate velocity relative to rotating atmosphere
    # v_rel = v_inertial - ω × r
    # where ω = [0, 0, OMEGA_EARTH] is Earth's rotation vector
    omega_cross_r = np.array([
        -c.OMEGA_EARTH * r_vec[1],
        c.OMEGA_EARTH * r_vec[0],
        0.0
    ])  # km/s

    v_rel = v_vec - omega_cross_r  # km/s
    v_rel_mag = np.linalg.norm(v_rel)

    # Calculate drag acceleration
    # a_drag = -0.5 * Cd * (A/m) * ρ * v_rel * |v_rel|
    # Note: ρ is in kg/m³, need to convert to kg/km³ by multiplying by 10^9
    # a_to_m is in km²/kg
    # Result is in km/s²
    rho_km = rho * 1e9  # Convert kg/m³ to kg/km³

    drag_factor = -0.5 * Cd * a_to_m * rho_km * v_rel_mag
    accel = drag_factor * v_rel

    return accel[0], accel[1], accel[2]


def atmospheric_drag_coesa76(t, y, Cd, a_to_m):
    """Calculates atmospheric drag acceleration using COESA76 atmosphere tables.

    Uses the U.S. Standard Atmosphere 1976 (COESA76) tabulated density values
    with log-linear interpolation. More accurate than exponential model for
    altitudes 100-1000 km.

    Accounts for Earth's rotation by computing velocity relative to the
    rotating atmosphere.

    Reference: NASA-TM-X-74335, "U.S. Standard Atmosphere, 1976"
               Vallado, "Fundamentals of Astrodynamics and Applications",
               4th edition, pg. 551-556.

    Args:
        t (float): Current Time in s (unused, included for consistency).
        y (np.ndarray): Current Orbital State in km and km/s.
        Cd (float): Coefficient of drag (dimensionless).
        a_to_m (float): Area-to-Mass Ratio of the satellite in km^2/kg.

    Returns:
        tuple: (ax, ay, az) components of drag acceleration in km/s^2.
    """
    # Extract position and velocity
    r_vec = y[:3]  # km
    v_vec = y[3:]  # km/s

    # Calculate altitude above Earth's surface
    r = np.linalg.norm(r_vec)
    altitude = r - c.RADIUS_EARTH  # km

    # Calculate atmospheric density using COESA76 tables
    rho = get_density_coesa76(altitude)  # kg/m³

    # Calculate velocity relative to rotating atmosphere
    # v_rel = v_inertial - ω × r
    # where ω = [0, 0, OMEGA_EARTH] is Earth's rotation vector
    omega_cross_r = np.array([
        -c.OMEGA_EARTH * r_vec[1],
        c.OMEGA_EARTH * r_vec[0],
        0.0
    ])  # km/s

    v_rel = v_vec - omega_cross_r  # km/s
    v_rel_mag = np.linalg.norm(v_rel)

    # Calculate drag acceleration
    # a_drag = -0.5 * Cd * (A/m) * ρ * v_rel * |v_rel|
    # Note: ρ is in kg/m³, need to convert to kg/km³ by multiplying by 10^9
    # a_to_m is in km²/kg
    # Result is in km/s²
    rho_km = rho * 1e9  # Convert kg/m³ to kg/km³

    drag_factor = -0.5 * Cd * a_to_m * rho_km * v_rel_mag
    accel = drag_factor * v_rel

    return accel[0], accel[1], accel[2]
