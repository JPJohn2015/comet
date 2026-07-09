# python imports
import numpy as np

# COMET imports
from comet.utilities.constants import Constants as c
from comet.components.component import Component
from comet.time.duration import Duration

# ---------------------------------------------------------------------------------------------------------------------------
class Thruster(Component):
    """Class that contains Thruster properties.

    Example Constructions:
        * thruster = Thruster(epoch, dv)
    """
    # ----------------------------------------------------------------------------------------------------------------------
    # Class Construction
    # ----------------------------------------------------------------------------------------------------------------------
    def __init__(self, thrust: float = 10.0, isp: float = 200.0, mass: float = 0.0, body_vector: np.ndarray = [1, 0, 0]):
        """Construct Thruster Component.

        Args:
            thrust (float): Thrust in N. Defaults to 10.0.
            isp (float): Specific Impulse in sec. Defaults to 200.0.
            mass (float, optional): Component mass in kg. Defaults to 0.0.
            body_vector (np.ndarray, optional): Pointing Vector in Body Frame. Defaults to [1, 0, 0].
            id (int, optional): Component ID. Defaults to Next Incremented ID.
        """
        # Initialize Parent Class
        super().__init__(mass, body_vector)

        # Assign Class attributes
        self.thrust = thrust
        self.isp = isp

    # ----------------------------------------------------------------------------------------------------------------------
    # Class Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def copy(self):
        """Returns a Copy of the Thruster. NOTE: Component ID will still increment.

        Returns:
            thruster (Thruster): Copy of Thruster Component.
        """
        return Thruster(self.thrust, self.isp, self.mass, self.body_vector)
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_thrust(self):
        """Returns the Thrust of the Thruster in N.

        Returns:
            thrust (float): Thrust in N.
        """
        return self.thrust
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_isp(self):
        """Returns the Specific Impulse of the Thruster in sec.

        Returns:
            isp (float): Specific Impulse in sec.
        """
        return self.isp
    
    # ----------------------------------------------------------------------------------------------------------------------
    def mass_flow_rate(self):
        """Returns the mass flow rate of the Thruster in kg/s. Value is negative to show mass is being lost.

        Returns:
            dmdt (float): Mass Flow Rate in kg/s.
        """
        return -self.thrust/(self.isp*c.SURFACE_GRAVITY)
    
    # ----------------------------------------------------------------------------------------------------------------------
    def acceleration(self, mass: float|int):
        """Returns the total acceleration for a specific total system mass.

        Args:
            mass (float|int): Total system mass in kg.

        Returns:
            accel (float): Acceleration from Thruster in m/s.
        """
        return self.thrust/mass
    
    # ----------------------------------------------------------------------------------------------------------------------
    def burn_given_dv(self, mass: float|int, dv: float|int|list|np.ndarray):
        """Calculates the expected burn time for a given Delta-V in km/s.

        Args:
            mass (float|int): Total system mass in kg.
            dv (float|int|list|np.ndarray): Delta-V vector or magnitude in km/s.

        Returns:
            burn_time (float): Burn time required to achieve Delta-V in sec.
            mass_expelled (float): Propellant mass expended in kg. Value is negative to show mass is being lost.
        """
        # Process Delta-V before calculations
        if isinstance(dv, list|np.ndarray):
            dv = np.linalg.norm(dv)
        dv = dv*1000 #km/s -> m/s
        
        # Calculate Thruster properties, then calculate burn time
        accel = self.acceleration(mass)
        dmdt = self.mass_flow_rate()

        # Calculate burn time and propellant expelled
        burn_time = (1/-dmdt)*(1 - np.exp((dmdt*dv)/accel))
        mass_expelled = dmdt*burn_time

        return burn_time, mass_expelled
    
    # ----------------------------------------------------------------------------------------------------------------------
    def burn_given_time(self, mass: float|int, duration: Duration|float):
        """Calculates the expected Delta-V for a given Burn duration in sec.

        Args:
            mass (float|int): Total system mass in kg.
            burn_time (float): Burn time in sec.
            
        Returns:
            dv (float|int|list|np.ndarray): Expected Delta-V magnitude in km/s.
            mass_expelled (float): Propellant mass expended in kg. Value is negative to show mass is being lost.
        """
        # Process duration before calculations
        if isinstance(duration, Duration):
            duration = duration.total_seconds()
        
        # Calculate Thruster properties, then calculate burn time
        accel = self.acceleration(mass)
        dmdt = self.mass_flow_rate()

        # Calculate burn time and propellant expelled
        dv = (accel/dmdt)*np.log(1 + dmdt*duration)
        dv = dv/1000 # m/s -> km/s
        mass_expelled = dmdt*duration

        return dv, mass_expelled


# Testing
if __name__ == "__main__":
    thruster = Thruster(10, 200)
