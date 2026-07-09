# python imports
import numpy as np

# COMET imports
from comet.utilities.constants import Constants as c
from comet.time.epoch import Epoch
from comet.celestial.celestial_fidelity import CelestialFidelity
from comet.time.timeline import Timeline, TIMELINE


class Sun:
    """Class that represents Solar properties."""

    def mu(self) -> float:
        """Returns Solar Gravitational Parameter.

        Returns:
            mu (float): Solar Gravitational Parameter in km^3/s^2.
        """
        return c.MU_SUN

    def radius(self) -> float:
        """Returns Solar Radius.

        Returns:
            radius (float): Solar Radius in km.
        """
        return c.RADIUS_SUN

    def get_position(
        self, epoch: Epoch = Epoch(c.J2000), fidelity: CelestialFidelity = CelestialFidelity.LoFi
    ) -> np.ndarray:
        """Calculates the Solar Position in Mean Equator of Date for a specific Epoch.

        Args:
            epoch (Epoch, optional): Epoch to get Solar Position. Defaults to J2000.
            fidelity (CelestialFidelity, optional): Celestial Body calculation Fidelity. Defaults to CelestialFidelity.LoFi.

        Returns:
            sun_position (np.ndarray): Solar Position in km.
        """
        # Get Julian Date from Epoch
        jd = epoch.julian_date()

        # Get Solar Position using specified Fidelity
        if fidelity == CelestialFidelity.LoFi:
            sun_position = self._sun_position_low_fidelity(jd)
        elif fidelity == CelestialFidelity.HiFi:
            raise NotImplemented("Sun(): High Fidelity Solar Position not Implemented")
        else:
            raise ValueError("Sun(): Invalid CelestialFidelity Enum")

        return sun_position

    def get_position_array(
        self, timeline: Timeline = TIMELINE, fidelity: CelestialFidelity = CelestialFidelity.LoFi
    ) -> np.ndarray:
        """Calculates the Solar Position array in Mean Equator of Date for a specific timeline.

        Args:
            timeline (Timeline, optional): Timeline over which to get Solar Position. Defaults to TIMELINE.
            fidelity (CelestialFidelity, optional): Celestial Body calculation Fidelity. Defaults to CelestialFidelity.LoFi.

        Returns:
            sun_position_array (np.ndarray): Nx3 Array of Solar Position in km.
        """
        # Get Julian Date list for timeline
        jd = timeline.get_julian_date_list()

        # Get Solar Position using specified Fidelity
        if fidelity == CelestialFidelity.LoFi:
            sun_position_array = self._sun_position_low_fidelity(jd)
        elif fidelity == CelestialFidelity.HiFi:
            raise NotImplemented("Sun(): High Fidelity Solar Position not Implemented")
        else:
            raise ValueError("Sun(): Invalid CelestialFidelity Enum")

        return sun_position_array

    def _sun_position_low_fidelity(self, julian_dates: np.ndarray) -> np.ndarray:
        """Calculates the Solar Position for a set of Julian Dates.
        This function uses Vallado's Low Fidelity Sun Position Vector Algorithm 29 on pg. 279-280

        Args:
            julian_dates (np.ndarray): Array of Julian Dates.

        Returns:
            sun_position (np.ndarray): Nx3 Array of Solar Positions.
        """
        # Calculate Julian Centuries
        julian_centuries = (julian_dates - c.J2000) / c.JULIAN_CENTURY

        # Calculate Angular quantities
        mean_longitude = 280.460 + 36000.771 * julian_centuries
        mean_anomaly = 357.5291092 + 35999.050 * julian_centuries
        ecliptic_longitude = (
            mean_longitude
            + 1.91466471 * np.sin(np.deg2rad(mean_anomaly))
            + 0.019994643 * np.sin(np.deg2rad(2 * mean_anomaly))
        )
        obliquity_ecliptic = 23.439291 - 0.0130042 * julian_centuries

        # Calculate Solar position in AU
        radius = (
            1.000140612
            - 0.016708617 * np.cos(np.deg2rad(mean_anomaly))
            - 0.000139589 * np.cos(np.deg2rad(2 * mean_anomaly))
        )
        sun_position_au = [
            radius * np.cos(np.deg2rad(ecliptic_longitude)),
            radius
            * np.cos(np.deg2rad(obliquity_ecliptic))
            * np.sin(np.deg2rad(ecliptic_longitude)),
            radius
            * np.sin(np.deg2rad(obliquity_ecliptic))
            * np.sin(np.deg2rad(ecliptic_longitude)),
        ]

        # Scale to km
        sun_position = np.array(sun_position_au).T * c.AU

        return sun_position


# Testing
if __name__ == "__main__":
    sun = Sun()
    sun_state = sun.get_position()
