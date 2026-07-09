# python imports
import numpy as np

# COMET imports
from comet.utilities.constants import Constants as c
from comet.time.epoch import Epoch
from comet.time.duration import Duration
from comet.celestial.celestial_fidelity import CelestialFidelity
from comet.time.timeline import Timeline, TIMELINE


class Moon:
    """Class that represents Lunar properties."""

    def mu(self) -> float:
        """Returns Lunar Gravitational Parameter.

        Returns:
            mu (float): Lunar Gravitational Parameter in km^3/s^2.
        """
        return c.MU_MOON

    def radius(self) -> float:
        """Returns Lunar Radius.

        Returns:
            radius (float): Lunar Radius in km.
        """
        return c.RADIUS_MOON

    def get_position(
        self,
        epoch: Epoch = Epoch(c.J2000),
        fidelity: str | CelestialFidelity = CelestialFidelity.LoFi,
    ) -> np.ndarray:
        """Calculates the Lunar Position in Mean Equator of Date for a specific Epoch.

        Args:
            epoch (Epoch, optional): Epoch to get Lunar Position. Defaults to J2000.
            fidelity (CelestialFidelity, optional): Celestial Body calculation Fidelity. Defaults to CelestialFidelity.LoFi.

        Returns:
            moon_position (np.ndarray): Lunar Position in km.
        """
        # Get Julian Date from Epoch
        jd = epoch.julian_date()

        # Get Lunar Position using specified Fidelity
        if fidelity == CelestialFidelity.LoFi:
            moon_position = self._moon_position_low_fidelity(jd)
        elif fidelity == CelestialFidelity.HiFi:
            raise NotImplemented("Moon(): High Fidelity Lunar Position not Implemented")
        else:
            raise ValueError("Moon(): Invalid CelestialFidelity Enum")

        return moon_position

    def get_position_array(
        self, timeline: Timeline = TIMELINE, fidelity: CelestialFidelity = CelestialFidelity.LoFi
    ) -> np.ndarray:
        """Calculates the Lunar Position array in Mean Equator of Date for a specific timeline.

        Args:
            timeline (Timeline, optional): Timeline over which to get Lunar Position. Defaults to TIMELINE.
            fidelity (CelestialFidelity, optional): Celestial Body calculation Fidelity. Defaults to CelestialFidelity.LoFi.

        Returns:
            moon_position_array (np.ndarray): Nx3 Array of Lunar Position in km.
        """
        # Get Julian Date list for timeline
        jd = timeline.get_julian_date_list()

        # Get Lunar Position using specified Fidelity
        if fidelity == CelestialFidelity.LoFi:
            moon_position_array = self._moon_position_low_fidelity(jd)
        elif fidelity == CelestialFidelity.HiFi:
            raise NotImplemented("Moon(): High Fidelity Lunar Position not Implemented")
        else:
            raise ValueError("Moon(): Invalid CelestialFidelity Enum")

        return moon_position_array

    def _moon_position_low_fidelity(self, julian_dates: np.ndarray) -> np.ndarray:
        """Calculates the Lunar Position for a set of Julian Dates.
        This function uses Vallado's Low Fidelity Moon Position Vector Algorithm 31 on pg. 288

        Args:
            julian_dates (np.ndarray): Array of Julian Dates.

        Returns:
            moon_position (np.ndarray): Nx3 Array of Lunar Positions.
        """
        # Calculate Julian Centuries
        julian_centuries = (julian_dates - c.J2000) / c.JULIAN_CENTURY

        # Calculate Angular quantities
        ecliptic_longitude = np.deg2rad(
            218.32
            + 481267.8813 * julian_centuries
            + 6.29 * np.sin(np.deg2rad(134.9 + 477198.85 * julian_centuries))
            - 1.27 * np.sin(np.deg2rad(259.2 - 413335.38 * julian_centuries))
            + 0.66 * np.sin(np.deg2rad(235.7 + 890534.24 * julian_centuries))
            + 0.21 * np.sin(np.deg2rad(269.9 + 954397.70 * julian_centuries))
            - 0.19 * np.sin(np.deg2rad(357.5 + 35999.05 * julian_centuries))
            - 0.11 * np.sin(np.deg2rad(186.6 + 966404.05 * julian_centuries))
        )
        ecliptic_latitude = np.deg2rad(
            5.13 * np.sin(np.deg2rad(93.3 + 483202.03 * julian_centuries))
            + 0.28 * np.sin(np.deg2rad(228.2 + 960400.87 * julian_centuries))
            - 0.28 * np.sin(np.deg2rad(318.3 + 6003.18 * julian_centuries))
            - 0.17 * np.sin(np.deg2rad(217.6 - 407332.20 * julian_centuries))
        )
        parallax = np.deg2rad(
            0.9508
            + 0.0518 * np.cos(np.deg2rad(134.9 + 477198.85 * julian_centuries))
            + 0.0095 * np.cos(np.deg2rad(259.2 - 413335.38 * julian_centuries))
            + 0.0078 * np.cos(np.deg2rad(235.7 + 890534.23 * julian_centuries))
            + 0.0028 * np.cos(np.deg2rad(269.9 + 954397.70 * julian_centuries))
        )
        obliquity_ecliptic = np.deg2rad(
            23.439291
            - 0.00130042 * julian_centuries
            - (1.64e-7) * julian_centuries**2
            + (5.04e-7) * julian_centuries**3
        )

        radius = c.RADIUS_EARTH / np.sin(parallax)
        moon_position_unit_vector = [
            np.cos(ecliptic_latitude) * np.cos(ecliptic_longitude),
            np.cos(obliquity_ecliptic) * np.cos(ecliptic_latitude) * np.sin(ecliptic_longitude)
            - np.sin(obliquity_ecliptic) * np.sin(ecliptic_latitude),
            np.sin(obliquity_ecliptic) * np.cos(ecliptic_latitude) * np.sin(ecliptic_longitude)
            - np.cos(obliquity_ecliptic) * np.sin(ecliptic_latitude),
        ]

        # Scale to km
        moon_position = np.array(moon_position_unit_vector * radius).T

        return moon_position


# Testing
if __name__ == "__main__":
    moon = Moon()
    moon_state = moon.get_position()
