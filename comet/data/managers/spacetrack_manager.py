# python imports
from pathlib import Path
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
import os.path
import httpx
import os


# --------------------------------------------------------------------------------------------------------------------------
class SpaceTrackManager:
    """Class that Processes, Downloads and Updates TLEs through the SpaceTrack API.

    Example Constructions:
        * st = SpaceTrackAPI()
        * st = SpaceTrackAPI(username, password)
    """

    # ----------------------------------------------------------------------------------------------------------------------
    # Class Construction
    # ----------------------------------------------------------------------------------------------------------------------
    def __init__(self, username: str = None, password: str = None):
        """Construction of SpaceTrackAPI interface. Username and Password can be provided to the Class directly, or accessed
        through environmental variables SPACETRACK_USERNAME and SPACETRACK_PASSWORD. To set SpaceTrack Credientials to
        environment:

        For Windows:
            set SPACETRACK_USERNAME=your_username
            set SPACETRACK_PASSWORD=your_password
        For macOS/linux:
            export SPACETRACK_USERNAME=your_username
            export SPACETRACK_PASSWORD=your_password

        Args:
            username (str, optional): SpaceTrack username. Defaults to None.
            password (str, optional): SpaceTrack password. Defaults to None.
        """
        # Spacetrack API Credentials
        self.username = username
        self.password = password
        if self.username is None:
            self.username = os.getenv("SPACETRACK_USERNAME")
        if self.password is None:
            self.password = os.getenv("SPACETRACK_PASSWORD")

        # If still no Username or Password is provided,
        if not self.username or not self.password:
            print(
                "No credentials provided to SpaceTrackAPI() class and no credentials stored in local Environment Variables."
            )
            print(" ")
            exit()

        # Build SpaceTrack Client to interface with
        self._st = SpaceTrackClient(
            self.username, self.password, httpx_client=httpx.Client(verify=False)
        )

        # Keys that can be used in query
        self._comparison_keys = [
            "mean_motion",
            "eccentricity",
            "period",
            "semimajor_axis",
            "perigee",
            "apogee",
            "inclination",
            "ra_of_asc_node",
            "arg_of_pericenter",
            "mean_anomaly",
        ]
        self._assigned_keys = ["ordinal", "epoch", "country"]

    # ----------------------------------------------------------------------------------------------------------------------
    # Class Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def query(
        self,
        filters: str | list[str] = None,
        norad_id: int | list[int] = None,
        format: str = "tle",
        save: bool = False,
        filename: str = None,
        path: Path | str = None,
    ):
        """Method that queries SpaceTrack API to retrieve most recent TLE's that match the provided filters.

        Example Query Strings:
            * '45 <= inclination <= 60'         Inclination between 45 and 60 degrees
            * 'period >= 150'                   Period greater than 150 minutes
            * 'epoch <= 30'                     TLE Epoch within the last 30 days
            * '0.99 <= mean_motion <= 1.01'     Mean Motion between 0.99 and 1.01

        Queryable parameters:
            mean_motion, eccentricity, period, semimajor_axis, perigee, apogee, inclination, ra_of_asc_node,
            arg_of_pericenter, mean_anomaly, ordinal, epoch, country
            https://www.space-track.org/documentation#/legend

        Args:
            filters (str|list[str], optional): List of SpaceTrack REST API query strings. Defaults to None.
            norad_id (int|list[int], optional): List of NORAD IDs to query. Defaults to None.
            format (str, optional): File format of the requested data. Defaults to 'tle'.
                Options: tle, 3le, gp, xml, html, csv, json
            save (bool, optional): Save data. Defaults to False.
            filename (str, optional): Filename of saved data. Defaults to None.
            path (Path|str, optional): Path to saved data. Defaults to comet/data/tle.

        Returns:
            data: Queried data in various formats.
        """
        # Unique string of query settings
        query_str = "query_" + str(norad_id) + "_" + str(filters) + "." + str(format)
        possible_path = Path(__file__).parents[1] / "tle" / "temp" / query_str

        # Load or Query
        if os.path.isfile(possible_path):
            # Query has already been made, load the data
            with open(possible_path, mode="r") as file:
                data = file.read()
        else:
            # Query doesn't exist, query SpaceTrack
            query_dict = self.build_query(filters, norad_id, format)
            data = self._st.generic_request("tle_latest", **query_dict)
            with open(possible_path, mode="w") as file:
                for line in data:
                    file.write(line)

        # Save data explicitly into data/tle folder
        if save:
            # Default save path
            if path is None:
                path = Path(__file__).parents[1] / "tle"

            # Save data
            if not os.path.isfile(path / filename):
                with open(path / filename, mode="w") as file:
                    for line in data:
                        file.write(line)

        return data

    # ----------------------------------------------------------------------------------------------------------------------
    def build_query(self, filters, norad_id, format):
        """Builds dictionary of SpaceTrack processed query to send to SpaceTrack API Client.

        Args:
            filters (str|list[str], optional): List of SpaceTrack REST API query strings. Defaults to None.
            norad_id (int|list[int], optional): List of NORAD IDs to query. Defaults to None.
            format (str, optional): File format of the requested data. Defaults to 'tle'.
                Options: tle, 3le, gp, xml, html, csv, json

        Returns:
            processed_query (dict): Dictionary of SpaceTrack processed query.
        """
        # Initialize Query disctionary
        processed_query = {}

        # Process Conditions in filters
        if filters is not None:
            for condition in filters:
                condition_array = condition.split()
                # Check if valid SpaceTrack Query key
                if bool(set(condition_array) & set(self._comparison_keys)):
                    match len(condition_array):
                        case 5:
                            # Lower and Upper bounds
                            bounds = [float(condition_array[0]), float(condition_array[-1])]
                            key = condition_array[2]
                            st_query = op.inclusive_range(min(bounds), max(bounds))
                        case 3:
                            # Singular Bounds
                            value = float(condition_array[-1])
                            key = condition_array[0]

                            # REST Operators
                            if bool(set(condition_array) & set(["<", "<=", "=<"])):
                                st_query = op.less_than(value)
                            elif bool(set(condition_array) & set([">", ">=", "=>"])):
                                st_query = op.greater_than(value)
                            elif bool(set(condition_array) & set(["=", "=="])):
                                st_query = op.like(value)
                            elif bool(set(condition_array) & set(["!="])):
                                st_query = op.not_equal(value)
                            else:
                                raise ValueError(
                                    "SpaceTrackAPI(): Invalid SpaceTrack REST Operator."
                                )
                        case _:
                            raise ValueError("SpaceTrackAPI(): Filter formatted incorrectly.")
                    # Add SpaceTrack Query into dictionary
                    processed_query[key] = st_query
                elif bool(set(condition_array) & set(self._assigned_keys)):
                    if "ordinal" in condition_array:
                        processed_query["ordinal"] = int(condition_array[-1])
                    elif "epoch" in condition_array:
                        processed_query["epoch"] = ">now-" + condition_array[-1]
                    elif "country" in condition_array:
                        processed_query["country"] = condition_array[-1]
                    else:
                        raise ValueError("SpaceTrackAPI(): Invalid SpaceTrack REST Operator.")
                else:
                    # Key is not in Query keys, skip it
                    continue
        if norad_id is not None:
            processed_query["norad_cat_id"] = norad_id
        processed_query["format"] = format

        return processed_query


# --------------------------------------------------------------------------------------------------------------------------
# Testing
if __name__ == "__main__":
    st = SpaceTrackManager(username="james.johnson@trueanomaly.space", password="ZaQ1XsW2CdE3VfR4_")
    filter = ["0.99 <= mean_motion <= 1.01", "eccentricity <= 0.01", "ordinal = 1", "epoch <= 30"]
    data = st.query(filter, save=True, filename="test.tle")
    print(data)
