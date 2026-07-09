# python imports
import numpy as np
from pathlib import Path
import requests
import time

# comet imports
from comet.utilities.constants import Constants as c
from comet.time.epoch import Epoch
from comet.utilities.rotations import rot3


def read_eop_data_file(file_path, file_type):
    """Loads Earth Orientation Parameter data from files into a dictionary.

    Args:
        file_path (Path|str): File Path to datafile.
        file_type (str): Data File type.

    Returns:
        data (dict): Earth Orientation Parameters Data.
    """
    data = {}
    if file_type == "c04":
        # Parse c04 Format
        with open(file_path) as input_file:
            for _ in range(14):
                input_file.readline()

            # Read each line and append to data dictionary
            for line in input_file:
                dat = tuple(line.strip().split())
                mjd_utc = int(dat[3])
                ut1_utc = float(dat[6])
                xp = float(dat[4]) * c.AS2RAD
                yp = float(dat[5]) * c.AS2RAD
                data.update({mjd_utc: (ut1_utc, xp, yp)})
    elif file_type == "2000ab":
        # Parse 2000AB Format
        with open(file_path) as input_file:
            for line in input_file:
                # Read each line and append to data dictionary
                if line[16:17] in ["P", "I"]:
                    mjd_utc = int(line[7:12])
                    ut1_utc = float(line[58:68])
                    xp = float(line[18:27]) * c.AS2RAD
                    yp = float(line[37:46]) * c.AS2RAD
                    data.update({mjd_utc: (ut1_utc, xp, yp)})
    else:
        raise Exception("read_data_file(): Invalid datafile type [c04, 2000ab]")

    return data


def read_iers_data_file(file_path):
    """Loads IERS data from files into a list.

    Args:
        file_path (Path|str): File Path to datafile.
        file_type (str): Data File type.

    Returns:
        data (list): IERS Data.
    """
    data = []
    with open(file_path) as input_file:
        for i in range(0, 5):
            if i == 0:
                # First time, skip 37 lines
                for _ in range(37):
                    input_file.readline()
            else:
                # Every other time, skip 2 lines
                for _ in range(2):
                    input_file.readline()

            table_data = []
            # Read each line and append to data dictionary
            for line in input_file:
                dat = line.strip().split()
                if dat == []:
                    break
                table_data.append(np.array(dat[1:]).astype(float))

            # Append to larger array
            data.append(table_data)

    return data


class EarthOrientationData:
    """Class that stores Earth Orientation Parameters."""

    # Initialize EarthOrientationData attributes
    _is_loaded = False
    _interpolate = False
    _data = {}

    def load(self, file_path, file_type, interpolate=False):
        """Loads in Earth Orientation Parameters data from file into Class.

        Args:
            file_path (Path|str): File Path to datafile.
            file_type (str): Data File type.
            interpolate (bool, optional): Perform sub-daily interpolation of data. Defaults to False.
        """
        # Reset any currently loaded data
        self._data.clear()
        self._interpolate = interpolate

        # Load data file
        self._data = read_eop_data_file(file_path, file_type)
        self._is_loaded = True

    def clear(self):
        """Clears Stored Earth Orientation Parameters data."""
        # Clear Data
        self._is_loaded = False
        self._data.clear()

    def set(self, mjd_utc, ut1_utc, xp, yp):
        """Sets specific values for Earth Orientation Parameters data.

        Args:
            mjd_utc (float): Modified Julian Date of Epoch in UTC.
            ut1_utc (float): UT1 - UTC offset in sec.
            xp (float): x-pole component in arcsec.
            yp (float): y-pole component in arcsec.
        """
        # Set X and Y Pole components
        xp = xp * c.AS2RAD
        yp = yp * c.AS2RAD
        self._data.update({np.floor(mjd_utc): (ut1_utc, xp, yp)})
        self._is_loaded = True

    def download(self):
        """Downloads a new IERS IAU2000 datafile if over 1 day out of date."""
        # Print download is in progress
        print("IERS IAU2000 datafile out of date, downloading new version.")

        # Get URL response
        url = "https://datacenter.iers.org/data/latestVersion/finals.all.iau2000.txt"
        response = requests.get(url)

        # Check if valid response
        if (not response.ok) or (response.status_code != 200):
            raise LookupError("EarthOrientationData(): Error downloading new IERS IAU2000 datafile")

        # Write datafile
        with open("comet/data/iau/iau2000A_finals_ab.txt", mode="wb") as file:
            file.write(response.content)

        # Print download completed
        print("IERS IAU2000 downloaded successfully.")

    def eop(self, mjd):
        """Gets Earth Orientation Parameter data for a specified modified julian data.

        Args:
            mjd (float): Modified Julian Date of Epoch in UTC.

        Returns:
            eop (tuple): Earth Orientation Data Paramters at epoch:
                ut1_utc (float): UT1 - UTC offset in sec.
                xp (float): x-pole component in arcsec.
                yp (float): y-pole component in arcsec.
        """
        if not self._is_loaded:
            # Load Default EOP Data if none has been loaded
            default_path = Path(__file__).parents[1] / "data" / "iau" / "iau2000A_finals_ab.txt"
            last_modified = default_path.stat().st_mtime / c.DAY
            current = time.time() / c.DAY
            if current - last_modified > 1.0:
                # Download new file
                self.download()
            self.load(file_path=default_path, file_type="2000ab", interpolate=False)

        # Get data, interpolating if necessary
        if self._interpolate:
            x1 = int(np.floor(mjd))
            y1 = self._data[x1]
            x2 = x1 + 1
            y2 = self._data[x2]
            x = mjd

            # Element-wise linear interpolation of tuples
            dat = tuple((yb - ya) / (x2 - x1) * (x - x1) + ya for ya, yb in zip(y1, y2))
        else:
            try:
                dat = self._data[int(np.floor(mjd))]
            except:
                # MJD is out of bounds, determine if above or below to use closest EOP data
                if mjd < 41684:
                    index = min(self._data)
                else:
                    index = max(self._data)
                dat = self._data[index]

        return dat

    def ut1_to_utc(self, mjd):
        """Returns the offset in seconds between UT1 and UTC.

        Args:
            mjd (float): Modified Julian Date of Epoch in UTC.

        Returns:
            ut1_utc (float): UT1 - UTC offset in sec.
        """
        if not self._is_loaded:
            # Load Default EOP Data if none has been loaded
            default_path = Path(__file__).parents[1] / "data" / "iau" / "iau2000A_finals_ab.txt"
            last_modified = default_path.stat().st_mtime / c.DAY
            current = time.time() / c.DAY
            if current - last_modified > 1.0:
                # Download new file
                self.download()
            self.load(file_path=default_path, file_type="2000ab", interpolate=False)

        # Get data, interpolating if necessary
        if self._interpolate:
            x1 = int(np.floor(mjd))
            y1 = self._data[x1][0]
            x2 = x1 + 1
            y2 = self._data[x2][0]
            x = mjd

            # Element-wise linear interpolation of tuples
            ut1_utc = (y2 - y1) / (x2 - x1) * (x - x1) + y1
        else:
            ut1_utc = self._data[int(np.floor(mjd))][0]

        return ut1_utc

    def utc_to_ut1(self, mjd):
        """Returns the offset in seconds between UTC and UT1.

        Args:
            mjd (float): Modified Julian Date of Epoch in UTC.

        Returns:
            utc_ut1 (float): UTC - UT1 offset in sec.
        """
        return -float(self.ut1_to_utc(mjd))


class IERSData:
    """Class that stores IERS Tabulated Data."""

    # Initialize IERSData attributes
    _table_2a = []
    _table_2b = []
    _table_2d = []
    _is_loaded = False

    def load(self, file_path, file_type):
        """Loads in IERS data from file into Class.

        Args:
            file_path (Path|str): File Path to datafile.
            file_type (str): Data File type.
            interpolate (bool, optional): Perform sub-daily interpolation of data. Defaults to False.
        """
        match file_type:
            case "tab5_2a":
                # Reset any currently loaded data and load new datafile
                self._table_2a = {}
                self._table_2a = read_iers_data_file(file_path)
            case "tab5_2b":
                # Reset any currently loaded data and load new datafile
                self._table_2b = {}
                self._table_2b = read_iers_data_file(file_path)
            case "tab5_2d":
                # Reset any currently loaded data and load new datafile
                self._table_2d = {}
                self._table_2d = read_iers_data_file(file_path)
            case _:
                raise ValueError("IERSData(): Invalid table type")

    def get_XYs(self, angles, Ttt):
        """Gets IERS data.

        Args:
            mjd (float): Modified Julian Date of Epoch in UTC.

        Returns:
            eop (tuple): Earth Orientation Data Paramters at epoch:
                ut1_utc (float): UT1 - UTC offset in sec.
                xp (float): x-pole component in arcsec.
                yp (float): y-pole component in arcsec.
        """
        if not self._is_loaded:
            # Load Default EOP Data if none has been loaded
            default_path = Path(__file__).parents[1] / "data" / "iau"
            self.load(file_path=default_path / "tab5_2a.txt", file_type="tab5_2a")
            self.load(file_path=default_path / "tab5_2b.txt", file_type="tab5_2b")
            self.load(file_path=default_path / "tab5_2d.txt", file_type="tab5_2d")
            self._is_loaded = True

        # Calculate X
        x = (
            -0.016617
            + 2004.191898 * Ttt
            - 0.4297828 * (Ttt**2)
            - 0.19861834 * (Ttt**3)
            + 7.578e-6 * (Ttt**4)
            + 5.9285e-6 * (Ttt**5)
        )
        for i, summation in enumerate(self._table_2a):
            summation = np.array(summation)
            Api = np.matmul(summation[:, 2:], angles)
            x += (
                np.sum(summation[:, 0] * np.sin(Api) + summation[:, 1] * np.cos(Api))
                * (Ttt ** (i))
                * (1e-6)
            )

        # Calculate Y
        y = (
            -0.006951
            - 0.025896 * Ttt
            - 22.4072747 * (Ttt**2)
            + 0.00190059 * (Ttt**3)
            + 0.001112526 * (Ttt**4)
            + 1.358e-7 * (Ttt**5)
        )
        for i, summation in enumerate(self._table_2b):
            summation = np.array(summation)
            Api = np.matmul(summation[:, 2:], angles)
            y += (
                np.sum(summation[:, 0] * np.sin(Api) + summation[:, 1] * np.cos(Api))
                * (Ttt ** (i))
                * (1e-6)
            )

        # Calculate s
        s = (
            -(x * y) / 2
            + 9.4e-5
            + 3.80865e-3 * Ttt
            - 1.2268e-4 * (Ttt**2)
            - 0.07257411 * (Ttt**3)
            + 2.798e-5 * (Ttt**4)
            + 1.562e-5 * (Ttt**5)
        )
        for i, summation in enumerate(self._table_2d):
            summation = np.array(summation)
            Api = np.matmul(summation[:, 2:], angles)
            s += (
                np.sum(summation[:, 0] * np.sin(Api) + summation[:, 1] * np.cos(Api))
                * (Ttt ** (i))
                * (1e-6)
            )

        return (x * c.AS2RAD, y * c.AS2RAD, s * c.AS2RAD)


# Define Singleton instances of Data processing classes
EOP_DATA = EarthOrientationData()
IERS_DATA = IERSData()


def precession_nutation(epoch: Epoch):
    """Calculates the Precession Nutation Rotation Matrix for the IAU2000 reduction.

    Accepts both single Epoch and array of Epochs for vectorized computation.

    Args:
        epoch (Epoch | np.ndarray): Epoch or array of Epochs.

    Returns:
        rot_pn (np.ndarray): 3x3 Precession Nutation Rotation Matrix (or Nx3x3 for arrays).
    """
    # Handle both single Epoch and array of Epochs
    epoch_array = np.atleast_1d(epoch)
    is_scalar = epoch_array.shape == (1,) and not isinstance(epoch, np.ndarray)

    # Preallocate output
    rot_pn = np.zeros((len(epoch_array), 3, 3))

    for idx, ep in enumerate(epoch_array):
        # Get Julian Centuries in TT
        Ttt = ep.julian_centuries("TT")
        Ttt2, Ttt3, Ttt4 = Ttt**2, Ttt**3, Ttt**4

        # Calculate Earth Nutation Angles
        Mms = (
            485868.249036 + 1717915923.2178 * Ttt + 31.8792 * Ttt2 + 0.051635 * Ttt3 - 0.00024470 * Ttt4
        ) * c.AS2DEG
        Ms = (
            1287104.79305 + 129596581.0481 * Ttt - 0.5532 * Ttt2 + 0.000136 * Ttt3 - 0.00001149 * Ttt4
        ) * c.AS2DEG
        uMm = (
            335779.526232 + 1739527262.8478 * Ttt - 12.7512 * Ttt2 - 0.001037 * Ttt3 + 0.00000417 * Ttt4
        ) * c.AS2DEG
        De = (
            1072260.70369 + 1602961601.2090 * Ttt - 6.3706 * Ttt2 + 0.006593 * Ttt3 - 0.00003169 * Ttt4
        ) * c.AS2DEG
        Omega = (
            450160.398036 - 6962890.5431 * Ttt + 7.4722 * Ttt2 + 0.007702 * Ttt3 - 0.00005939 * Ttt4
        ) * c.AS2DEG

        # Calculate Planetary Nutation Angles
        L_Me = 252.250905494 + 149472.6746358 * Ttt
        L_Ve = 181.979800852 + 58517.8156748 * Ttt
        L_Ea = 100.466448494 + 35999.3728521 * Ttt
        L_Ma = 355.433274605 + 19140.299314 * Ttt
        L_Ju = 34.351483900 + 3034.90567464 * Ttt
        L_Sa = 50.0774713998 + 1222.11379404 * Ttt
        L_Ur = 314.055005137 + 428.466998313 * Ttt
        L_Ne = 304.348665499 + 218.486200208 * Ttt
        pL = 1.39697137214 * Ttt + 0.0003086 * Ttt2

        # Package angles to calculate X, Y and s from IERS Data
        angles = np.deg2rad([Mms, Ms, uMm, De, Omega, L_Me, L_Ve, L_Ea, L_Ma, L_Ju, L_Sa, L_Ur, L_Ne, pL]) % (
            2 * np.pi
        )
        x, y, s = IERS_DATA.get_XYs(angles, Ttt)
        a = 0.5 + (x**2 + y**2) / 8

        # Compute Rotation Matrix for Precession and Nutation
        rot_pn[idx] = np.matmul(
            np.array(
                [
                    [1 - a * x**2, -a * x * y, x],
                    [-a * x * y, 1 - a * y**2, y],
                    [-x, -y, 1 - a * (x**2 + y**2)],
                ]
            ),
            rot3(s),
        )

    return rot_pn[0] if is_scalar else rot_pn


def earth_rotation(epoch: Epoch):
    """Calculates the Earth Rotation Matrix for the IAU2000 reduction.

    Accepts both single Epoch and array of Epochs for vectorized computation.

    Args:
        epoch (Epoch | np.ndarray): Epoch or array of Epochs.

    Returns:
        rot_era (np.ndarray): 3x3 Earth Rotation Matrix (or Nx3x3 for arrays).
    """
    # Handle both single Epoch and array of Epochs
    epoch_array = np.atleast_1d(epoch)
    is_scalar = epoch_array.shape == (1,) and not isinstance(epoch, np.ndarray)

    # Preallocate output
    rot_era = np.zeros((len(epoch_array), 3, 3))

    for idx, ep in enumerate(epoch_array):
        # Calculate UT1
        jd_utc = ep.julian_date("UTC")
        utc_to_ut1, _, _ = EOP_DATA.eop(ep.modified_julian_date("UTC"))
        jd_ut1 = jd_utc + utc_to_ut1 / c.DAY

        # Compute Earth Rotation Angle
        era = 280.46061837504 + 360.985612288808 * (jd_ut1 - c.J2000)
        era = era % 360

        # Compute Rotation Matrix for Earth Rotation
        rot_era[idx] = rot3(-np.deg2rad(era))

    return rot_era[0] if is_scalar else rot_era


def polar_motion(epoch: Epoch):
    """Calculates the Polar Motion Rotation Matrix for the IAU2000 reduction.

    Accepts both single Epoch and array of Epochs for vectorized computation.

    Args:
        epoch (Epoch | np.ndarray): Epoch or array of Epochs.

    Returns:
        rot_pm (np.ndarray): 3x3 Polar Motion Rotation Matrix (or Nx3x3 for arrays).
    """
    # Handle both single Epoch and array of Epochs
    epoch_array = np.atleast_1d(epoch)
    is_scalar = epoch_array.shape == (1,) and not isinstance(epoch, np.ndarray)

    # Preallocate output
    rot_pm = np.zeros((len(epoch_array), 3, 3))

    for idx, ep in enumerate(epoch_array):
        # Get Earth Orientation Data
        _, xp, yp = EOP_DATA.eop(ep.modified_julian_date("UTC"))
        tt = ep.julian_centuries("TT")
        sp = -0.000047 * tt * c.AS2RAD

        # Precompute trig functions
        cx, sx = np.cos(xp), np.sin(xp)
        cy, sy = np.cos(yp), np.sin(yp)
        cs, ss = np.cos(sp), np.sin(sp)

        # Compute Rotation Matrix for Polar Motion
        rot_pm[idx] = np.array(
            [
                [cx * cs, -cy * ss + sy * sx * cs, -sy * ss - cy * sx * cs],
                [cx * ss, cy * cs + sy * sx * ss, sy * cs - cy * sx * ss],
                [sx, -sy * cx, cy * cx],
            ]
        )

    return rot_pm[0] if is_scalar else rot_pm
