# python imports
from datetime import datetime

# comet imports
from comet.time.epoch import Epoch
from comet.time.epoch import Duration
from comet.utilities.constants import Constants as c
from comet.time.time_conversions import *

EPOCH_A = Epoch(2000, 1, 1)
EPOCH_B = Epoch(2000, 1, 2)
EPOCH_C = Epoch(2000, 1, 3)
DURATION_A = Duration(days=1)
DURATION_B = Duration(days=2)


class TestEpoch:
    def test_epoch_construction(self):
        # Test Class Constructors
        assert Epoch().julian_date() == c.J2000
        assert Epoch(2000, 1, 1).julian_date() == c.J2000
        assert Epoch(2000, 1, 1, 12, 0, 0).julian_date() == c.J2000
        assert Epoch("2000-01-01T12:00:00.0").julian_date() == c.J2000
        assert Epoch(c.J2000).julian_date() == c.J2000

    def test_epoch_operations(self):
        # Equality and Non-Equality
        assert EPOCH_A == EPOCH_A
        assert EPOCH_B != EPOCH_C

        # LE(ET) and GE(ET)
        assert EPOCH_B < EPOCH_C
        assert EPOCH_A <= EPOCH_A
        assert EPOCH_A <= EPOCH_B
        assert EPOCH_C > EPOCH_B
        assert EPOCH_C >= EPOCH_B
        assert EPOCH_C >= EPOCH_C

        # Addition and Subtraction
        assert EPOCH_A + DURATION_A == EPOCH_B
        assert EPOCH_B - DURATION_A == EPOCH_A
        assert EPOCH_B - EPOCH_A == DURATION_A
        assert EPOCH_C - EPOCH_A == DURATION_B

    def test_epoch_representations(self):
        assert EPOCH_A.__str__() == "01/01/2000 12:00:00.000 UTC"
        assert EPOCH_A.__repr__() == "Epoch(2000, 1, 1, 12, 0, 00.000)"

        random_epoch = Epoch(2024, 2, 26, 17, 38, 52.749)
        assert random_epoch.__str__() == "02/26/2024 17:16:04.749 UTC"
        assert random_epoch.__repr__() == "Epoch(2024, 2, 26, 17, 16, 04.749)"

    def test_epoch_julian_date(self):
        assert EPOCH_A.julian_date() == c.J2000
        assert EPOCH_B.julian_date() == c.J2000 + 1
        assert EPOCH_A.julian_date("UTC") == c.J2000
        assert EPOCH_A.julian_date("TT") == c.J2000 + (c.UTC_TT / c.DAY)
        assert EPOCH_A.julian_date("TAI") == c.J2000 + (c.UTC_TAI / c.DAY)

    def test_epoch_modified_julian_date(self):
        assert EPOCH_A.modified_julian_date() == c.MJ2000
        assert EPOCH_B.modified_julian_date() == c.MJ2000 + 1
        assert EPOCH_A.modified_julian_date("UTC") == c.MJ2000
        assert EPOCH_A.modified_julian_date("TT") == c.MJ2000 + (c.UTC_TT / c.DAY)
        assert EPOCH_A.modified_julian_date("TAI") == c.MJ2000 + (c.UTC_TAI / c.DAY)

    def test_epoch_unix(self):
        assert EPOCH_A.unix() == jd_to_unix(c.J2000)
        assert EPOCH_B.unix() == jd_to_unix(c.J2000) + 86400
        assert EPOCH_A.unix("UTC") == jd_to_unix(c.J2000)
        assert EPOCH_A.unix("TT") == jd_to_unix(c.J2000) + c.UTC_TT
        assert EPOCH_A.unix("TAI") == jd_to_unix(c.J2000) + c.UTC_TAI

    def test_epoch_julian_centuries(self):
        assert EPOCH_A.julian_centuries() == 0.0
        assert EPOCH_B.julian_centuries() == 1 / c.JULIAN_CENTURY
        assert EPOCH_A.julian_centuries("UTC") == 0.0
        assert np.isclose(
            EPOCH_A.julian_centuries("TT"), (c.UTC_TT / c.DAY) / c.JULIAN_CENTURY, 1e-10
        )
        assert np.isclose(
            EPOCH_A.julian_centuries("TAI"), (c.UTC_TAI / c.DAY) / c.JULIAN_CENTURY, 1e-10
        )
