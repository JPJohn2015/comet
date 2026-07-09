# python imports
import pytest

# comet imports
from comet.frames.frame import ManeuverFrame, StateFrame


class TestManeuverFrame:
    """Tests for ManeuverFrame enum."""

    def test_maneuver_frame_values(self):
        """Test ManeuverFrame has correct values."""
        assert ManeuverFrame.ECI == "ECI"
        assert ManeuverFrame.RIC == "RIC"

    def test_maneuver_frame_membership(self):
        """Test ManeuverFrame membership."""
        assert "ECI" in ManeuverFrame.__members__.values()
        assert "RIC" in ManeuverFrame.__members__.values()

    def test_maneuver_frame_count(self):
        """Test ManeuverFrame has exactly 2 members."""
        assert len(ManeuverFrame) == 2

    def test_maneuver_frame_string_behavior(self):
        """Test ManeuverFrame string behavior."""
        assert str(ManeuverFrame.ECI) == "ManeuverFrame.ECI"
        assert str(ManeuverFrame.RIC) == "ManeuverFrame.RIC"

    def test_maneuver_frame_equality(self):
        """Test ManeuverFrame equality comparisons."""
        assert ManeuverFrame.ECI == "ECI"
        assert ManeuverFrame.RIC == "RIC"
        assert ManeuverFrame.ECI != ManeuverFrame.RIC


class TestStateFrame:
    """Tests for StateFrame enum."""

    def test_state_frame_values(self):
        """Test StateFrame has correct values."""
        assert StateFrame.ECI == "ECI"
        assert StateFrame.ECEF == "ECEF"
        assert StateFrame.LLA == "LLA"

    def test_state_frame_membership(self):
        """Test StateFrame membership."""
        assert "ECI" in StateFrame.__members__.values()
        assert "ECEF" in StateFrame.__members__.values()
        assert "LLA" in StateFrame.__members__.values()

    def test_state_frame_count(self):
        """Test StateFrame has exactly 3 members."""
        assert len(StateFrame) == 3

    def test_state_frame_string_behavior(self):
        """Test StateFrame string behavior."""
        assert str(StateFrame.ECI) == "StateFrame.ECI"
        assert str(StateFrame.ECEF) == "StateFrame.ECEF"
        assert str(StateFrame.LLA) == "StateFrame.LLA"

    def test_state_frame_equality(self):
        """Test StateFrame equality comparisons."""
        assert StateFrame.ECI == "ECI"
        assert StateFrame.ECEF == "ECEF"
        assert StateFrame.LLA == "LLA"
        assert StateFrame.ECI != StateFrame.ECEF
        assert StateFrame.ECEF != StateFrame.LLA
