"""Tests for mission segment dataclasses."""
import pytest
from aerisplane.mission.segments import Climb, Cruise, Loiter, Descent, Return, Mission


class TestSegments:
    def test_climb_defaults(self):
        s = Climb(to_altitude=100.0, climb_rate=2.0, velocity=12.0)
        assert s.name == "climb"
        assert s.to_altitude == 100.0

    def test_cruise_defaults(self):
        s = Cruise(distance=5000.0, velocity=15.0)
        assert s.altitude == 100.0

    def test_loiter_defaults(self):
        s = Loiter(duration=600.0, velocity=12.0)
        assert s.altitude == 100.0

    def test_descent_defaults(self):
        s = Descent(to_altitude=0.0)
        assert s.descent_rate == 2.0
        assert s.velocity == 15.0

    def test_return_segment(self):
        s = Return(distance=5000.0, velocity=15.0)
        assert s.name == "return"

    def test_mission_holds_segments(self):
        m = Mission(segments=[
            Climb(to_altitude=100, climb_rate=2.0, velocity=12.0),
            Cruise(distance=5000, velocity=15.0),
        ])
        assert len(m.segments) == 2
        assert m.start_altitude == 0.0
