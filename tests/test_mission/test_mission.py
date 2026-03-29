"""Integration tests for mission analyze()."""
import pytest
from aerisplane.mission import analyze
from aerisplane.mission.segments import Mission, Climb, Cruise, Loiter, Descent
from aerisplane.mission.result import MissionResult, SegmentResult


class TestMissionAnalyze:
    def test_returns_mission_result(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Climb(to_altitude=100, climb_rate=2.0, velocity=12.0),
            Cruise(distance=3000, velocity=15.0, altitude=100.0),
            Descent(to_altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert isinstance(result, MissionResult)

    def test_total_energy_positive(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=3000, velocity=15.0, altitude=100.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert result.total_energy > 0

    def test_feasible_short_mission(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=1000, velocity=15.0, altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert result.feasible

    def test_segment_count_matches(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Climb(to_altitude=100, climb_rate=2.0, velocity=12.0),
            Cruise(distance=3000, velocity=15.0, altitude=100.0),
            Descent(to_altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert len(result.segments) == 3

    def test_report_non_empty(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=3000, velocity=15.0, altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert len(result.report()) > 100

    def test_energy_margin_between_0_and_1(self, perf_aircraft, perf_weight_result):
        mission = Mission(segments=[
            Cruise(distance=1000, velocity=15.0, altitude=0.0),
        ])
        result = analyze(perf_aircraft, perf_weight_result, mission)
        assert 0 <= result.energy_margin <= 1.0
