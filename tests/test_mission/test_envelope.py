"""Tests for flight envelope computation."""
import numpy as np
import pytest
from aerisplane.mission.envelope import compute_envelope, EnvelopeResult


class TestEnvelope:
    def test_returns_envelope_result(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert isinstance(env, EnvelopeResult)

    def test_altitudes_ascending(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert all(env.altitudes[i] <= env.altitudes[i+1]
                   for i in range(len(env.altitudes) - 1))

    def test_stall_speed_increases(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert env.stall_speeds[-1] > env.stall_speeds[0]

    def test_max_roc_decreases(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert env.max_rocs[0] > env.max_rocs[-1]

    def test_report_non_empty(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert len(env.report()) > 100

    def test_sea_level_summary(self, perf_aircraft, perf_weight_result):
        env = compute_envelope(perf_aircraft, perf_weight_result)
        assert env.ld_max > 0
        assert env.endurance_s > 0
        assert env.range_m > 0
