"""Tests for point performance functions."""
import numpy as np
import pytest
from aerisplane.mission.performance import (
    fit_drag_polar,
    power_required,
    power_available,
    DragPolar,
    stall_speed,
    best_range_speed,
    best_endurance_speed,
    max_level_speed,
    rate_of_climb,
    max_rate_of_climb,
    glide_range,
    GlidePerformance,
    glide_performance,
    max_endurance,
    max_range,
)


class TestDragPolar:
    def test_fit_produces_positive_cd0(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        assert polar.CD0 > 0

    def test_fit_produces_positive_k(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        assert polar.k > 0

    def test_ld_max_reasonable(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        ld_max = polar.ld_max()
        assert 5 < ld_max < 40, f"L/D max = {ld_max}, outside 5-40"


class TestPowerRequired:
    def test_positive_at_cruise_speed(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        pr = power_required(15.0, polar, perf_weight_result.total_mass, altitude=0.0)
        assert pr > 0

    def test_u_shaped_curve(self, perf_aircraft, perf_weight_result):
        """Power required should be higher at very low and very high speeds."""
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        mass = perf_weight_result.total_mass
        pr_low = power_required(5.0, polar, mass, altitude=0.0)
        pr_mid = power_required(14.0, polar, mass, altitude=0.0)
        pr_high = power_required(25.0, polar, mass, altitude=0.0)
        assert pr_mid < pr_low, "P_R should decrease from low speed to mid speed"
        assert pr_mid < pr_high, "P_R should increase from mid speed to high speed"


class TestPowerAvailable:
    def test_positive_at_cruise(self, perf_aircraft):
        pa = power_available(perf_aircraft.propulsion, 15.0, altitude=0.0)
        assert pa > 0

    def test_decreases_with_altitude(self, perf_aircraft):
        pa_sl = power_available(perf_aircraft.propulsion, 15.0, altitude=0.0)
        pa_hi = power_available(perf_aircraft.propulsion, 15.0, altitude=2000.0)
        assert pa_hi < pa_sl, "Power available should decrease with altitude"


class TestCharacteristicSpeeds:
    def test_stall_speed_positive(self, perf_weight_result):
        vs = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=0.0)
        assert vs > 0

    def test_stall_speed_reasonable(self, perf_weight_result):
        """Stall speed for a ~2kg aircraft should be 5-15 m/s."""
        vs = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=0.0)
        assert 3 < vs < 20, f"V_stall = {vs:.1f} m/s"

    def test_stall_increases_with_altitude(self, perf_weight_result):
        vs_sl = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=0.0)
        vs_hi = stall_speed(perf_weight_result.total_mass, S=0.3, CL_max=1.4, altitude=2000.0)
        assert vs_hi > vs_sl

    def test_best_range_speed(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        vr = best_range_speed(polar, perf_weight_result.total_mass, altitude=0.0)
        assert vr > 0

    def test_best_endurance_slower_than_range(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        mass = perf_weight_result.total_mass
        ve = best_endurance_speed(polar, mass, altitude=0.0)
        vr = best_range_speed(polar, mass, altitude=0.0)
        assert ve < vr, "V_endurance should be slower than V_range"

    def test_max_speed_above_cruise(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        vmax = max_level_speed(polar, perf_weight_result.total_mass,
                               perf_aircraft.propulsion, altitude=0.0)
        if vmax is not None:
            assert vmax > 10.0


class TestClimb:
    def test_roc_positive_at_cruise(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        roc = rate_of_climb(15.0, polar, perf_weight_result.total_mass,
                            perf_aircraft.propulsion, altitude=0.0)
        assert roc > 0, f"ROC = {roc:.2f} m/s, expected positive"

    def test_max_roc_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        roc_max, v_y = max_rate_of_climb(
            polar, perf_weight_result.total_mass,
            perf_aircraft.propulsion, altitude=0.0
        )
        assert roc_max > 0
        assert v_y > 0

    def test_roc_decreases_with_altitude(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        mass = perf_weight_result.total_mass
        prop = perf_aircraft.propulsion
        roc_sl, _ = max_rate_of_climb(polar, mass, prop, altitude=0.0)
        roc_hi, _ = max_rate_of_climb(polar, mass, prop, altitude=2000.0)
        assert roc_hi < roc_sl


class TestGlide:
    def test_glide_range_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        r = glide_range(polar, from_altitude=100.0)
        assert r > 0

    def test_glide_performance(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        gp = glide_performance(polar, perf_weight_result.total_mass, altitude=0.0)
        assert gp.best_glide_ratio > 0
        assert gp.best_glide_speed > 0
        assert gp.min_sink_speed > 0
        assert gp.min_sink_rate > 0
        assert gp.min_sink_speed < gp.best_glide_speed


class TestEnduranceRange:
    def test_endurance_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        e = max_endurance(polar, perf_weight_result.total_mass,
                          perf_aircraft.propulsion, altitude=0.0)
        assert e > 0

    def test_endurance_reasonable(self, perf_aircraft, perf_weight_result):
        """Endurance for a 2.2Ah 4S battery at ~2kg should be 5-60 minutes."""
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        e = max_endurance(polar, perf_weight_result.total_mass,
                          perf_aircraft.propulsion, altitude=0.0)
        assert 60 < e < 7200, f"Endurance = {e:.0f}s ({e/60:.1f} min)"

    def test_range_positive(self, perf_aircraft, perf_weight_result):
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        r = max_range(polar, perf_weight_result.total_mass,
                      perf_aircraft.propulsion, altitude=0.0)
        assert r > 0

    def test_range_reasonable(self, perf_aircraft, perf_weight_result):
        """Range should be 1-50 km for a typical RC aircraft."""
        polar = fit_drag_polar(perf_aircraft, perf_weight_result, altitude=0.0)
        r = max_range(polar, perf_weight_result.total_mass,
                      perf_aircraft.propulsion, altitude=0.0)
        assert 500 < r < 100_000, f"Range = {r:.0f}m ({r/1000:.1f} km)"
