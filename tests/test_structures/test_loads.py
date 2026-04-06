# tests/test_structures/test_loads.py
"""Tests for design load factor functions."""
import pytest
import aerisplane as ap
import aerisplane.weights as wts
from aerisplane.structures.loads import (
    maneuver_load_factor,
    gust_load_factor,
    design_load_factor,
)


class TestManeuverLoadFactor:
    def test_default_35g(self):
        assert maneuver_load_factor() == pytest.approx(3.5)

    def test_custom_value(self):
        assert maneuver_load_factor(n_limit=4.0) == pytest.approx(4.0)


class TestGustLoadFactor:
    def test_returns_greater_than_one(self, cruise_condition):
        n = gust_load_factor(
            velocity=cruise_condition.velocity,
            altitude=cruise_condition.altitude,
            cl_alpha_per_rad=5.5,
            wing_loading=60.0,  # 60 Pa — typical RC
        )
        assert n > 1.0

    def test_increases_with_velocity(self):
        n_slow = gust_load_factor(velocity=10.0, altitude=0.0,
                                  cl_alpha_per_rad=5.5, wing_loading=60.0)
        n_fast = gust_load_factor(velocity=20.0, altitude=0.0,
                                  cl_alpha_per_rad=5.5, wing_loading=60.0)
        assert n_fast > n_slow

    def test_decreases_with_wing_loading(self):
        n_light = gust_load_factor(velocity=15.0, altitude=0.0,
                                   cl_alpha_per_rad=5.5, wing_loading=30.0)
        n_heavy = gust_load_factor(velocity=15.0, altitude=0.0,
                                   cl_alpha_per_rad=5.5, wing_loading=100.0)
        assert n_light > n_heavy


class TestDesignLoadFactor:
    def test_at_least_maneuver_times_safety(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        n = design_load_factor(simple_aircraft, cruise_condition, wr,
                               n_limit=3.5, safety_factor=1.5)
        # Must be at least n_limit × safety_factor
        assert n >= 3.5 * 1.5

    def test_increases_with_safety_factor(self, simple_aircraft, cruise_condition):
        wr = wts.analyze(simple_aircraft)
        n1 = design_load_factor(simple_aircraft, cruise_condition, wr,
                                safety_factor=1.5)
        n2 = design_load_factor(simple_aircraft, cruise_condition, wr,
                                safety_factor=2.0)
        assert n2 > n1

    def test_higher_at_high_speed(self, simple_aircraft):
        wr = wts.analyze(simple_aircraft)
        cond_slow = ap.FlightCondition(velocity=10.0, altitude=0.0, alpha=3.0)
        cond_fast = ap.FlightCondition(velocity=25.0, altitude=0.0, alpha=3.0)
        n_slow = design_load_factor(simple_aircraft, cond_slow, wr)
        n_fast = design_load_factor(simple_aircraft, cond_fast, wr)
        assert n_fast >= n_slow
