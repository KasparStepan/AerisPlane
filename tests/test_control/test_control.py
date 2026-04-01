"""Integration tests for control.analyze()."""

import math

import numpy as np
import pytest

from aerisplane.control import analyze


class TestAnalyze:
    """Test the top-level control analyze() function."""

    def test_returns_control_result(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        from aerisplane.control.result import ControlResult

        result = analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        assert isinstance(result, ControlResult)

    def test_report_non_empty(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        result = analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        report = result.report()
        assert len(report) > 100
        assert "Roll" in report or "roll" in report

    def test_roll_rate_positive(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        result = analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        assert result.max_roll_rate > 0, (
            f"max_roll_rate = {result.max_roll_rate}, expected > 0"
        )

    def test_pitch_acceleration_positive(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        result = analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        assert result.max_pitch_acceleration > 0, (
            f"max_pitch_acceleration = {result.max_pitch_acceleration}, expected > 0"
        )

    def test_authority_in_range(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        result = analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        assert 0 <= result.aileron_authority <= 1.0
        assert 0 <= result.elevator_authority <= 1.0
        assert 0 <= result.rudder_authority <= 1.0

    def test_servo_margins_positive(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        result = analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        for name, margin in [
            ("aileron", result.aileron_servo_margin),
            ("elevator", result.elevator_servo_margin),
            ("rudder", result.rudder_servo_margin),
        ]:
            if margin is not None:
                assert margin > 0, f"{name} servo margin = {margin}, expected > 0"

    def test_crosswind_non_negative(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        result = analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        assert result.max_crosswind >= 0

    def test_does_not_mutate_aircraft(
        self, control_aircraft, flight_condition, weight_result, stability_result
    ):
        original_ref = list(control_aircraft.xyz_ref)
        analyze(
            control_aircraft, flight_condition, weight_result, stability_result,
            aero_method="vlm",
        )
        assert control_aircraft.xyz_ref == original_ref
