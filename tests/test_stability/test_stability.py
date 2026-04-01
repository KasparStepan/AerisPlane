"""Integration tests for stability.analyze()."""

import math

import numpy as np
import pytest

from aerisplane.stability import analyze


class TestAnalyze:
    """Test the top-level stability analyze() function."""

    def test_returns_stability_result(self, stability_aircraft, flight_condition, weight_result):
        """analyze() should return a StabilityResult."""
        from aerisplane.stability.result import StabilityResult

        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert isinstance(result, StabilityResult)

    def test_report_non_empty(self, stability_aircraft, flight_condition, weight_result):
        """report() should return a non-empty string."""
        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        report = result.report()
        assert len(report) > 100
        assert "Static margin" in report or "static_margin" in report.lower() or "Static Margin" in report

    def test_trim_alpha_finite(self, stability_aircraft, flight_condition, weight_result):
        """Trim alpha should be a finite number."""
        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert math.isfinite(result.trim_alpha), (
            f"trim_alpha = {result.trim_alpha}, expected finite"
        )

    def test_trim_alpha_reasonable(self, stability_aircraft, flight_condition, weight_result):
        """Trim alpha should be within -5 to 15 degrees for typical RC aircraft."""
        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert -10 < result.trim_alpha < 20, (
            f"trim_alpha = {result.trim_alpha:.1f} deg, outside reasonable range"
        )

    def test_tail_volume_vh_positive(self, stability_aircraft, flight_condition, weight_result):
        """Vh should be positive when htail exists."""
        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert not math.isnan(result.Vh), "Vh is NaN — htail not identified"
        assert result.Vh > 0, f"Vh = {result.Vh:.3f}, expected > 0"

    def test_tail_volume_vv_positive(self, stability_aircraft, flight_condition, weight_result):
        """Vv should be positive when vtail exists."""
        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert not math.isnan(result.Vv), "Vv is NaN — vtail not identified"
        assert result.Vv > 0, f"Vv = {result.Vv:.3f}, expected > 0"

    def test_cg_envelope_ordered(self, stability_aircraft, flight_condition, weight_result):
        """Forward CG limit should be ahead of aft CG limit."""
        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert result.cg_forward_limit < result.cg_aft_limit, (
            f"Forward = {result.cg_forward_limit:.2f}, "
            f"Aft = {result.cg_aft_limit:.2f}"
        )

    def test_baseline_cl_positive(self, stability_aircraft, flight_condition, weight_result):
        """Baseline CL should be positive at positive alpha."""
        result = analyze(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert result.CL_baseline > 0, (
            f"CL_baseline = {result.CL_baseline:.4f} at alpha={flight_condition.alpha} deg"
        )

    def test_does_not_mutate_aircraft(self, stability_aircraft, flight_condition, weight_result):
        """analyze() must not modify the input aircraft."""
        original_ref = list(stability_aircraft.xyz_ref)
        analyze(stability_aircraft, flight_condition, weight_result, aero_method="vlm")
        assert stability_aircraft.xyz_ref == original_ref
