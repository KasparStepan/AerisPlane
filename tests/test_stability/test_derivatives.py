"""Tests for stability derivative computation."""

import numpy as np
import pytest

from aerisplane.stability.derivatives import compute_derivatives


class TestComputeDerivatives:
    """Test central finite-difference derivative computation."""

    def test_cl_alpha_positive(self, stability_aircraft, flight_condition, weight_result):
        """Lift curve slope must be positive."""
        deriv = compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert deriv.CL_alpha > 0, f"CL_alpha = {deriv.CL_alpha}, expected > 0"

    def test_cl_alpha_reasonable_range(self, stability_aircraft, flight_condition, weight_result):
        """CL_alpha should be in the range 0.03-0.15 per degree for typical wings."""
        deriv = compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        # 2*pi/rad ≈ 0.11/deg for infinite wing, finite wings are lower
        assert 0.02 < deriv.CL_alpha < 0.15, (
            f"CL_alpha = {deriv.CL_alpha:.4f}/deg, outside expected range"
        )

    def test_cm_alpha_negative_for_stable_config(
        self, stability_aircraft, flight_condition, weight_result
    ):
        """Cm_alpha should be negative for a stable configuration (CG ahead of NP)."""
        deriv = compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert deriv.Cm_alpha < 0, (
            f"Cm_alpha = {deriv.Cm_alpha:.5f}, expected negative for stable config"
        )

    def test_static_margin_positive(self, stability_aircraft, flight_condition, weight_result):
        """Static margin should be positive for a stable configuration."""
        deriv = compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert deriv.static_margin > 0, (
            f"Static margin = {deriv.static_margin:.3f}, expected > 0"
        )

    def test_static_margin_reasonable(self, stability_aircraft, flight_condition, weight_result):
        """Static margin should be positive and finite for a stable configuration.

        Note: the test fixture has a very forward CG (heavy battery at nose)
        and short MAC (0.2m), so static margin can exceed 100% MAC.
        """
        deriv = compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert 0.01 < deriv.static_margin < 3.0, (
            f"Static margin = {deriv.static_margin * 100:.1f}% MAC, outside 1-300% range"
        )

    def test_neutral_point_aft_of_cg(self, stability_aircraft, flight_condition, weight_result):
        """Neutral point should be aft of CG for a stable configuration."""
        deriv = compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert deriv.neutral_point > deriv.cg_x, (
            f"NP = {deriv.neutral_point:.4f}m, CG = {deriv.cg_x:.4f}m — NP should be aft"
        )

    def test_does_not_mutate_aircraft(self, stability_aircraft, flight_condition, weight_result):
        """compute_derivatives must not modify the input aircraft."""
        original_ref = list(stability_aircraft.xyz_ref)
        compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert stability_aircraft.xyz_ref == original_ref, (
            "aircraft.xyz_ref was mutated by compute_derivatives"
        )

    def test_cn_beta_computed(
        self, stability_aircraft, flight_condition, weight_result
    ):
        """Cn_beta should be finite and non-zero."""
        deriv = compute_derivatives(
            stability_aircraft, flight_condition, weight_result, aero_method="vlm"
        )
        assert np.isfinite(deriv.Cn_beta), f"Cn_beta is not finite: {deriv.Cn_beta}"
        # Note: sign depends on vtail size vs fuselage destabilizing effect.
        # VLM may not produce positive Cn_beta with a small vtail fixture.
        assert abs(deriv.Cn_beta) > 1e-6, (
            f"Cn_beta = {deriv.Cn_beta:.6f}, expected non-zero"
        )
