"""Phase 2e — Native solver validation tests.

All four vendored solvers are tested against analytical expectations and
cross-checked against each other where appropriate.  No AeroSandbox
dependency required.

Reference values for a rectangular NACA 0012 wing (AR = 8, α = 4°):
  - Thin-airfoil 3-D CL ≈ 2π·AR/(AR+2) · α_rad ≈ 0.35–0.42
  - Induced drag coefficient CDi ≈ CL² / (π·e·AR), e ≈ 0.9 → CDi ≈ 0.004–0.006
"""

from __future__ import annotations

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.aero import analyze


# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def rect_wing():
    """Rectangular wing: NACA 0012, 0.25 m chord, 1.0 m semispan (AR = 8)."""
    af = ap.Airfoil.from_naca("0012")
    return ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.25, airfoil=af),
            ap.WingXSec(xyz_le=[0.0, 1.0, 0.0], chord=0.25, airfoil=af),
        ],
        symmetric=True,
    )


@pytest.fixture(scope="module")
def tapered_wing():
    """Tapered wing: NACA 2412, 0.30 m root, 0.15 m tip, 0.8 m semispan."""
    af = ap.Airfoil.from_naca("2412")
    return ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.00, 0.00, 0.00], chord=0.30, airfoil=af),
            ap.WingXSec(xyz_le=[0.02, 0.80, 0.05], chord=0.15, airfoil=af),
        ],
        symmetric=True,
    )


@pytest.fixture(scope="module")
def rect_aircraft(rect_wing):
    return ap.Aircraft(name="rect", wings=[rect_wing])


@pytest.fixture(scope="module")
def tapered_aircraft(tapered_wing):
    return ap.Aircraft(name="tapered", wings=[tapered_wing])


@pytest.fixture(scope="module")
def cruise_condition():
    """Cruise: 20 m/s, 300 m, α = 4°."""
    return ap.FlightCondition(velocity=20.0, altitude=300.0, alpha=4.0)


@pytest.fixture(scope="module")
def zero_condition():
    """Zero-lift check: 20 m/s, 300 m, α = 0°."""
    return ap.FlightCondition(velocity=20.0, altitude=300.0, alpha=0.0)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _rel(a, b):
    """Relative difference |a - b| / (|b| + 1e-12)."""
    return abs(a - b) / (abs(b) + 1e-12)


# ------------------------------------------------------------------ #
# VLM tests
# ------------------------------------------------------------------ #

class TestVLMRectangular:
    """VLM on a plain rectangular NACA 0012 wing (AR = 8)."""

    @pytest.fixture(scope="class")
    def result(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition, method="vlm",
                       spanwise_resolution=8, chordwise_resolution=4)

    def test_CL_in_range(self, result):
        """3-D CL for AR=8, α=4°: thin airfoil theory gives ~0.38."""
        assert 0.25 < result.CL < 0.55, f"CL={result.CL:.4f} out of expected range"

    def test_CD_positive_and_small(self, result):
        """Induced drag only: CDi should be small (0.001–0.02) at α=4°."""
        assert 0.001 < result.CD < 0.02, f"CD={result.CD:.5f} out of range"

    def test_Cm_negative(self, result):
        """With xyz_ref at LE, NACA 0012 at α=4° has nose-down (negative) Cm."""
        assert result.Cm < 0.0, f"Cm={result.Cm:.4f} should be negative"

    def test_Cl_zero(self, result):
        """Rolling moment must be ~0 for symmetric wing at β=0."""
        assert abs(result.Cl) < 0.005, f"Cl={result.Cl:.6f} (expected ~0)"

    def test_Cn_zero(self, result):
        assert abs(result.Cn) < 0.005, f"Cn={result.Cn:.6f} (expected ~0)"

    def test_CDi_equals_CD(self, result):
        """VLM is inviscid — CDi must equal CD."""
        assert result.CDi is not None
        assert abs(result.CDi - result.CD) < 1e-10


class TestVLMZeroAlpha:
    """VLM at α = 0°: symmetric airfoil → CL ≈ 0, Cm ≈ 0."""

    @pytest.fixture(scope="class")
    def result(self, rect_aircraft, zero_condition):
        return analyze(rect_aircraft, zero_condition, method="vlm")

    def test_CL_near_zero(self, result):
        assert abs(result.CL) < 0.02, f"CL={result.CL:.4f} at α=0 (expected ~0)"

    def test_Cm_near_zero(self, result):
        assert abs(result.Cm) < 0.02, f"Cm={result.Cm:.4f} at α=0 (expected ~0)"


class TestVLMTapered:
    """VLM on tapered cambered wing — basic sanity."""

    @pytest.fixture(scope="class")
    def result(self, tapered_aircraft, cruise_condition):
        return analyze(tapered_aircraft, cruise_condition, method="vlm",
                       spanwise_resolution=8, chordwise_resolution=4)

    def test_CL_positive(self, result):
        assert result.CL > 0.3, f"CL={result.CL:.4f} unexpectedly low"

    def test_CD_positive(self, result):
        assert result.CD > 0.0

    def test_Cm_negative(self, result):
        assert result.Cm < 0.0, f"Cm={result.Cm:.4f} should be negative"


# ------------------------------------------------------------------ #
# LiftingLine tests
# ------------------------------------------------------------------ #

class TestLiftingLine:
    """LiftingLine (linear NeuralFoil polar) — sanity checks."""

    @pytest.fixture(scope="class")
    def result(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition, method="lifting_line",
                       spanwise_resolution=8, model_size="medium")

    def test_CL_in_range(self, result):
        assert 0.25 < result.CL < 0.55, f"CL={result.CL:.4f}"

    def test_CD_positive(self, result):
        assert result.CD > 0.0

    def test_Cm_negative(self, result):
        assert result.Cm < 0.0, f"Cm={result.Cm:.4f} should be negative"

    def test_Cl_zero(self, result):
        assert abs(result.Cl) < 0.005, f"Cl={result.Cl:.6f}"

    def test_CL_consistent_with_VLM(self, rect_aircraft, cruise_condition):
        """LL (NeuralFoil polars) and VLM (flat-plate) use different physical models;
        CL agreement within 20% is expected for a simple wing."""
        r_ll  = analyze(rect_aircraft, cruise_condition, method="lifting_line",
                        spanwise_resolution=8)
        r_vlm = analyze(rect_aircraft, cruise_condition, method="vlm",
                        spanwise_resolution=8)
        assert _rel(r_ll.CL, r_vlm.CL) < 0.20, \
            f"LL CL={r_ll.CL:.4f}, VLM CL={r_vlm.CL:.4f}"


# ------------------------------------------------------------------ #
# AeroBuildup tests
# ------------------------------------------------------------------ #

class TestAeroBuildup:
    """AeroBuildup — sanity and drag breakdown checks."""

    @pytest.fixture(scope="class")
    def result(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition, method="aero_buildup",
                       model_size="medium")

    def test_CL_in_range(self, result):
        assert 0.25 < result.CL < 0.55, f"CL={result.CL:.4f}"

    def test_CD_positive(self, result):
        assert result.CD > 0.0

    def test_Cm_negative(self, result):
        assert result.Cm < 0.0, f"Cm={result.Cm:.4f} should be negative"

    def test_CDi_CDp_available(self, result):
        """AeroBuildup must provide separate CDi and CDp."""
        assert result.CDi is not None, "CDi is None"
        assert result.CDp is not None, "CDp is None"
        assert result.CDi >= 0.0
        assert result.CDp >= 0.0

    def test_drag_split_sums(self, result):
        """CDi + CDp must equal CD."""
        assert abs(result.CDi + result.CDp - result.CD) < 1e-6, \
            f"CDi+CDp={result.CDi+result.CDp:.6f} != CD={result.CD:.6f}"

    def test_CL_consistent_with_VLM(self, rect_aircraft, cruise_condition):
        """AeroBuildup (NeuralFoil) vs VLM (flat-plate): agree within 20%."""
        r_ab  = analyze(rect_aircraft, cruise_condition, method="aero_buildup")
        r_vlm = analyze(rect_aircraft, cruise_condition, method="vlm")
        assert _rel(r_ab.CL, r_vlm.CL) < 0.20, \
            f"AeroBuildup CL={r_ab.CL:.4f}, VLM CL={r_vlm.CL:.4f}"


# ------------------------------------------------------------------ #
# NonlinearLiftingLine tests
# ------------------------------------------------------------------ #

class TestNonlinearLiftingLine:
    """NLL: convergence, physical sanity, rough agreement with LL."""

    @pytest.fixture(scope="class")
    def nll_result(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition,
                       method="nonlinear_lifting_line",
                       spanwise_resolution=8, model_size="medium")

    @pytest.fixture(scope="class")
    def ll_result(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition,
                       method="lifting_line",
                       spanwise_resolution=8, model_size="medium")

    def test_converged(self, nll_result):
        solver = nll_result._solver
        assert hasattr(solver, "n_iter"), "NLL solver has no n_iter attribute"
        assert solver.n_iter < solver.max_iter, \
            f"NLL did not converge (n_iter={solver.n_iter})"

    def test_CL_positive(self, nll_result):
        assert nll_result.CL > 0.1

    def test_CD_positive(self, nll_result):
        assert nll_result.CD > 0.0

    def test_Cl_zero(self, nll_result):
        assert abs(nll_result.Cl) < 0.005

    def test_CL_close_to_LL(self, nll_result, ll_result):
        """NLL and LL use different model formulations; agree within 20%."""
        assert _rel(nll_result.CL, ll_result.CL) < 0.20, \
            f"CL: NLL={nll_result.CL:.4f}, LL={ll_result.CL:.4f}"


# ------------------------------------------------------------------ #
# Sign-convention and physics invariant tests
# ------------------------------------------------------------------ #

class TestSignConventions:
    """Basic invariants that must hold for all methods."""

    @pytest.mark.parametrize("method", ["vlm", "lifting_line", "aero_buildup"])
    def test_CL_increases_with_alpha(self, rect_aircraft, method):
        """CL at α=6° must exceed CL at α=2°."""
        r_low  = analyze(rect_aircraft,
                         ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=2.0),
                         method=method)
        r_high = analyze(rect_aircraft,
                         ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=6.0),
                         method=method)
        assert r_high.CL > r_low.CL, \
            f"{method}: CL does not increase with alpha ({r_low.CL:.3f} → {r_high.CL:.3f})"

    @pytest.mark.parametrize("method", ["vlm", "lifting_line", "aero_buildup"])
    def test_CD_positive(self, rect_aircraft, method):
        r = analyze(rect_aircraft,
                    ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=4.0),
                    method=method)
        assert r.CD > 0.0, f"{method}: CD={r.CD:.5f} is not positive"

    @pytest.mark.parametrize("method", ["vlm", "lifting_line", "aero_buildup"])
    def test_L_equals_CL_qS(self, rect_aircraft, method):
        """L = CL * q * S_ref must hold to floating-point precision."""
        cond = ap.FlightCondition(velocity=20.0, altitude=0.0, alpha=4.0)
        r = analyze(rect_aircraft, cond, method=method)
        L_expected = r.CL * r.dynamic_pressure * r.s_ref
        assert abs(r.L - L_expected) < 0.01, \
            f"{method}: L={r.L:.3f}, CL*q*S={L_expected:.3f}"
