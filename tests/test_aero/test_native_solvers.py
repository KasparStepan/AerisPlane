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


# ------------------------------------------------------------------ #
# Phase 4 — Control surface deflection tests
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def elevator_aircraft():
    """Symmetric tail with a full-span elevator (35% chord)."""
    af = ap.Airfoil.from_naca("0012")
    htail = ap.Wing(
        name="htail",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.00, 0.0], chord=0.15, airfoil=af),
            ap.WingXSec(xyz_le=[0.0, 0.30, 0.0], chord=0.15, airfoil=af),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="elevator",
                span_start=0.0, span_end=1.0,
                chord_fraction=0.35,
                symmetric=True,         # both sides deflect together
            ),
        ],
    )
    return ap.Aircraft(name="tail", wings=[htail])


@pytest.fixture(scope="module")
def aileron_aircraft():
    """Wing with part-span ailerons (outboard 50–90%, 25% chord)."""
    af = ap.Airfoil.from_naca("0012")
    wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.25, airfoil=af),
            ap.WingXSec(xyz_le=[0.0, 1.0, 0.0], chord=0.25, airfoil=af),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="aileron",
                span_start=0.5, span_end=0.9,
                chord_fraction=0.25,
                symmetric=False,        # differential: right TE-down, left TE-up
            ),
        ],
    )
    return ap.Aircraft(name="aileron_wing", wings=[wing])


@pytest.fixture(scope="module")
def flap_aircraft():
    """Wing with full-span flap (30% chord)."""
    af = ap.Airfoil.from_naca("0012")
    wing = ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.25, airfoil=af),
            ap.WingXSec(xyz_le=[0.0, 1.0, 0.0], chord=0.25, airfoil=af),
        ],
        symmetric=True,
        control_surfaces=[
            ap.ControlSurface(
                name="flap",
                span_start=0.0, span_end=1.0,
                chord_fraction=0.30,
                symmetric=True,
            ),
        ],
    )
    return ap.Aircraft(name="flap_wing", wings=[wing])


def _vlm(aircraft, alpha=4.0, deflections=None):
    cond = ap.FlightCondition(
        velocity=20.0, altitude=0.0, alpha=alpha,
        deflections=deflections or {},
    )
    return analyze(aircraft, cond, method="vlm",
                   spanwise_resolution=8, chordwise_resolution=4)


class TestElevatorDeflection:
    """Positive elevator (TE down) should increase nose-down pitching moment."""

    def test_elevator_down_increases_nose_down_Cm(self, elevator_aircraft):
        """δe = +10° (TE down) must give more negative Cm than δe = 0°."""
        r0  = _vlm(elevator_aircraft, deflections={})
        r10 = _vlm(elevator_aircraft, deflections={"elevator": 10.0})
        assert r10.Cm < r0.Cm, (
            f"Elevator +10° expected Cm < Cm(0): "
            f"{r10.Cm:.4f} vs {r0.Cm:.4f}"
        )

    def test_elevator_up_decreases_nose_down_Cm(self, elevator_aircraft):
        """δe = −10° (TE up) must give less negative Cm than δe = 0°."""
        r0   = _vlm(elevator_aircraft, deflections={})
        rm10 = _vlm(elevator_aircraft, deflections={"elevator": -10.0})
        assert rm10.Cm > r0.Cm, (
            f"Elevator −10° expected Cm > Cm(0): "
            f"{rm10.Cm:.4f} vs {r0.Cm:.4f}"
        )

    def test_Cm_monotone_with_elevator(self, elevator_aircraft):
        """Cm must decrease monotonically as elevator deflection increases."""
        deflections = [-20.0, -10.0, 0.0, 10.0, 20.0]
        Cms = [_vlm(elevator_aircraft,
                    deflections={"elevator": d}).Cm
               for d in deflections]
        for i in range(len(Cms) - 1):
            assert Cms[i] > Cms[i + 1], (
                f"Cm not monotone at index {i}: "
                f"Cm({deflections[i]}°)={Cms[i]:.4f}, "
                f"Cm({deflections[i+1]}°)={Cms[i+1]:.4f}"
            )

    def test_elevator_Cm_magnitude(self, elevator_aircraft):
        """ΔCm between ±20° should be meaningful (> 0.05) but not absurd (< 5)."""
        rm20 = _vlm(elevator_aircraft, deflections={"elevator": -20.0})
        rp20 = _vlm(elevator_aircraft, deflections={"elevator":  20.0})
        delta_Cm = rm20.Cm - rp20.Cm
        assert 0.05 < delta_Cm < 5.0, (
            f"ΔCm(±20°) = {delta_Cm:.4f} outside expected range [0.05, 5]"
        )

    def test_zero_deflection_matches_no_deflections(self, elevator_aircraft):
        """Passing deflections={} and deflections={'elevator': 0.0} must give same result."""
        r_none = _vlm(elevator_aircraft, deflections={})
        r_zero = _vlm(elevator_aircraft, deflections={"elevator": 0.0})
        assert abs(r_none.CL - r_zero.CL) < 1e-8
        assert abs(r_none.Cm - r_zero.Cm) < 1e-8

    def test_elevator_preserves_symmetry(self, elevator_aircraft):
        """Symmetric elevator deflection must keep Cl ≈ 0 and Cn ≈ 0."""
        r = _vlm(elevator_aircraft, deflections={"elevator": 15.0})
        assert abs(r.Cl) < 0.005, f"Cl={r.Cl:.6f} (expected ~0)"
        assert abs(r.Cn) < 0.005, f"Cn={r.Cn:.6f} (expected ~0)"


class TestAileronDeflection:
    """Positive aileron (right TE down, left TE up) produces positive roll."""

    def test_aileron_positive_deflection_rolls_left(self, aileron_aircraft):
        """δa = +5° (right TE-down, left TE-up): more lift on right wing →
        right wing up → left roll → Cl < 0."""
        r = _vlm(aileron_aircraft, deflections={"aileron": 5.0})
        assert r.Cl < 0.0, f"Aileron +5°: expected Cl < 0, got Cl={r.Cl:.6f}"

    def test_aileron_negative_deflection_rolls_right(self, aileron_aircraft):
        """δa = −5° (right TE-up, left TE-down): more lift on left wing →
        left wing up → right roll → Cl > 0."""
        r = _vlm(aileron_aircraft, deflections={"aileron": -5.0})
        assert r.Cl > 0.0, f"Aileron −5°: expected Cl > 0, got Cl={r.Cl:.6f}"

    def test_aileron_antisymmetric(self, aileron_aircraft):
        """Cl(+δ) must equal −Cl(−δ) (antisymmetry)."""
        rp = _vlm(aileron_aircraft, deflections={"aileron":  10.0})
        rm = _vlm(aileron_aircraft, deflections={"aileron": -10.0})
        assert abs(rp.Cl + rm.Cl) < 0.005 * abs(rp.Cl), (
            f"Aileron not antisymmetric: Cl(+10°)={rp.Cl:.5f}, Cl(−10°)={rm.Cl:.5f}"
        )

    def test_aileron_does_not_change_CL_much(self, aileron_aircraft):
        """Aileron (antisymmetric) should not change total CL by more than 5%."""
        r0 = _vlm(aileron_aircraft, deflections={})
        r  = _vlm(aileron_aircraft, deflections={"aileron": 10.0})
        assert abs(r.CL - r0.CL) < 0.05 * abs(r0.CL + 1e-6), (
            f"Aileron changed CL too much: ΔCL={r.CL - r0.CL:.4f}"
        )

    def test_aileron_Cl_magnitude(self, aileron_aircraft):
        """|Cl| at 10° aileron must be non-trivial (> 0.005)."""
        r = _vlm(aileron_aircraft, deflections={"aileron": 10.0})
        assert abs(r.Cl) > 0.005, f"Cl={r.Cl:.5f} (expected |Cl| > 0.005)"


class TestFlapDeflection:
    """Symmetric flap (TE down) increases CL and Cm at fixed alpha."""

    def test_flap_increases_CL(self, flap_aircraft):
        """δf = +15° must increase CL compared to clean configuration."""
        r0  = _vlm(flap_aircraft, deflections={})
        r15 = _vlm(flap_aircraft, deflections={"flap": 15.0})
        assert r15.CL > r0.CL, (
            f"Flap +15°: expected CL increase, "
            f"got CL={r15.CL:.4f} vs {r0.CL:.4f}"
        )

    def test_flap_preserves_symmetry(self, flap_aircraft):
        """Symmetric flap must keep Cl ≈ 0."""
        r = _vlm(flap_aircraft, deflections={"flap": 20.0})
        assert abs(r.Cl) < 0.005, f"Cl={r.Cl:.6f} (expected ~0)"

    def test_flap_CL_increases_with_deflection(self, flap_aircraft):
        """CL must increase monotonically with flap deflection at fixed alpha."""
        deflections = [0.0, 10.0, 20.0, 30.0]
        CLs = [_vlm(flap_aircraft, deflections={"flap": d}).CL for d in deflections]
        for i in range(len(CLs) - 1):
            assert CLs[i] < CLs[i + 1], (
                f"CL not monotone with flap: "
                f"CL({deflections[i]}°)={CLs[i]:.4f}, "
                f"CL({deflections[i+1]}°)={CLs[i+1]:.4f}"
            )


# ------------------------------------------------------------------ #
# Fuselage interference tests
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def fuselage_aircraft(rect_wing):
    """Rectangular wing + fuselage (d ~ 10% of span)."""
    fuse = ap.Fuselage(
        name="fuselage",
        xsecs=[
            ap.FuselageXSec(x=0.00, radius=0.01),
            ap.FuselageXSec(x=0.05, radius=0.05),
            ap.FuselageXSec(x=0.15, radius=0.10),
            ap.FuselageXSec(x=0.50, radius=0.10),
            ap.FuselageXSec(x=0.80, radius=0.05),
            ap.FuselageXSec(x=1.00, radius=0.01),
        ],
    )
    return ap.Aircraft(name="with_fuse", wings=[rect_wing], fuselages=[fuse])


class TestFuselageUpwashLL:
    """Fuselage displacement effect should increase wing CL in LL solver."""

    @pytest.fixture(scope="class")
    def result_no_fuse(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition, method="lifting_line",
                       spanwise_resolution=8)

    @pytest.fixture(scope="class")
    def result_with_fuse(self, fuselage_aircraft, cruise_condition):
        return analyze(fuselage_aircraft, cruise_condition, method="lifting_line",
                       spanwise_resolution=8)

    def test_CL_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        """Fuselage upwash should increase effective alpha → higher CL on wing."""
        assert result_with_fuse.CL > result_no_fuse.CL, (
            f"CL_fuse={result_with_fuse.CL:.4f} should be > CL_no_fuse={result_no_fuse.CL:.4f}"
        )

    def test_CL_increase_is_small(self, result_no_fuse, result_with_fuse):
        """For d/b ~ 10%, the CL increase should be modest (< 10%)."""
        delta = (result_with_fuse.CL - result_no_fuse.CL) / result_no_fuse.CL
        assert delta < 0.10, f"ΔCL/CL = {delta:.3f} is too large"

    def test_CD_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        """Fuselage adds its own drag contribution."""
        assert result_with_fuse.CD > result_no_fuse.CD


class TestFuselageUpwashNLL:
    """Fuselage displacement effect in nonlinear lifting line."""

    @pytest.fixture(scope="class")
    def result_no_fuse(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition,
                       method="nonlinear_lifting_line", spanwise_resolution=8)

    @pytest.fixture(scope="class")
    def result_with_fuse(self, fuselage_aircraft, cruise_condition):
        return analyze(fuselage_aircraft, cruise_condition,
                       method="nonlinear_lifting_line", spanwise_resolution=8)

    def test_CL_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        assert result_with_fuse.CL > result_no_fuse.CL, (
            f"CL_fuse={result_with_fuse.CL:.4f} <= CL_no_fuse={result_no_fuse.CL:.4f}"
        )

    def test_CL_increase_is_small(self, result_no_fuse, result_with_fuse):
        delta = (result_with_fuse.CL - result_no_fuse.CL) / result_no_fuse.CL
        assert delta < 0.10, f"ΔCL/CL = {delta:.3f} is too large"


class TestAeroBuildupInterference:
    """AeroBuildup with fuselage should include junction drag."""

    @pytest.fixture(scope="class")
    def result_no_fuse(self, rect_aircraft, cruise_condition):
        return analyze(rect_aircraft, cruise_condition, method="aero_buildup")

    @pytest.fixture(scope="class")
    def result_with_fuse(self, fuselage_aircraft, cruise_condition):
        return analyze(fuselage_aircraft, cruise_condition, method="aero_buildup")

    def test_CD_increases_with_fuselage(self, result_no_fuse, result_with_fuse):
        """Fuselage adds skin friction + junction drag → higher CD."""
        assert result_with_fuse.CD > result_no_fuse.CD


class TestLLJunctionDrag:
    """LL with fuselage should include junction drag."""

    def test_ll_CD_with_fuse_higher_than_without(self, rect_aircraft, fuselage_aircraft, cruise_condition):
        r_no = analyze(rect_aircraft, cruise_condition, method="lifting_line", spanwise_resolution=8)
        r_with = analyze(fuselage_aircraft, cruise_condition, method="lifting_line", spanwise_resolution=8)
        assert r_with.CD > r_no.CD, (
            f"LL CD_fuse={r_with.CD:.5f} should be > CD_no_fuse={r_no.CD:.5f}"
        )
