"""Control surface deflection tests for LL, NLL, and AeroBuildup.

Verifies that:
  1. Elevator (symmetric): positive deflection increases CL and shifts Cm negative.
  2. Aileron (asymmetric): positive deflection produces rolling moment Cl with
     negligible net pitching moment change.
  3. Zero deflection reproduces the baseline result unchanged.
  4. VLM and LL agree in sign and rough magnitude for elevator ΔCm.

Physics baseline: Glauert thin-airfoil theory for a plain flap with
chord_fraction = 0.3 predicts:
  theta_f = arccos(1 - 2*0.3) = 1.159 rad
  dCl/ddelta = 2*(pi - theta_f + sin(theta_f)) ≈ 4.05 / rad  ≈ 0.0707 / deg
  dCm/ddelta = -(sin(theta_f) + sin(2*theta_f)/2) / 2 ≈ -0.465 / rad ≈ -0.00812 / deg

For a 10° elevator deflection on a whole-wing surface, we expect:
  ΔCl_2D ≈ +0.707,  ΔCm_2D ≈ -0.081
These are 2-D section values; 3-D will be smaller.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import aerisplane as ap
from aerisplane.aero import analyze
from aerisplane.core.control_surface import ControlSurface


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

CHORD_FRACTION = 0.30   # flap-to-chord ratio for all test surfaces
ELEVATOR_DEG   = 10.0   # elevator deflection [deg]
AILERON_DEG    = 10.0   # aileron deflection [deg]


@pytest.fixture(scope="module")
def af():
    return ap.Airfoil.from_naca("0012")


@pytest.fixture(scope="module")
def elevator_wing(af):
    """Rectangular wing with a symmetric elevator spanning the full semispan."""
    elevator = ControlSurface(
        name="elevator",
        span_start=0.0,
        span_end=1.0,
        chord_fraction=CHORD_FRACTION,
        symmetric=True,
    )
    return ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.25, airfoil=af),
            ap.WingXSec(xyz_le=[0.0, 1.0, 0.0], chord=0.25, airfoil=af),
        ],
        symmetric=True,
        control_surfaces=[elevator],
    )


@pytest.fixture(scope="module")
def aileron_wing(af):
    """Rectangular wing with an asymmetric aileron on the outer 50% semispan."""
    aileron = ControlSurface(
        name="aileron",
        span_start=0.5,
        span_end=1.0,
        chord_fraction=CHORD_FRACTION,
        symmetric=False,
    )
    return ap.Wing(
        name="main_wing",
        xsecs=[
            ap.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=0.25, airfoil=af),
            ap.WingXSec(xyz_le=[0.0, 0.5, 0.0], chord=0.25, airfoil=af),
            ap.WingXSec(xyz_le=[0.0, 1.0, 0.0], chord=0.25, airfoil=af),
        ],
        symmetric=True,
        control_surfaces=[aileron],
    )


@pytest.fixture(scope="module")
def elevator_aircraft(elevator_wing):
    return ap.Aircraft(name="elevator_test", wings=[elevator_wing])


@pytest.fixture(scope="module")
def aileron_aircraft(aileron_wing):
    return ap.Aircraft(name="aileron_test", wings=[aileron_wing])


@pytest.fixture(scope="module")
def base_condition():
    return ap.FlightCondition(velocity=20.0, altitude=300.0, alpha=4.0)


@pytest.fixture(scope="module")
def elevator_condition(base_condition):
    cond = base_condition.copy()
    cond.deflections = {"elevator": ELEVATOR_DEG}
    return cond


@pytest.fixture(scope="module")
def aileron_condition(base_condition):
    cond = base_condition.copy()
    cond.deflections = {"aileron": AILERON_DEG}
    return cond


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

SOLVERS = ["lifting_line", "nonlinear_lifting_line", "aero_buildup"]


def _run(aircraft, condition, method):
    return analyze(aircraft, condition, method=method, spanwise_resolution=2)


# ------------------------------------------------------------------ #
# 1. Elevator: positive deflection → ΔCL > 0
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("method", SOLVERS)
def test_elevator_increases_CL(elevator_aircraft, base_condition, elevator_condition, method):
    """Trailing-edge-down elevator must increase lift coefficient."""
    r_base = _run(elevator_aircraft, base_condition, method)
    r_defl = _run(elevator_aircraft, elevator_condition, method)
    assert r_defl.CL > r_base.CL, (
        f"[{method}] Expected CL increase from elevator deflection: "
        f"base={r_base.CL:.4f}, deflected={r_defl.CL:.4f}"
    )


# ------------------------------------------------------------------ #
# 2. Elevator: positive deflection → ΔCm < 0  (nose-down)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("method", SOLVERS)
def test_elevator_shifts_Cm_negative(elevator_aircraft, base_condition, elevator_condition, method):
    """Trailing-edge-down elevator must shift Cm in nose-down (negative) direction."""
    r_base = _run(elevator_aircraft, base_condition, method)
    r_defl = _run(elevator_aircraft, elevator_condition, method)
    assert r_defl.Cm < r_base.Cm, (
        f"[{method}] Expected Cm to decrease from elevator deflection: "
        f"base={r_base.Cm:.4f}, deflected={r_defl.Cm:.4f}"
    )


# ------------------------------------------------------------------ #
# 3. Elevator: ΔCm magnitude is plausible (thin-airfoil theory bound)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("method", SOLVERS)
def test_elevator_Cm_magnitude(elevator_aircraft, base_condition, elevator_condition, method):
    """ΔCm should be negative and non-trivial.

    Note: the moment reference is at the LE (xyz_ref=[0,0,0]), so ΔCm includes
    both the quarter-chord increment (ΔCm_qc ≈ −0.08) and the arm from ΔCl
    acting at the aerodynamic centre (ΔCl * x_ac/c ≈ 0.18), giving a total
    ΔCm ≈ −0.26 to −0.35 for a full-span elevator at 10°.  We only bound the
    minimum to confirm the surface is active.
    """
    r_base = _run(elevator_aircraft, base_condition, method)
    r_defl = _run(elevator_aircraft, elevator_condition, method)
    dCm = r_defl.Cm - r_base.Cm

    assert dCm < 0, f"[{method}] ΔCm should be negative, got {dCm:.4f}"
    assert abs(dCm) > 0.01, f"[{method}] ΔCm too small ({dCm:.4f}), CS likely not active"


# ------------------------------------------------------------------ #
# 4. Aileron: positive deflection → nonzero rolling moment Cl
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("method", SOLVERS)
def test_aileron_produces_roll(aileron_aircraft, base_condition, aileron_condition, method):
    """Positive aileron (right TE down) must produce a nonzero roll moment."""
    r_base = _run(aileron_aircraft, base_condition, method)
    r_defl = _run(aileron_aircraft, aileron_condition, method)
    dCl = r_defl.Cl - r_base.Cl

    # Right aileron positive → right wing more lift → |Cl| > 0
    assert abs(dCl) > 1e-4, (
        f"[{method}] Aileron produced no rolling moment: dCl={dCl:.6f}"
    )


# ------------------------------------------------------------------ #
# 5. Aileron: symmetric wing → negligible net ΔCm (cancels left/right)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("method", SOLVERS)
def test_aileron_no_net_pitching_moment(aileron_aircraft, base_condition, aileron_condition, method):
    """Aileron on symmetric wing should produce negligible net ΔCm."""
    r_base = _run(aileron_aircraft, base_condition, method)
    r_defl = _run(aileron_aircraft, aileron_condition, method)
    dCm = abs(r_defl.Cm - r_base.Cm)

    # The aileron correction is antisymmetric, so Cm contributions cancel
    assert dCm < 0.02, (
        f"[{method}] Aileron produced unexpectedly large ΔCm={dCm:.4f}; "
        "left/right contributions should cancel"
    )


# ------------------------------------------------------------------ #
# 6. Zero deflection → result identical to baseline
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("method", SOLVERS)
def test_zero_deflection_unchanged(elevator_aircraft, base_condition, method):
    """Passing an explicit zero deflection should not change the result."""
    cond_zero = base_condition.copy()
    cond_zero.deflections = {"elevator": 0.0}

    r_base = _run(elevator_aircraft, base_condition, method)
    r_zero = _run(elevator_aircraft, cond_zero, method)

    assert abs(r_zero.CL - r_base.CL) < 1e-8, (
        f"[{method}] Zero deflection changed CL: {r_base.CL:.6f} → {r_zero.CL:.6f}"
    )
    assert abs(r_zero.Cm - r_base.Cm) < 1e-8, (
        f"[{method}] Zero deflection changed Cm: {r_base.Cm:.6f} → {r_zero.Cm:.6f}"
    )


# ------------------------------------------------------------------ #
# 7. VLM vs LL consistency: ΔCm same sign for small elevator deflection
# ------------------------------------------------------------------ #

def test_vlm_ll_elevator_same_sign(elevator_aircraft, base_condition):
    """VLM (Rodrigues) and LL (thin-airfoil) should agree in sign of ΔCm."""
    cond_small = base_condition.copy()
    cond_small.deflections = {"elevator": 5.0}

    r_vlm_base = _run(elevator_aircraft, base_condition, "vlm")
    r_vlm_defl = _run(elevator_aircraft, cond_small, "vlm")
    r_ll_base  = _run(elevator_aircraft, base_condition, "lifting_line")
    r_ll_defl  = _run(elevator_aircraft, cond_small, "lifting_line")

    dCm_vlm = r_vlm_defl.Cm - r_vlm_base.Cm
    dCm_ll  = r_ll_defl.Cm  - r_ll_base.Cm

    assert dCm_vlm < 0, f"VLM ΔCm should be negative, got {dCm_vlm:.4f}"
    assert dCm_ll  < 0, f"LL ΔCm should be negative, got {dCm_ll:.4f}"
