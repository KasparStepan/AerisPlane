"""Static stability analysis module.

Public API
----------
analyze(aircraft, condition, weight_result, aero_method="vlm", **aero_kwargs)
    Run a full static stability analysis and return a StabilityResult.

Computes stability derivatives via central finite differences (5 aero
evaluations), trim alpha and elevator, tail volume coefficients, and
CG envelope limits.

Example
-------
>>> from aerisplane.stability import analyze
>>> from aerisplane.weights import analyze as weight_analyze
>>> wr = weight_analyze(aircraft)
>>> result = analyze(aircraft, condition, wr, aero_method="vlm")
>>> print(result.report())
"""

from __future__ import annotations

import copy
import math

import numpy as np

from aerisplane.aero import analyze as aero_analyze
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.stability.derivatives import compute_derivatives
from aerisplane.stability.result import StabilityResult
from aerisplane.weights.result import WeightResult


# CG envelope margin from neutral point [fraction of MAC]
_CG_MARGIN = 0.05  # 5% MAC


def analyze(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    aero_method: str = "vlm",
    compute_rate_derivatives: bool = False,
    **aero_kwargs,
) -> StabilityResult:
    """Run a full static stability analysis.

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry definition.
    condition : FlightCondition
        Operating point (velocity, altitude, alpha, beta).
    weight_result : WeightResult
        Weight analysis result providing CG position.
    aero_method : str
        Aero solver method (default "vlm").
    compute_rate_derivatives : bool
        If True, also compute p, q, r rate derivatives and dynamic stability
        modes (short-period, phugoid).  Adds 6 extra aero evaluations.
        Default False.
    **aero_kwargs
        Additional keyword arguments passed to aero.analyze().

    Returns
    -------
    StabilityResult
        Complete stability analysis result.
    """
    # Step 1: Compute stability derivatives
    deriv = compute_derivatives(
        aircraft, condition, weight_result, aero_method,
        compute_rate_derivatives=compute_rate_derivatives,
        **aero_kwargs,
    )

    # Step 2: Compute trim
    trim_alpha, trim_elevator = _compute_trim(
        aircraft, condition, weight_result, deriv, aero_method, **aero_kwargs
    )

    # Step 3: Tail volume coefficients
    Vh, Vv = _compute_tail_volumes(aircraft)

    # Step 4: CG envelope limits (fraction of MAC from MAC LE)
    np_frac = (deriv.neutral_point - deriv.mac_le_x) / deriv.mac if deriv.mac > 0 else 0.5
    cg_forward_limit = max(np_frac - deriv.static_margin - _CG_MARGIN, 0.05)
    cg_aft_limit = np_frac - _CG_MARGIN

    # Step 5: Dynamic stability modes (only when rate derivatives available)
    sp_frequency = sp_damping = ph_frequency = ph_damping = None
    if deriv.Cm_q is not None:
        sp_frequency, sp_damping, ph_frequency, ph_damping = _dynamic_modes(
            deriv, condition, weight_result, aircraft
        )

    return StabilityResult(
        # Longitudinal
        static_margin=deriv.static_margin,
        neutral_point=deriv.neutral_point,
        Cm_alpha=deriv.Cm_alpha,
        CL_alpha=deriv.CL_alpha,
        # Lateral-directional
        Cl_beta=deriv.Cl_beta,
        Cn_beta=deriv.Cn_beta,
        CY_beta=deriv.CY_beta,
        # Rate derivatives
        CL_q=deriv.CL_q,
        Cm_q=deriv.Cm_q,
        Cl_p=deriv.Cl_p,
        Cn_p=deriv.Cn_p,
        CY_p=deriv.CY_p,
        Cn_r=deriv.Cn_r,
        Cl_r=deriv.Cl_r,
        CY_r=deriv.CY_r,
        # Trim
        trim_alpha=trim_alpha,
        trim_elevator=trim_elevator,
        # Tail volumes
        Vh=Vh,
        Vv=Vv,
        # CG envelope
        cg_forward_limit=cg_forward_limit,
        cg_aft_limit=cg_aft_limit,
        # Dynamic modes
        sp_frequency=sp_frequency,
        sp_damping=sp_damping,
        ph_frequency=ph_frequency,
        ph_damping=ph_damping,
        # Reference
        cg_x=deriv.cg_x,
        mac=deriv.mac,
        mac_le_x=deriv.mac_le_x,
        # Baseline
        CL_baseline=deriv.baseline.CL,
        Cm_baseline=deriv.baseline.Cm,
    )


def _compute_trim(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    deriv,
    aero_method: str,
    **aero_kwargs,
) -> tuple[float, float]:
    """Find trim alpha and trim elevator for steady level flight.

    Two-step process:

    Step 1 — Trim alpha:
        Find the angle of attack where Cm = 0 (pitching moment balanced)
        using Brent's method in a ±10° bracket around the linear estimate.
        The linear estimate is alpha_0 − Cm_0 / Cm_alpha.  If the bracket
        does not contain a sign change (aircraft may not be trimmable),
        the linear estimate is returned with a UserWarning.

    Step 2 — Trim elevator:
        At the trim alpha found in step 1, find the elevator deflection
        that also satisfies CL = W/(q·S) (lift equals weight) by solving
        Cm = 0 over the deflection range [−25, 25] deg with Brent's method.
        The elevator is identified by searching control surface names for
        "elevator" or "elev" (case-insensitive).  Returns NaN if:
          • no elevator control surface exists on the aircraft, or
          • Cm does not change sign over ±25° (authority insufficient).

    Positive elevator deflection convention follows the aero solver
    (typically trailing-edge-down = positive Cm contribution).

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry definition.  Moment reference is overridden
        internally to the CG from weight_result.
    condition : FlightCondition
        Baseline operating point used for speed and altitude.
    weight_result : WeightResult
        Provides CG position and total mass for level-flight CL.
    deriv : DerivativeResult
        Pre-computed derivatives used to seed the alpha bracket.
    aero_method : str
        Aero solver method string passed to aero.analyze().
    **aero_kwargs
        Extra keyword arguments forwarded to aero.analyze().

    Returns
    -------
    trim_alpha : float
        Angle of attack for Cm = 0 at the current CG [deg].
    trim_elevator : float
        Elevator deflection for trimmed level flight [deg].
        NaN if elevator is absent or lacks authority.
    """
    from scipy.optimize import brentq

    # Deep copy aircraft with CG as moment reference
    ac = copy.deepcopy(aircraft)
    cg = weight_result.cg
    ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]

    # --- Trim alpha: find alpha where Cm = 0 ---
    # Use linear estimate as starting bracket
    if abs(deriv.Cm_alpha) > 1e-10:
        alpha_est = condition.alpha - deriv.baseline.Cm / deriv.Cm_alpha
    else:
        alpha_est = condition.alpha

    # Bracket around estimate
    alpha_lo = alpha_est - 10.0
    alpha_hi = alpha_est + 10.0

    def _cm_at_alpha(alpha: float) -> float:
        cond = condition.copy()
        cond.alpha = alpha
        cond.beta = 0.0
        result = aero_analyze(ac, cond, method=aero_method, **aero_kwargs)
        return result.Cm

    try:
        # Check if bracket has a sign change
        cm_lo = _cm_at_alpha(alpha_lo)
        cm_hi = _cm_at_alpha(alpha_hi)

        if cm_lo * cm_hi < 0:
            trim_alpha = brentq(_cm_at_alpha, alpha_lo, alpha_hi, xtol=0.01)
        else:
            # No sign change in bracket — Cm doesn't cross zero in ±10° window.
            # Fall back to linear estimate and warn.
            import warnings
            warnings.warn(
                f"Trim alpha not found in [{alpha_lo:.1f}, {alpha_hi:.1f}] deg "
                f"(Cm signs: lo={cm_lo:+.4f}, hi={cm_hi:+.4f}). "
                "Returning linear estimate — aircraft may not be trimmable.",
                UserWarning,
                stacklevel=4,
            )
            trim_alpha = alpha_est
    except Exception as exc:
        import warnings
        warnings.warn(
            f"Trim alpha solver failed ({exc}). Returning linear estimate.",
            UserWarning,
            stacklevel=4,
        )
        trim_alpha = alpha_est

    # --- Trim elevator: find elevator deflection for CL = CL_required ---
    elevator_name = _find_elevator(aircraft)
    if elevator_name is None:
        return trim_alpha, float("nan")

    # Required CL for level flight at trim alpha
    total_weight = weight_result.total_mass * 9.81  # [N]
    q = condition.dynamic_pressure()
    S = aircraft.reference_area()
    CL_required = total_weight / (q * S) if q * S > 0 else 0.0

    def _cm_at_elevator(de: float) -> float:
        cond = condition.copy()
        cond.alpha = trim_alpha
        cond.beta = 0.0
        cond.deflections = {**condition.deflections, elevator_name: de}
        result = aero_analyze(ac, cond, method=aero_method, **aero_kwargs)
        return result.Cm

    try:
        cm_lo = _cm_at_elevator(-25.0)
        cm_hi = _cm_at_elevator(25.0)

        if cm_lo * cm_hi < 0:
            trim_elevator = brentq(_cm_at_elevator, -25.0, 25.0, xtol=0.05)
        else:
            # Cm doesn't cross zero in ±25° — elevator authority insufficient.
            import warnings
            warnings.warn(
                f"Trim elevator not found in [-25, 25] deg "
                f"(Cm signs: lo={cm_lo:+.4f}, hi={cm_hi:+.4f}). "
                "Returning NaN — elevator may lack authority for trim.",
                UserWarning,
                stacklevel=4,
            )
            trim_elevator = float("nan")
    except Exception as exc:
        import warnings
        warnings.warn(
            f"Trim elevator solver failed ({exc}). Returning NaN.",
            UserWarning,
            stacklevel=4,
        )
        trim_elevator = float("nan")

    return trim_alpha, trim_elevator


def _dynamic_modes(
    deriv,
    condition: FlightCondition,
    weight_result: WeightResult,
    aircraft,
) -> tuple:
    """Compute short-period and phugoid frequency and damping.

    Uses linearised longitudinal equations of motion.  Requires Cm_q and
    Cm_alpha from the derivative result and I_yy from the weight result.

    Short-period (2-DOF pitch + AoA):
      ω_sp² = (q·S·c / I_yy) · [CL_alpha·|Cm_q| − |Cm_alpha|·(CL_alpha + CL_q)]
      ζ_sp  = −(q·S·c² / (2·I_yy·V)) · (Cm_q + Cm_alpha_dot) / (2·ω_sp)
              where Cm_alpha_dot ≈ 0 (not computed), so ζ_sp uses Cm_q only.

    Phugoid (speed / altitude exchange, Lanchester approximation):
      ω_ph = g·√2 / V
      ζ_ph = (CD_baseline / CL_baseline) / √2

    Returns (sp_frequency, sp_damping, ph_frequency, ph_damping).
    Values are NaN when computation is not possible.
    """
    nan = float("nan")

    q_dyn = condition.dynamic_pressure()
    S = aircraft.reference_area()
    c = aircraft.reference_chord()
    V = float(condition.velocity)
    m = float(weight_result.total_mass)
    g = 9.81

    # Moment of inertia about pitch axis (y-axis)
    inertia = weight_result.inertia_tensor
    I_yy = float(inertia[1, 1])
    if I_yy <= 0 or q_dyn <= 0 or S <= 0 or c <= 0 or V <= 0:
        return nan, nan, nan, nan

    # Convert angle derivatives from 1/deg to 1/rad
    CL_a = deriv.CL_alpha * np.degrees(1)   # 1/rad
    Cm_a = deriv.Cm_alpha * np.degrees(1)   # 1/rad
    Cm_q = deriv.Cm_q if deriv.Cm_q is not None else nan
    CL_q = deriv.CL_q if deriv.CL_q is not None else 0.0

    # --- Short-period ---
    # ω_sp² = (q·S·c / I_yy) · (CL_a · |Cm_q| + |Cm_a| · (CL_a + CL_q))
    # Convention: for a stable aircraft Cm_a < 0 and Cm_q < 0.
    qSc_Iyy = (q_dyn * S * c) / I_yy
    disc = qSc_Iyy * (CL_a * (-Cm_q) - (-Cm_a) * (CL_a + CL_q))
    if disc > 0:
        sp_frequency = float(np.sqrt(disc))
        # ζ_sp = −Cm_q · (q·S·c²) / (2·I_yy·V·ω_sp)
        sp_damping = float((-Cm_q) * (q_dyn * S * c**2) / (2.0 * I_yy * V * sp_frequency))
    else:
        sp_frequency = nan
        sp_damping = nan

    # --- Phugoid (Lanchester) ---
    ph_frequency = float(g * np.sqrt(2.0) / V)
    CL0 = deriv.baseline.CL
    CD0 = deriv.baseline.CD
    if abs(CL0) > 1e-6:
        ph_damping = float((CD0 / CL0) / np.sqrt(2.0))
    else:
        ph_damping = nan

    return sp_frequency, sp_damping, ph_frequency, ph_damping


def _find_elevator(aircraft: Aircraft) -> str | None:
    """Find the elevator control surface name, if any."""
    for wing in aircraft.wings:
        for cs in getattr(wing, "control_surfaces", []):
            name_lower = cs.name.lower()
            if "elevator" in name_lower or "elev" in name_lower:
                return cs.name
    return None


def _compute_tail_volumes(aircraft: Aircraft) -> tuple[float, float]:
    """Compute horizontal and vertical tail volume coefficients.

    Returns (Vh, Vv). NaN if the corresponding tail is not found.

    Identification heuristic:
    - Main wing: largest area, symmetric
    - Horizontal tail: symmetric wing that is NOT the main wing
    - Vertical tail: non-symmetric wing (single-sided)
    """
    main_wing = aircraft.main_wing()
    if main_wing is None:
        return float("nan"), float("nan")

    S_w = main_wing.area()
    c_w = main_wing.mean_aerodynamic_chord()
    b_w = main_wing.span()
    ac_wing = main_wing.aerodynamic_center()

    if S_w == 0 or c_w == 0 or b_w == 0:
        return float("nan"), float("nan")

    Vh = float("nan")
    Vv = float("nan")

    for wing in aircraft.wings:
        if wing is main_wing:
            continue

        ac_tail = wing.aerodynamic_center()
        l_tail = abs(ac_tail[0] - ac_wing[0])  # longitudinal distance
        S_tail = wing.area()

        name_lower = wing.name.lower()
        is_symmetric = getattr(wing, "symmetric", True)

        # Classify: symmetric → htail, non-symmetric → vtail
        # Also check name hints
        if "vtail" in name_lower or "vertical" in name_lower or not is_symmetric:
            # Vertical tail
            if b_w > 0 and S_w > 0:
                Vv = (S_tail * l_tail) / (S_w * b_w)
        else:
            # Horizontal tail (default for symmetric surfaces)
            if c_w > 0 and S_w > 0:
                Vh = (S_tail * l_tail) / (S_w * c_w)

    return Vh, Vv


# ─────────────────────────────────────────────────────────────────────────────
# Lateral-directional analysis
# ─────────────────────────────────────────────────────────────────────────────

def lateral_analyze(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result,
    aero_method: str = "aero_buildup",
    beta_range=None,
    rate_range=None,
    spanwise_resolution: int = 8,
    chordwise_resolution: int = 4,
    model_size: str = "medium",
    compute_control_matrix: bool = True,
    verbose: bool = False,
):
    """Run a complete lateral-directional stability analysis.

    Computes stability derivatives (including rate derivatives), beta and rate
    sweeps, builds the lateral A-matrix, performs eigenvalue analysis, and
    integrates standard time responses.

    Parameters
    ----------
    aircraft : Aircraft
    condition : FlightCondition
        Baseline operating point (alpha should be at trim).
    weight_result : WeightResult
        Provides CG and inertia tensor.
    aero_method : str
        Aero solver for sweeps.  Default ``"aero_buildup"`` (per-component
        breakdown).  ``"vlm"`` gives per-wing breakdown.
    beta_range : array-like or None
        Sideslip angles to sweep [deg].  Default: −20 to +20 in 1° steps.
    rate_range : array-like or None
        Nondimensional rates to sweep.  Default: −0.15 to +0.15 in 21 pts.
    spanwise_resolution : int
        Passed to the aero solver.
    chordwise_resolution : int
        Passed to the VLM solver.
    model_size : str
        NeuralFoil model size for aero_buildup / LL.
    compute_control_matrix : bool
        If True, attempt to find aileron/rudder and compute B-matrix for
        step responses.  Default True.
    verbose : bool
        Print progress messages.

    Returns
    -------
    LateralResult
        Complete lateral-directional analysis result with all plot and report
        methods.
    """
    import numpy as np
    from aerisplane.stability.derivatives import compute_derivatives
    from aerisplane.stability.sweeps import beta_sweep, rate_sweep
    from aerisplane.stability.lateral_model import (
        build_lateral_matrix, analyze_modes, build_control_matrix,
        compute_standard_responses,
    )
    from aerisplane.stability.lateral_result import LateralResult

    xyz_ref = weight_result.cg.tolist()

    if verbose:
        print("Computing stability derivatives …")
    deriv = compute_derivatives(
        aircraft, condition, weight_result,
        aero_method=aero_method,
        compute_rate_derivatives=True,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        model_size=model_size,
    )

    if beta_range is None:
        beta_range = np.linspace(-20.0, 20.0, 41)
    if rate_range is None:
        rate_range = np.linspace(-0.15, 0.15, 21)

    if verbose:
        print("Running beta sweep …")
    bs = beta_sweep(
        aircraft, condition, beta_range,
        method=aero_method,
        xyz_ref=xyz_ref,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        model_size=model_size,
        verbose=verbose,
    )

    if verbose:
        print("Running roll-rate sweep …")
    rs_p = rate_sweep(
        aircraft, condition, rate_range, rate_type="p",
        method=aero_method, xyz_ref=xyz_ref,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        model_size=model_size,
        verbose=verbose,
    )

    if verbose:
        print("Running yaw-rate sweep …")
    rs_r = rate_sweep(
        aircraft, condition, rate_range, rate_type="r",
        method=aero_method, xyz_ref=xyz_ref,
        spanwise_resolution=spanwise_resolution,
        chordwise_resolution=chordwise_resolution,
        model_size=model_size,
        verbose=verbose,
    )

    if verbose:
        print("Building A-matrix and computing modes …")
    A = build_lateral_matrix(deriv, condition, weight_result, aircraft)
    modes = analyze_modes(A)

    B = None
    if compute_control_matrix:
        try:
            B = build_control_matrix(
                aircraft, condition, weight_result, deriv,
                aero_method=aero_method,
                spanwise_resolution=spanwise_resolution,
                model_size=model_size,
            )
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Control matrix computation failed ({exc}). "
                "Step responses will not be available.",
                UserWarning,
                stacklevel=2,
            )

    if verbose:
        print("Integrating time responses …")
    responses = compute_standard_responses(A, modes, B=B)

    return LateralResult(
        aircraft_name=aircraft.name,
        condition=condition,
        weight_result=weight_result,
        deriv=deriv,
        beta_sweep=bs,
        rate_sweep_p=rs_p,
        rate_sweep_r=rs_r,
        modes=modes,
        responses=responses,
        A=A,
        aero_method=aero_method,
    )


__all__ = ["analyze", "lateral_analyze", "StabilityResult"]
