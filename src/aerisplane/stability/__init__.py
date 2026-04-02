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
    """Find trim alpha (Cm=0) and trim elevator for level flight.

    Returns (trim_alpha, trim_elevator). trim_elevator is NaN if no
    elevator control surface is found.
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


__all__ = ["analyze", "StabilityResult"]
