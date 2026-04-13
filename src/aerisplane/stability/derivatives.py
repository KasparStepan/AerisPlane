"""Numerical stability derivative computation via central finite differences.

Computes longitudinal and lateral-directional stability derivatives.
Static derivatives (α, β) use 5 aero evaluations. Rate derivatives
(p, q, r) add 6 more evaluations (2 per rate) and are optional.

Step sizes:
  d_alpha = 0.5 deg, d_beta = 1.0 deg
  d_p, d_q, d_r: 0.001 in nondimensional rate (pb/2V, qc/2V, rb/2V)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aerisplane.aero import analyze as aero_analyze
from aerisplane.aero.result import AeroResult
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.weights.result import WeightResult


# Finite-difference step sizes
D_ALPHA = 0.5    # degrees
D_BETA = 1.0     # degrees
D_RATE = 0.001   # nondimensional rate hat (pb/2V etc.)


@dataclass
class DerivativeResult:
    """Raw stability derivative outputs from finite-difference computation.

    Static derivatives (CL_alpha, Cm_alpha, Cl_beta, Cn_beta) are always
    computed.  Rate derivatives are optional — None if not requested.

    All angle derivatives are per degree (1/deg).
    Rate derivatives are per unit nondimensional rate (pb/2V, qc/2V, rb/2V).
    """

    # Static derivatives [1/deg]
    CL_alpha: float
    Cm_alpha: float
    Cl_beta: float
    Cn_beta: float

    neutral_point: float   # x-position [m]
    static_margin: float   # fraction of MAC

    # Reference values
    cg_x: float
    mac: float
    mac_le_x: float

    # Baseline aero at the analysis condition
    baseline: AeroResult

    # Side force due to sideslip (always computed along with Cl_beta/Cn_beta)
    CY_beta: float = 0.0   # [1/deg]

    # Rate derivatives (None if compute_rate_derivatives=False)
    # Pitch rate q: longitudinal damping
    CL_q: Optional[float] = None   # lift due to pitch rate
    Cm_q: Optional[float] = None   # pitch moment due to pitch rate (pitch damping)
    # Roll rate p: roll damping and adverse yaw
    Cl_p: Optional[float] = None   # roll moment due to roll rate (roll damping)
    Cn_p: Optional[float] = None   # yaw moment due to roll rate (adverse yaw)
    CY_p: Optional[float] = None   # side force due to roll rate
    # Yaw rate r: yaw damping and roll due to yaw
    Cn_r: Optional[float] = None   # yaw moment due to yaw rate (yaw damping)
    Cl_r: Optional[float] = None   # roll moment due to yaw rate
    CY_r: Optional[float] = None   # side force due to yaw rate


def compute_derivatives(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    aero_method: str = "vlm",
    compute_rate_derivatives: bool = False,
    **aero_kwargs,
) -> DerivativeResult:
    """Compute stability derivatives via central finite differences.

    Static derivatives always use 5 aero evaluations:
      1. baseline  (alpha,           beta        )
      2. alpha_pos (alpha + d_alpha, beta        )
      3. alpha_neg (alpha - d_alpha, beta        )
      4. beta_pos  (alpha,           beta + d_beta)
      5. beta_neg  (alpha,           beta - d_beta)

    When compute_rate_derivatives=True, 6 additional evaluations are run
    (±delta for each of p, q, r).

    Parameters
    ----------
    aircraft : Aircraft
        Aircraft geometry definition.
    condition : FlightCondition
        Operating point (velocity, altitude, alpha, beta).
    weight_result : WeightResult
        Weight analysis result — provides CG position.
    aero_method : str
        Aero solver to use (default "vlm").
    compute_rate_derivatives : bool
        If True, also compute p, q, r rate derivatives (6 extra aero calls).
        Default False.
    **aero_kwargs
        Additional keyword arguments passed to aero.analyze().

    Returns
    -------
    DerivativeResult
        Stability derivatives, neutral point, and static margin.
    """
    # Deep copy aircraft to avoid mutating the caller's object
    ac = copy.deepcopy(aircraft)

    # Set moment reference to CG
    cg = weight_result.cg
    ac.xyz_ref = [float(cg[0]), float(cg[1]), float(cg[2])]

    # Reference geometry from the main wing
    main_wing = ac.main_wing()
    mac = ac.reference_chord()
    mac_le_x = float(main_wing.mean_aerodynamic_chord_le()[0]) if main_wing else 0.0

    V = float(condition.velocity)
    b_ref = ac.reference_span()
    c_ref = mac

    def _run_at(alpha: float, beta: float,
                p: float = 0.0, q: float = 0.0, r: float = 0.0) -> AeroResult:
        cond = condition.copy()
        cond.alpha = alpha
        cond.beta = beta
        cond.p = p
        cond.q = q
        cond.r = r
        return aero_analyze(ac, cond, method=aero_method, **aero_kwargs)

    alpha_0 = float(condition.alpha)
    beta_0 = float(condition.beta)
    p_0 = float(getattr(condition, "p", 0.0))
    q_0 = float(getattr(condition, "q", 0.0))
    r_0 = float(getattr(condition, "r", 0.0))

    # --- Static derivatives: 5 evaluations ---
    baseline  = _run_at(alpha_0, beta_0, p_0, q_0, r_0)
    alpha_pos = _run_at(alpha_0 + D_ALPHA, beta_0, p_0, q_0, r_0)
    alpha_neg = _run_at(alpha_0 - D_ALPHA, beta_0, p_0, q_0, r_0)
    beta_pos  = _run_at(alpha_0, beta_0 + D_BETA, p_0, q_0, r_0)
    beta_neg  = _run_at(alpha_0, beta_0 - D_BETA, p_0, q_0, r_0)

    CL_alpha = (alpha_pos.CL - alpha_neg.CL) / (2.0 * D_ALPHA)   # 1/deg
    Cm_alpha = (alpha_pos.Cm - alpha_neg.Cm) / (2.0 * D_ALPHA)    # 1/deg
    Cl_beta  = (beta_pos.Cl  - beta_neg.Cl)  / (2.0 * D_BETA)    # 1/deg
    Cn_beta  = (beta_pos.Cn  - beta_neg.Cn)  / (2.0 * D_BETA)    # 1/deg
    CY_beta  = (beta_pos.CY  - beta_neg.CY)  / (2.0 * D_BETA)    # 1/deg

    # --- Neutral point ---
    # x_np = x_cg - (dCm/dCL) * c_ref,  where dCm/dCL = Cm_alpha / CL_alpha
    cg_x = float(cg[0])

    if abs(CL_alpha) > 1e-12:
        neutral_point = cg_x - (Cm_alpha / CL_alpha) * c_ref
    else:
        neutral_point = cg_x  # indeterminate

    static_margin = (neutral_point - cg_x) / mac if mac > 0 else 0.0

    # --- Rate derivatives (optional): 6 more evaluations ---
    CL_q = Cm_q = None
    Cl_p = Cn_p = CY_p = None
    Cn_r = Cl_r = CY_r = None

    if compute_rate_derivatives and b_ref > 0 and c_ref > 0:
        # Dimensional step for each rate: D_RATE in nondimensional hat →
        #   delta_q = D_RATE * (2V/c),  delta_p = D_RATE * (2V/b), etc.
        dq = D_RATE * (2.0 * V) / c_ref   # rad/s
        dp = D_RATE * (2.0 * V) / b_ref   # rad/s
        dr = D_RATE * (2.0 * V) / b_ref   # rad/s

        q_pos = _run_at(alpha_0, beta_0, p_0, q_0 + dq, r_0)
        q_neg = _run_at(alpha_0, beta_0, p_0, q_0 - dq, r_0)
        p_pos = _run_at(alpha_0, beta_0, p_0 + dp, q_0, r_0)
        p_neg = _run_at(alpha_0, beta_0, p_0 - dp, q_0, r_0)
        r_pos = _run_at(alpha_0, beta_0, p_0, q_0, r_0 + dr)
        r_neg = _run_at(alpha_0, beta_0, p_0, q_0, r_0 - dr)

        # Derivatives w.r.t. nondimensional rate hat (dimensionless)
        CL_q = (q_pos.CL - q_neg.CL) / (2.0 * D_RATE)
        Cm_q = (q_pos.Cm - q_neg.Cm) / (2.0 * D_RATE)
        Cl_p = (p_pos.Cl - p_neg.Cl) / (2.0 * D_RATE)
        Cn_p = (p_pos.Cn - p_neg.Cn) / (2.0 * D_RATE)
        CY_p = (p_pos.CY - p_neg.CY) / (2.0 * D_RATE)
        Cn_r = (r_pos.Cn - r_neg.Cn) / (2.0 * D_RATE)
        Cl_r = (r_pos.Cl - r_neg.Cl) / (2.0 * D_RATE)
        CY_r = (r_pos.CY - r_neg.CY) / (2.0 * D_RATE)

    return DerivativeResult(
        CL_alpha=CL_alpha,
        Cm_alpha=Cm_alpha,
        Cl_beta=Cl_beta,
        Cn_beta=Cn_beta,
        CY_beta=CY_beta,
        neutral_point=neutral_point,
        static_margin=static_margin,
        cg_x=cg_x,
        mac=mac,
        mac_le_x=mac_le_x,
        baseline=baseline,
        CL_q=CL_q,
        Cm_q=Cm_q,
        Cl_p=Cl_p,
        Cn_p=Cn_p,
        CY_p=CY_p,
        Cn_r=Cn_r,
        Cl_r=Cl_r,
        CY_r=CY_r,
    )
