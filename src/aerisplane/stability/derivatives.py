"""Numerical stability derivative computation via central finite differences.

Computes longitudinal and lateral-directional stability derivatives using
5 aero evaluations. Step sizes follow the AerisPlane spec:
  d_alpha = 0.5 deg, d_beta = 1.0 deg.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from aerisplane.aero import analyze as aero_analyze
from aerisplane.aero.result import AeroResult
from aerisplane.core.aircraft import Aircraft
from aerisplane.core.flight_condition import FlightCondition
from aerisplane.weights.result import WeightResult


# Finite-difference step sizes [degrees]
D_ALPHA = 0.5
D_BETA = 1.0


@dataclass
class DerivativeResult:
    """Raw stability derivative outputs from finite-difference computation.

    All derivatives are per degree (1/deg).
    """

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


def compute_derivatives(
    aircraft: Aircraft,
    condition: FlightCondition,
    weight_result: WeightResult,
    aero_method: str = "vlm",
    **aero_kwargs,
) -> DerivativeResult:
    """Compute stability derivatives via central finite differences.

    Runs 5 aero evaluations with the moment reference set to the CG:
      1. baseline  (alpha,          beta         )
      2. alpha_pos (alpha + d_alpha, beta         )
      3. alpha_neg (alpha - d_alpha, beta         )
      4. beta_pos  (alpha,          beta + d_beta )
      5. beta_neg  (alpha,          beta - d_beta )

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

    # --- Run 5 aero evaluations ---
    def _run(alpha: float, beta: float) -> AeroResult:
        cond = condition.copy()
        cond.alpha = alpha
        cond.beta = beta
        return aero_analyze(ac, cond, method=aero_method, **aero_kwargs)

    alpha_0 = condition.alpha
    beta_0 = condition.beta

    baseline = _run(alpha_0, beta_0)
    alpha_pos = _run(alpha_0 + D_ALPHA, beta_0)
    alpha_neg = _run(alpha_0 - D_ALPHA, beta_0)
    beta_pos = _run(alpha_0, beta_0 + D_BETA)
    beta_neg = _run(alpha_0, beta_0 - D_BETA)

    # --- Central differences ---
    two_da = 2.0 * D_ALPHA  # denominator for alpha derivatives [deg]
    two_db = 2.0 * D_BETA   # denominator for beta derivatives [deg]

    CL_alpha = (alpha_pos.CL - alpha_neg.CL) / two_da   # 1/deg
    Cm_alpha = (alpha_pos.Cm - alpha_neg.Cm) / two_da    # 1/deg
    Cl_beta = (beta_pos.Cl - beta_neg.Cl) / two_db       # 1/deg
    Cn_beta = (beta_pos.Cn - beta_neg.Cn) / two_db       # 1/deg

    # --- Neutral point ---
    # NP is the point where dCm/dCL = 0 (moments taken about NP).
    # With moments about CG:  dCm/dalpha = dCm_np/dalpha - (x_np - x_cg)/c * dCL/dalpha
    # At neutral point dCm_np/dalpha = 0, so:
    #   x_np = x_cg - (dCm/dCL) * c_ref
    # where dCm/dCL = (dCm/dalpha) / (dCL/dalpha)
    cg_x = float(cg[0])
    c_ref = mac

    if abs(CL_alpha) > 1e-12:
        dCm_dCL = Cm_alpha / CL_alpha
        neutral_point = cg_x - dCm_dCL * c_ref
    else:
        # CL_alpha ~ 0 — cannot determine NP
        neutral_point = cg_x

    # Static margin: positive = stable (NP aft of CG)
    if mac > 0:
        static_margin = (neutral_point - cg_x) / mac
    else:
        static_margin = 0.0

    return DerivativeResult(
        CL_alpha=CL_alpha,
        Cm_alpha=Cm_alpha,
        Cl_beta=Cl_beta,
        Cn_beta=Cn_beta,
        neutral_point=neutral_point,
        static_margin=static_margin,
        cg_x=cg_x,
        mac=mac,
        mac_le_x=mac_le_x,
        baseline=baseline,
    )
