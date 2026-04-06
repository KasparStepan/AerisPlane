# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/library/aerodynamics/inviscid.py
"""Inviscid aerodynamic correlations: lift-curve slope, Oswald efficiency."""
from __future__ import annotations

import numpy as np
from aerisplane.aero._np_compat import softmax, tand


def induced_drag(
    lift: float | np.ndarray,
    span: float | np.ndarray,
    dynamic_pressure: float | np.ndarray,
    oswalds_efficiency: float | np.ndarray = 1.0,
) -> float | np.ndarray:
    """Induced drag of a lifting planar wing.

    Args:
        lift: Lift force [N].
        span: Wing span [m].
        dynamic_pressure: Dynamic pressure [Pa].
        oswalds_efficiency: Oswald's efficiency factor [-].

    Returns: Induced drag force [N].
    """
    return lift**2 / (dynamic_pressure * np.pi * span**2 * oswalds_efficiency)


def oswalds_efficiency(
    taper_ratio: float,
    aspect_ratio: float,
    sweep: float = 0.0,
    fuselage_diameter_to_span_ratio: float = 0.0,
    method: str = "nita_scholz",
) -> float:
    """Oswald's efficiency factor for a planar, tapered, swept wing.

    Based on Nita & Scholz (2012), "Estimating the Oswald Factor from Basic
    Aircraft Geometrical Parameters", Hamburg Univ. of Applied Sciences.

    Only valid for backwards-swept wings (0 <= sweep < 90 deg).

    Args:
        taper_ratio: tip_chord / root_chord [-].
        aspect_ratio: b^2 / S [-].
        sweep: Quarter-chord sweep angle [deg].
        fuselage_diameter_to_span_ratio: d_fus / b [-].
        method: "nita_scholz" or "kroo".

    Returns: Oswald's efficiency factor [-].
    """
    sweep = np.clip(sweep, 0, 90)

    def f(l):
        return 0.0524 * l**4 - 0.15 * l**3 + 0.1659 * l**2 - 0.0706 * l + 0.0119

    delta_lambda = -0.357 + 0.45 * np.exp(-0.0375 * sweep)
    e_theo = 1 / (1 + f(taper_ratio - delta_lambda) * aspect_ratio)

    k_e_F = 1 - 2 * fuselage_diameter_to_span_ratio**2
    k_e_D0 = np.mean([0.873, 0.864, 0.804, 0.804])
    k_e_M = 1

    if method == "nita_scholz":
        return e_theo * k_e_F * k_e_D0 * k_e_M
    elif method == "kroo":
        from aerisplane.aero.library.viscous import Cf_flat_plate
        Q = 1 / (e_theo * k_e_F)
        P = 0.38 * Cf_flat_plate(Re_L=1e6)
        return 1 / (Q + P * np.pi * aspect_ratio)
    else:
        raise ValueError(f"Unknown method {method!r}.")


def CL_over_Cl(
    aspect_ratio: float,
    mach: float = 0.0,
    sweep: float = 0.0,
    Cl_is_compressible: bool = True,
) -> float:
    """Ratio of 3-D wing CL to 2-D section Cl (Raymer DATCOM formulation).

    Args:
        aspect_ratio: Wing aspect ratio b^2/S.
        mach: Freestream Mach number.
        sweep: Quarter-chord sweep at locus of thickest points [deg].
        Cl_is_compressible: If True, Cl already includes compressibility effects.

    Returns: CL_3D / CL_2D.
    """
    prandtl_glauert_beta_squared_ideal = 1 - mach**2
    beta_squared = softmax(
        prandtl_glauert_beta_squared_ideal,
        -prandtl_glauert_beta_squared_ideal,
        hardness=3.0,
    )

    eta = 1.0
    CL_ratio = aspect_ratio / (
        2
        + (
            4
            + (aspect_ratio**2 * beta_squared / eta**2)
            + (tand(sweep) * aspect_ratio / eta) ** 2
        )
        ** 0.5
    )

    if Cl_is_compressible:
        CL_ratio = CL_ratio * beta_squared**0.5

    return CL_ratio
