# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/library/aerodynamics/viscous.py
"""Viscous aerodynamic correlations: skin friction, cylinder drag."""
from __future__ import annotations

import numpy as np
from aerisplane.aero._np_compat import blend


def Cd_cylinder(
    Re_D: float | np.ndarray,
    mach: float | np.ndarray = 0.0,
    include_mach_effects: bool = True,
    subcritical_only: bool = False,
) -> float | np.ndarray:
    """Drag coefficient of a cylinder in crossflow as a function of Re and Mach.

    Args:
        Re_D: Reynolds number, referenced to diameter.
        mach: Mach number.
        include_mach_effects: If False, assumes Mach = 0.
        subcritical_only: If True, model only subcritical (Re < 300k) regime.

    Returns: Drag coefficient.
    """
    csigc = 5.5766722118597247
    csigh = 23.7460859935990563
    csub0 = -0.6989492360435040
    csub1 = 1.0465189382830078
    csub2 = 0.7044228755898569
    csub3 = 0.0846501115443938
    csup0 = -0.0823564417206403
    csupc = 6.8020230357616764
    csuph = 9.9999999999999787
    csupscl = -0.4570690347113859

    x = np.log10(np.abs(Re_D) + 1e-16)

    if subcritical_only:
        Cd_mach_0 = 10 ** (csub0 * x + csub1) + csub2 + csub3 * x
    else:
        log10_Cd = (np.log10(10 ** (csub0 * x + csub1) + csub2 + csub3 * x)) * (
            1 - 1 / (1 + np.exp(-csigh * (x - csigc)))
        ) + (csup0 + csupscl / csuph * np.log(np.exp(csuph * (csupc - x)) + 1)) * (
            1 / (1 + np.exp(-csigh * (x - csigc)))
        )
        Cd_mach_0 = 10**log10_Cd

    if include_mach_effects:
        m = mach
        p = {
            "a_sub": 0.03458900259594298,
            "a_sup": -0.7129528087049688,
            "cd_sub": 1.163206940186374,
            "cd_sup": 1.2899213533122527,
            "s_sub": 3.436601777569716,
            "s_sup": -1.37123096976983,
            "trans": 1.022819211244295,
            "trans_str": 19.017600596069848,
        }

        Cd_over_Cd_mach_0 = (
            blend(
                p["trans_str"] * (m - p["trans"]),
                p["cd_sup"] + np.exp(p["a_sup"] + p["s_sup"] * (m - p["trans"])),
                p["cd_sub"] + np.exp(p["a_sub"] + p["s_sub"] * (m - p["trans"])),
            )
            / 1.1940010047391572
        )

        Cd = Cd_mach_0 * Cd_over_Cd_mach_0
    else:
        Cd = Cd_mach_0

    return Cd


def Cf_flat_plate(
    Re_L: float | np.ndarray,
    method: str = "hybrid-sharpe-convex",
) -> float | np.ndarray:
    """Mean skin friction coefficient over a flat plate.

    Don't forget to double it (two sides) for a drag coefficient.

    Args:
        Re_L: Reynolds number normalized to plate length.
        method: One of "blasius", "turbulent", "hybrid-cengel",
                "hybrid-schlichting", "hybrid-sharpe-convex",
                "hybrid-sharpe-nonconvex".

    Returns: Skin friction coefficient.
    """
    from aerisplane.aero._np_compat import softmax

    Re_L = np.abs(Re_L)

    if method == "blasius":
        return 1.328 / Re_L**0.5
    elif method == "turbulent":
        return 0.074 / Re_L ** (1 / 5)
    elif method == "hybrid-cengel":
        return 0.074 / Re_L ** (1 / 5) - 1742 / Re_L
    elif method == "hybrid-schlichting":
        return 0.02666 * Re_L**-0.139
    elif method == "hybrid-sharpe-convex":
        return softmax(
            Cf_flat_plate(Re_L, method="blasius"),
            Cf_flat_plate(Re_L, method="hybrid-schlichting"),
            hardness=1e3,
        )
    elif method == "hybrid-sharpe-nonconvex":
        return softmax(
            Cf_flat_plate(Re_L, method="blasius"),
            Cf_flat_plate(Re_L, method="hybrid-cengel"),
            hardness=1e3,
        )
    else:
        raise ValueError(
            f"{method!r} is not a valid method for Cf_flat_plate(). "
            f"Valid options are: 'blasius', 'turbulent', 'hybrid-cengel', "
            f"'hybrid-schlichting', 'hybrid-sharpe-convex', 'hybrid-sharpe-nonconvex'."
        )


def fuselage_upsweep_drag_area(
    upsweep_angle_rad: float,
    fuselage_xsec_area_max: float,
) -> float:
    """Drag area of aft fuselage upsweep (Raymer Eq. 12.36).

    Args:
        upsweep_angle_rad: Aft fuselage upsweep angle relative to centerline [rad].
        fuselage_xsec_area_max: Maximum fuselage cross-section area [m^2].

    Returns: Drag area D/q [m^2].
    """
    return 3.83 * np.abs(upsweep_angle_rad) ** 2.5 * fuselage_xsec_area_max
