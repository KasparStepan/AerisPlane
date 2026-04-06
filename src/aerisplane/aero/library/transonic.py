# Adapted from AeroSandbox v4.2.9 by Peter Sharpe (MIT License)
# https://github.com/peterdsharpe/AeroSandbox
# Original: aerosandbox/library/aerodynamics/transonic.py
#           aerosandbox/modeling/splines/hermite.py  (cubic_hermite_patch)
"""Transonic aerodynamic correlations: wave drag models."""
from __future__ import annotations

import numpy as np
from aerisplane.aero._np_compat import softmax, cosd, blend


# ── Hermite cubic patch (inlined from aerosandbox.modeling.splines.hermite) ──

def _cubic_hermite_patch(
    x: float | np.ndarray,
    x_a: float,
    x_b: float,
    f_a: float,
    f_b: float,
    dfdx_a: float,
    dfdx_b: float,
) -> float | np.ndarray:
    """Cubic Hermite polynomial patch through (x_a, f_a) and (x_b, f_b).

    Matches function values and first derivatives at both endpoints.
    Extrapolates beyond [x_a, x_b] as a polynomial.
    """
    dx = x_b - x_a
    t = (x - x_a) / dx
    return (
        (t**3) * f_b
        + (t**2 * (1 - t)) * (3 * f_b - dfdx_b * dx)
        + (t * (1 - t) ** 2) * (3 * f_a + dfdx_a * dx)
        + ((1 - t) ** 3) * f_a
    )


# ── Wave drag functions ───────────────────────────────────────────────────────

def sears_haack_drag_from_volume(volume: float, length: float) -> float:
    """Idealized wave drag area (D/q) for a Sears-Haack body.

    Assumes linearised supersonic (Prandtl-Glauert) flow.

    Args:
        volume: Body volume [m^3].
        length: Body length [m].

    Returns: Drag area D/q [m^2].
    """
    return 128 * volume**2 / (np.pi * length**4)


def mach_crit_Korn(
    CL: float | np.ndarray,
    t_over_c: float | np.ndarray,
    sweep: float | np.ndarray = 0.0,
    kappa_A: float = 0.95,
) -> float | np.ndarray:
    """Critical Mach from the Korn equation (Mason, Configuration Aerodynamics §7.5.2).

    Args:
        CL: Sectional lift coefficient.
        t_over_c: Thickness-to-chord ratio.
        sweep: Sweep angle [deg].
        kappa_A: Airfoil technology factor (0.95 supercritical, 0.87 NACA 6-series).

    Returns: Critical Mach number.
    """
    smooth_abs_CL = softmax(CL, -CL, hardness=10)
    M_dd = (
        kappa_A / cosd(sweep)
        - t_over_c / cosd(sweep) ** 2
        - smooth_abs_CL / (10 * cosd(sweep) ** 3)
    )
    return M_dd - (0.1 / 80) ** (1 / 3)


def Cd_wave_Korn(
    Cl: float | np.ndarray,
    t_over_c: float | np.ndarray,
    mach: float | np.ndarray,
    sweep: float | np.ndarray = 0.0,
    kappa_A: float = 0.95,
) -> float | np.ndarray:
    """Wave drag from the Korn equation (Mason, Configuration Aerodynamics §7.5.2).

    Args:
        Cl: Sectional lift coefficient.
        t_over_c: Thickness-to-chord ratio.
        mach: Freestream Mach number.
        sweep: Sweep angle [deg].
        kappa_A: Airfoil technology factor.

    Returns: Wave drag coefficient.
    """
    smooth_abs_Cl = softmax(Cl, -Cl, hardness=10)
    mach = np.fmax(mach, 0)
    Mdd = (
        kappa_A / cosd(sweep)
        - t_over_c / cosd(sweep) ** 2
        - smooth_abs_Cl / (10 * cosd(sweep) ** 3)
    )
    Mcrit = Mdd - (0.1 / 80) ** (1 / 3)
    return np.where(mach > Mcrit, 20 * (mach - Mcrit) ** 4, 0)


def approximate_CD_wave(
    mach: float | np.ndarray,
    mach_crit: float | np.ndarray,
    CD_wave_at_fully_supersonic: float | np.ndarray,
) -> float | np.ndarray:
    """Approximate transonic wave drag coefficient.

    Valid from Mach 0 to roughly Mach 2–3. Combines methodology from Raymer
    (Aircraft Design §12.5.10) and W.H. Mason (Configuration Aerodynamics Ch. 7).

    Args:
        mach: Operating Mach number.
        mach_crit: Critical Mach number of the body.
        CD_wave_at_fully_supersonic: Wave drag coefficient at M=1.2 (first fully
            supersonic condition). Derive from Sears-Haack + efficiency markup.

    Returns: Wave drag coefficient (same reference area as CD_wave_at_fully_supersonic).
    """
    mach_crit_max = 1 - (0.1 / 80) ** (1 / 3)
    mach_crit = -softmax(-mach_crit, -mach_crit_max, hardness=50)
    mach_dd = mach_crit + (0.1 / 80) ** (1 / 3)

    return CD_wave_at_fully_supersonic * np.where(
        mach < mach_crit,
        0,
        np.where(
            mach < mach_dd,
            20 * (mach - mach_crit) ** 4,
            np.where(
                mach < 1.05,
                _cubic_hermite_patch(
                    mach,
                    x_a=mach_dd, x_b=1.05,
                    f_a=20 * (0.1 / 80) ** (4 / 3), f_b=1,
                    dfdx_a=0.1, dfdx_b=8,
                ),
                np.where(
                    mach < 1.2,
                    _cubic_hermite_patch(
                        mach, x_a=1.05, x_b=1.2,
                        f_a=1, f_b=1,
                        dfdx_a=8, dfdx_b=-4,
                    ),
                    blend(
                        switch=4 * 2 * (mach - 1.2) / (1.2 - 0.8),
                        value_switch_high=0.8,
                        value_switch_low=1.2,
                    ),
                ),
            ),
        ),
    )
