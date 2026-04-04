# src/aerisplane/structures/checks.py
"""Structural margin-of-safety functions for wing sizing.

All margin functions return:  MoS = capacity / demand - 1
Positive = safe, negative = failed, zero = exactly at limit.
"""
from __future__ import annotations

import numpy as np

from aerisplane.core.airfoil import Airfoil
from aerisplane.core.structures import Spar
from aerisplane.structures.section import spar_fits_in_airfoil


def bending_margin(spar: Spar, bending_moment: float) -> float:
    """Margin of safety against bending yield at a given moment [N*m].

    MoS = sigma_yield / sigma_bending - 1

    Parameters
    ----------
    spar : Spar
    bending_moment : float
        Applied bending moment [N*m].

    Returns
    -------
    float
        Positive = safe, negative = failed.
    """
    return spar.margin_of_safety(bending_moment)


def shear_margin(spar: Spar, shear_force: float) -> float:
    """Margin of safety against shear yield (thin annular tube).

    Uses a conservative estimate: tau_max = V / A_wall.
    Shear yield from von Mises: tau_yield = sigma_yield / sqrt(3).

    Parameters
    ----------
    spar : Spar
    shear_force : float
        Applied shear force [N].

    Returns
    -------
    float
    """
    if shear_force == 0.0:
        return float("inf")
    A = spar.section.area()
    if A <= 0.0:
        return float("inf")
    tau_applied = abs(shear_force) / A
    tau_yield = spar.material.yield_strength / np.sqrt(3.0)
    return tau_yield / tau_applied - 1.0


def buckling_margin(spar: Spar, bending_moment: float) -> float:
    """Margin of safety against local shell buckling (Timoshenko).

    sigma_cr = 0.6 * E * t_wall / R_outer

    Parameters
    ----------
    spar : Spar
    bending_moment : float
        Applied bending moment [N*m].

    Returns
    -------
    float
    """
    if bending_moment == 0.0:
        return float("inf")
    R = spar.section.outer_diameter / 2.0
    t = spar.section.wall_thickness
    if R <= 0.0 or t <= 0.0:
        return float("inf")
    sigma_cr = 0.6 * spar.material.E * t / R
    sigma_applied = spar.max_bending_stress(bending_moment)
    if sigma_applied <= 0.0:
        return float("inf")
    return sigma_cr / sigma_applied - 1.0


def fits_in_airfoil(airfoil: Airfoil, spar: Spar, chord: float) -> bool:
    """True if the spar outer diameter fits in the airfoil at its position.

    Delegates to aerisplane.structures.section.spar_fits_in_airfoil.
    """
    return spar_fits_in_airfoil(airfoil, spar, chord)


def divergence_speed(
    GJ_root: float,
    cl_alpha_per_rad: float,
    wing_area: float,
    density: float,
    e: float,
) -> float:
    """Torsional divergence speed [m/s].

    V_div = sqrt(2 * GJ / (rho * a * e * S))

    where e = x_AC - x_SC [m] (positive when AC is behind SC).
    Returns inf when e <= 0 (no divergence risk).

    Parameters
    ----------
    GJ_root : float
        Torsional stiffness at root [N*m^2/rad].
    cl_alpha_per_rad : float
        Lift curve slope [1/rad].
    wing_area : float
        Wing planform area [m^2].
    density : float
        Air density [kg/m^3].
    e : float
        Distance from aerodynamic centre to shear centre [m].
        Positive = AC behind SC (divergence possible).

    Returns
    -------
    float
    """
    if e <= 0.0 or GJ_root <= 0.0:
        return float("inf")
    denom = density * cl_alpha_per_rad * e * wing_area
    if denom <= 0.0:
        return float("inf")
    return float(np.sqrt(2.0 * GJ_root / denom))
